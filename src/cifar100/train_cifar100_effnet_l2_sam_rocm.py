# train_cifar100_effnet_l2_sam_rocm.py
import argparse
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import time

# ======================= SAM (inline) =======================
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (Foret et al., 2021)
    Usage:
      loss.backward()
      optimizer.first_step(zero_grad=True)
      loss2.backward()
      optimizer.second_step(zero_grad=True)
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, "rho must be non-negative"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        self.base_optimizer = base_optimizer(params, **kwargs)
        super().__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (p**2 if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, *args, **kwargs):
        raise RuntimeError("Use first_step()/second_step() with SAM.")

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = torch.norm(p.detach(), p=2) if group["adaptive"] else 1.0
                norms.append(torch.norm(p.grad.detach() * scale, p=2))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

# ======================= Utils =======================
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    B = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / B)).item())
    return res

class CrossEntropyLabelSmoothing(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    def forward(self, logits, target):
        if self.smoothing <= 0:
            return F.cross_entropy(logits, target)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(true_dist * log_probs).sum(dim=1).mean()

class CosineLRScheduler:
    """Per-iteration cosine LR with warmup."""
    def __init__(self, optimizer, base_lr, min_lr, warmup_epochs, total_epochs, iters_per_epoch):
        self.opt = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.iters_per_epoch = iters_per_epoch
        self.t = 0  # global step
    def step(self, epoch_idx: int):
        self.t += 1
        current_epoch = epoch_idx + self.t / max(1, self.iters_per_epoch)
        if current_epoch < self.warmup_epochs:
            lr = self.base_lr * (current_epoch / max(1e-8, self.warmup_epochs))
        else:
            progress = (current_epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr

# ======================= Data =======================
def get_cifar100_loaders(img_size: int, batch_size: int, workers: int) -> Tuple[DataLoader, DataLoader]:
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=max(64, batch_size), shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, test_loader

# ======================= Train / Eval =======================
def train_step_sam_iter(model, images, targets, criterion, optimizer, device, channels_last=False, amp_dtype=None):
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if channels_last:
        images = images.to(memory_format=torch.channels_last)

    ac = (torch.autocast(device_type="cuda", dtype=amp_dtype)
          if amp_dtype is not None else torch.autocast(enabled=False, device_type="cuda"))

    # 1st pass
    with ac:
        outputs = model(images)
        loss = criterion(outputs, targets)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # 2nd pass at perturbed weights
    with ac:
        outputs2 = model(images)
        loss2 = criterion(outputs2, targets)
    loss2.backward()
    optimizer.second_step(zero_grad=True)

    acc1, acc5 = accuracy(outputs2.detach(), targets, topk=(1, 5))
    return loss2.item(), acc1, acc5

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, channels_last=False, amp_dtype=None):
    model.eval()
    total_loss, total_acc1, total_acc5, n = 0.0, 0.0, 0.0, 0
    ac = (torch.autocast(device_type="cuda", dtype=amp_dtype)
          if amp_dtype is not None else torch.autocast(enabled=False, device_type="cuda"))

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        with ac:
            outputs = model(images)
            loss = criterion(outputs, targets)
        B = images.size(0)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        total_loss += loss.item() * B
        total_acc1 += acc1 * B
        total_acc5 += acc5 * B
        n += B
    return total_loss / n, total_acc1 / n, total_acc5 / n

# ======================= Main =======================
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model', type=str, default='tf_efficientnet_l2_ns', help='timm model')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--warmup-epochs', type=float, default=5.0)
    ap.add_argument('--lr', type=float, default=0.2)
    ap.add_argument('--min-lr', type=float, default=1e-4)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--rho', type=float, default=0.05, help='SAM neighborhood')
    ap.add_argument('--smoothing', type=float, default=0.1)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--channels-last', action='store_true')
    ap.add_argument('--checkpoint', type=str, default='best_cifar100_effnet_l2_sam.pth')
    ap.add_argument('--amp', type=str, choices=['off', 'bf16'], default='bf16', help='mixed precision (bf16 good on RDNA3)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemError("CUDA device (ROCm) not available. Install ROCm PyTorch and retry.")
    device = "cuda"
    print("Using device:", device, "| GPU:", torch.cuda.get_device_name(0))

    # Data
    train_loader, test_loader = get_cifar100_loaders(args.img_size, args.batch_size, args.workers)

    # Model (pretrained head -> 100 classes)
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=100,
        drop_rate=0.2,
        drop_path_rate=0.2
    ).to(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # Optimizer: SAM + SGD
    base_optimizer = torch.optim.SGD
    optimizer = SAM(
        model.parameters(),
        base_optimizer=base_optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rho=args.rho,
        nesterov=True,
    )

    # LR scheduler
    iters_per_epoch = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        iters_per_epoch=iters_per_epoch
    )

    # Loss
    criterion = CrossEntropyLabelSmoothing(num_classes=100, smoothing=args.smoothing)

    # AMP
    amp_dtype = torch.bfloat16 if args.amp == 'bf16' else None
    if amp_dtype is not None:
        print("Using AMP: bfloat16")

    best_top1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        run_loss = run_top1 = run_top5 = n = 0.0

        for i, (images, targets) in enumerate(train_loader):
            scheduler.step(epoch)
            loss, acc1, acc5 = train_step_sam_iter(
                model, images, targets, criterion, optimizer, device, args.channels_last, amp_dtype=amp_dtype
            )
            B = images.size(0)
            run_loss += loss * B
            run_top1 += acc1 * B
            run_top5 += acc5 * B
            n += B

            if (i + 1) % 50 == 0 or (i + 1) == iters_per_epoch:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{args.epochs} | Iter {i+1}/{iters_per_epoch} | "
                      f"LR {lr_now:.5f} | loss {run_loss/n:.4f} | top1 {run_top1/n:.2f} | top5 {run_top5/n:.2f}")

        val_loss, val_top1, val_top5 = evaluate(model, test_loader, criterion, device, args.channels_last, amp_dtype=amp_dtype)
        print(f"[VAL] Epoch {epoch+1}: loss {val_loss:.4f} | top1 {val_top1:.2f} | top5 {val_top5:.2f}")

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'best_top1': best_top1,
                    'args': vars(args),
                },
                args.checkpoint
            )
            print(f"Saved best checkpoint to {args.checkpoint} (top1={best_top1:.2f})")

    print(f"Training complete. Best top1: {best_top1:.2f}")

if __name__ == "__main__":
    t_1=time.time()   
    main()
    t_2 = time.time()-t_1
    print(f"elapsed: {t_2} seconds.")