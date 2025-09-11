#!/usr/bin/env python3
"""
MNIST Ensemble of Three Simple CNNs (3x3 / 5x5 / 7x7) with Rotation/Translation Augmentation

- Trains three small CNNs that differ only by kernel size, then ensembles by averaging logits.
- Uses AMP (FP16/BF16) for speed; works on NVIDIA CUDA and AMD ROCm (via torch.cuda APIs).
- Strong baseline aiming toward 99.8–99.9% with enough training/augmentation.

Usage examples:
  pip install torch torchvision torchaudio
  # FP16 (recommended on most GPUs)
  python mnist_cnn_ensemble.py --epochs 25 --batch-size 512 --amp fp16
  # BF16 (great on RDNA3 / Ampere+)
  python mnist_cnn_ensemble.py --epochs 25 --batch-size 512 --amp bf16
  # Plain FP32
  python mnist_cnn_ensemble.py --amp none

Notes:
- You can raise --epochs (e.g., 50–100), tweak augmentation, or add EMA/MixUp to push accuracy.
- Script saves the best-accuracy checkpoint (all three models) to --save.
"""

import argparse
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST mean/std
MMEAN = (0.1307,)
MSTD  = (0.3081,)


def get_args():
    p = argparse.ArgumentParser(description="MNIST 3-CNN ensemble (3x3/5x5/7x7) with augmentation")
    p.add_argument('--data', type=str, default='./data', help='dataset root')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--num-workers', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--accum-steps', type=int, default=1, help='gradient accumulation steps')
    p.add_argument('--channels-last', action='store_true', help='use channels_last memory format')
    p.add_argument('--amp', type=str, choices=['fp16', 'bf16', 'none'], default='fp16', help='mixed precision mode')
    p.add_argument('--save', type=str, default='mnist_ensemble_best.pth')
    p.add_argument('--no-compile', action='store_true', help='disable torch.compile')
    p.add_argument('--aux-loss', action='store_true', help='add per-model auxiliary loss in addition to ensemble loss')
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    """Two Conv blocks + FC for MNIST (28x28).
    Each instance can choose a different kernel size (k in {3,5,7}).
    Structure per block: Conv(k,k, padding=k//2) -> BN -> ReLU -> MaxPool(2) -> Dropout2d(0.25)
    Then: Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 10)
    ~1.2M params across all three nets combined.
    """
    def __init__(self, k: int):
        super().__init__()
        pad = k // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, k, padding=pad, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, k, padding=pad, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28->14->7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@dataclass
class Ensemble:
    m3: SmallCNN
    m5: SmallCNN
    m7: SmallCNN

    def to(self, *args, **kwargs):
        self.m3.to(*args, **kwargs)
        self.m5.to(*args, **kwargs)
        self.m7.to(*args, **kwargs)
        return self

    def train(self):
        self.m3.train(); self.m5.train(); self.m7.train();
        return self

    def eval(self):
        self.m3.eval(); self.m5.eval(); self.m7.eval();
        return self

    def parameters(self):
        for p in self.m3.parameters():
            yield p
        for p in self.m5.parameters():
            yield p
        for p in self.m7.parameters():
            yield p

    def __call__(self, x):
        # NOTE: clone() to avoid "CUDAGraphs output overwritten" when using
        # torch.compile + cudagraphs. The compiled graphs reuse static output
        # buffers; cloning detaches each tensor to its own storage while
        # preserving gradients.
        o3 = self.m3(x).clone()
        o5 = self.m5(x).clone()
        o7 = self.m7(x).clone()
        return o3, o5, o7


# Data loaders with rotation/translation augmentation (no scaling keeps digits consistent)
# Feel free to tune degrees/translate slightly higher for more regularization

def make_loaders(root: str, batch_size: int, workers: int):
    train_tfms = transforms.Compose([
        transforms.RandomAffine(degrees=10.0, translate=(0.10, 0.10)),
        transforms.ToTensor(),
        transforms.Normalize(MMEAN, MSTD),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MMEAN, MSTD),
    ])

    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=train_tfms)
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=workers, persistent_workers=workers>0)
    test_loader  = DataLoader(test_ds, batch_size=max(512, batch_size), shuffle=False, pin_memory=True,
                              num_workers=workers, persistent_workers=workers>0)
    return train_loader, test_loader


def accuracy_from_logits(logits, targets):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return (pred == targets).float().mean().item()


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = make_loaders(args.data, args.batch_size, args.num_workers)

    ensemble = Ensemble(SmallCNN(3), SmallCNN(5), SmallCNN(7)).to(device)
    if args.channels_last and device.type == 'cuda':
        ensemble.m3.to(memory_format=torch.channels_last)
        ensemble.m5.to(memory_format=torch.channels_last)
        ensemble.m7.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"Total trainable params (3 nets): {total_params/1e6:.3f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(ensemble.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    # AMP setup
    amp_enabled = args.amp in {'fp16', 'bf16'} and device.type == 'cuda'
    if args.amp == 'fp16' and amp_enabled:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler(enabled=True, device='cuda')
    elif args.amp == 'bf16' and amp_enabled:
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler(enabled=False, device='cuda')
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler(enabled=False, device='cuda')
        amp_enabled = False

    # Optional compile (per-model to keep it simple)
    if not args.no_compile and device.type == 'cuda':
        try:
            ensemble.m3 = torch.compile(ensemble.m3, mode='max-autotune')
            ensemble.m5 = torch.compile(ensemble.m5, mode='max-autotune')
            ensemble.m7 = torch.compile(ensemble.m7, mode='max-autotune')
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")

    best_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        ensemble.train()
        t0 = time.time()
        run_loss, run_acc, seen = 0.0, 0.0, 0

        optimizer.zero_grad(set_to_none=True)

        for it, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if args.channels_last and device.type == 'cuda':
                images = images.to(memory_format=torch.channels_last)

            ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled) \
                  if amp_enabled else nullcontext()

            with ctx:
                o3, o5, o7 = ensemble(images)
                # Ensemble logits by arithmetic mean
                logits_ens = (o3 + o5 + o7) / 3.0

                # Main loss: average of per-model CE to ensure each branch learns
                losses = [criterion(o3, targets), criterion(o5, targets), criterion(o7, targets)]
                loss = sum(losses) / 3.0

                # Optional extra: also supervise the ensemble output
                if args.aux_loss:
                    loss = 0.5 * loss + 0.5 * criterion(logits_ens, targets)

                acc = accuracy_from_logits(logits_ens, targets)

            if amp_enabled and args.amp == 'fp16':
                scaler.scale(loss).backward()
                if it % args.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(ensemble.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            else:
                loss.backward()
                if it % args.accum_steps == 0:
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(ensemble.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            bs = images.size(0)
            run_loss += loss.item() * bs
            run_acc  += acc * bs
            seen     += bs
            global_step += 1

            if it % 100 == 0:
                print(f"Epoch {epoch} [{it}/{steps_per_epoch}] step={global_step} "
                      f"loss={run_loss/seen:.4f} acc={100*run_acc/seen:.2f}%")

        dt = time.time() - t0
        print(f"Epoch {epoch} done in {dt:.1f}s | train loss={run_loss/seen:.4f} | acc={100*run_acc/seen:.2f}%")

        # Evaluation
        ensemble.eval()
        test_loss_sum, test_correct, test_seen = 0.0, 0, 0
        with torch.no_grad():
            ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled)
            with ctx:
                for images, targets in test_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    if args.channels_last and device.type == 'cuda':
                        images = images.to(memory_format=torch.channels_last)
                    o3, o5, o7 = ensemble(images)
                    logits_ens = (o3 + o5 + o7) / 3.0
                    loss = criterion(logits_ens, targets)
                    test_loss_sum += loss.item() * images.size(0)
                    test_correct  += (logits_ens.argmax(1) == targets).sum().item()
                    test_seen     += images.size(0)

        val_loss = test_loss_sum / max(test_seen, 1)
        val_acc  = 100.0 * test_correct / max(test_seen, 1)
        print(f"[val] epoch {epoch} | loss {val_loss:.4f} | acc {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'm3': ensemble.m3.state_dict(),
                'm5': ensemble.m5.state_dict(),
                'm7': ensemble.m7.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc,
                'args': vars(args),
            }, args.save)
            print(f"Saved best checkpoint to {args.save} (acc={best_acc:.2f}%)")

    print(f"Training complete. Best val acc: {best_acc:.2f}%")


if __name__ == '__main__':
    t_1=time.time()    
    main()
    t_2 = time.time()-t_1
    print(f"elapsed: {t_2} seconds.")