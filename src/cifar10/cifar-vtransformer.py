import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import timm
except ImportError as e:
    raise SystemExit("timm is required. Install with: pip install timm")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_args():
    p = argparse.ArgumentParser(description="CIFAR-10 ViT fine-tuning with AMP")
    p.add_argument('--data', type=str, default='./data', help='dataset root')
    p.add_argument('--model', type=str, default='vit_tiny_patch16_224',
                   help='timm model name, e.g., vit_tiny_patch16_224, vit_base_patch16_224')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--num-workers', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--compile', dest='compile', action='store_true', default=True,
                   help='use torch.compile (default on)')
    p.add_argument('--no-compile', dest='compile', action='store_false')
    p.add_argument('--channels-last', action='store_true', help='use channels_last memory format')
    p.add_argument('--amp-dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'none'],
                   help='mixed precision dtype (fp16, bf16) or none for FP32')
    p.add_argument('--accum-steps', type=int, default=1, help='gradient accumulation steps')
    p.add_argument('--save', type=str, default='best_vit_cifar10.pth')
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(data_root: str, batch_size: int, num_workers: int):
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tfms)
    test_ds  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=num_workers, persistent_workers=num_workers>0)
    test_loader = DataLoader(
        test_ds, batch_size=max(256, batch_size), shuffle=False, pin_memory=True,
        num_workers=num_workers, persistent_workers=num_workers>0)
    return train_loader, test_loader


def build_model(name: str, device: torch.device, channels_last: bool):
    # Load ImageNet-pretrained ViT and adapt to CIFAR-10
    model = timm.create_model(name, pretrained=True, num_classes=10)
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return model


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)


def main():
    args = get_args()
    set_seed(args.seed)

    assert torch.cuda.is_available(), "A CUDA/ROCm-capable GPU is required for this script."
    device = torch.device('cuda')  # ROCm build uses CUDA APIs under the hood

    train_loader, test_loader = create_dataloaders(args.data, args.batch_size, args.num_workers)

    model = build_model(args.model, device, args.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)

    # AMP setup
    if args.amp_dtype == 'fp16':
        amp_dtype = torch.float16
        amp_enabled = True
        scaler = torch.amp.GradScaler(enabled=True, device='cuda')
    elif args.amp_dtype == 'bf16':
        amp_dtype = torch.bfloat16
        amp_enabled = True
        scaler = torch.amp.GradScaler(enabled=False, device='cuda')  # not used in bf16
    else:
        amp_dtype = torch.float32
        amp_enabled = False
        scaler = torch.amp.GradScaler(enabled=False, device='cuda')

    # Optional compile
    if args.compile:
        try:
            model = torch.compile(model, mode='max-autotune')
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")

    best_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for it, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)

            ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled) \
                if amp_enabled else nullcontext()

            with ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)
                acc = accuracy(outputs, targets)

            if amp_enabled and args.amp_dtype == 'fp16':
                scaler.scale(loss).backward()
                # gradient accumulation
                if it % args.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            else:
                # BF16 or FP32 path
                loss.backward()
                if it % args.accum_steps == 0:
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_acc += acc * bs
            seen += bs
            global_step += 1

            if it % 50 == 0:
                print(f"Epoch {epoch} [{it}/{len(train_loader)}] step={global_step} "
                      f"loss={running_loss/seen:.4f} acc={100*running_acc/seen:.2f}%")

        dt = time.time() - t0
        train_loss = running_loss / max(seen, 1)
        train_acc = 100 * running_acc / max(seen, 1)
        print(f"Epoch {epoch} finished in {dt:.1f}s | train loss {train_loss:.4f} | train acc {train_acc:.2f}%")

        # Eval
        model.eval()
        val_loss_sum, val_correct, val_seen = 0.0, 0, 0
        with torch.no_grad():
            ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled)
            with ctx:
                for images, targets in test_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    if args.channels_last:
                        images = images.to(memory_format=torch.channels_last)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss_sum += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(1) == targets).sum().item()
                    val_seen += images.size(0)
        val_loss = val_loss_sum / max(val_seen, 1)
        val_acc = 100.0 * val_correct / max(val_seen, 1)
        print(f"[val] epoch {epoch} | loss {val_loss:.4f} | acc {val_acc:.2f}%")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc,
                'args': vars(args),
            }, args.save)
            print(f"Saved checkpoint to {args.save} (acc={best_acc:.2f}%)")

    print(f"Training complete. Best val acc: {best_acc:.2f}%")


if __name__ == '__main__':
    t_1 = time.time()
    main()
    t_2 = time.time()-t_1
    print(f"elapsed: {t_2} seconds.")