#!/usr/bin/env python3
"""
Fashion-MNIST: small CNN + data augmentation (PyTorch)

- Architecture inspired by "small CNN" recipes that top Fashion-MNIST (≈3 conv blocks + 2 FC + dropout).
- Augmentations approximate ImageDataGenerator settings often reported in top results:
    rotation ≈ 7.5°, width/height shift ≈ 0.08, zoom/scale ≈ ±8.5%.
- Mixed precision supported (FP16/BF16) and works on NVIDIA CUDA and AMD ROCm.

Usage:
  # Install
  #   pip install torch torchvision torchaudio  # pick a wheel for your platform (CUDA/ROCm/CPU)
  #   
  # Train (FP16 AMP)
  #   python fashion_mnist_cnn_aug.py --epochs 50 --batch-size 256 --amp fp16
  # Train (BF16 AMP)
  #   python fashion_mnist_cnn_aug.py --epochs 50 --batch-size 256 --amp bf16
  # Train (FP32)
  #   python fashion_mnist_cnn_aug.py --amp none

Notes:
- This script is meant as a strong baseline; exact "SOTA" numbers depend on training details
  (optimizer schedule, longer training, ensembling, augmentation tuning, etc.).
"""

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

# Fashion-MNIST stats (grayscale)
FMEAN = (0.2860405969887955,)  # ~0.2860
FSTD  = (0.35302424451492237,) # ~0.3530


def parse_args():
    p = argparse.ArgumentParser(description="Fashion-MNIST small CNN with augmentation")
    p.add_argument('--data', type=str, default='./data', help='dataset root')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--num-workers', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--channels-last', action='store_true', help='use channels_last memory format')
    p.add_argument('--amp', type=str, choices=['fp16', 'bf16', 'none'], default='fp16',
                   help='mixed precision dtype (fp16/bf16) or none')
    p.add_argument('--save', type=str, default='best_fmnist_cnn.pth')
    p.add_argument('--no-compile', action='store_true', help='disable torch.compile')
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    """3 conv blocks (32,64,128) + 2 FC (~688k params with fc=512).
    Conv: 3x3 (pad=1) -> BN -> ReLU -> MaxPool(2) -> Dropout2d(0.25).
    FC:  (128*3*3)->512 -> Dropout(0.5) -> 10.
    """
    def __init__(self, fc_dim: int = 512, p_drop_conv: float = 0.25, p_drop_fc: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p_drop_conv),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p_drop_conv),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28->14->7->3
            nn.Dropout2d(p_drop_conv),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop_fc),
            nn.Linear(fc_dim, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_loaders(root: str, batch_size: int, workers: int):
    # Data augmentation approximating: rot=7.5°, translate≈0.08, scale 0.915..1.085
    train_tfms = transforms.Compose([
        transforms.RandomAffine(degrees=7.5, translate=(0.08, 0.08), scale=(0.915, 1.085)),
        transforms.ToTensor(),
        transforms.Normalize(FMEAN, FSTD),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FMEAN, FSTD),
    ])

    train_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=train_tfms)
    test_ds  = datasets.FashionMNIST(root=root, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=workers, persistent_workers=workers>0)
    test_loader  = DataLoader(test_ds, batch_size=max(512, batch_size), shuffle=False, pin_memory=True,
                              num_workers=workers, persistent_workers=workers>0)
    return train_loader, test_loader


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item()


def main():
    args = parse_args()
    set_seed(args.seed)

    assert torch.cuda.is_available(), "A CUDA/ROCm-capable GPU is required."
    device = torch.device('cuda')

    train_loader, test_loader = make_loaders(args.data, args.batch_size, args.num_workers)

    model = SmallCNN(fc_dim=512).to(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.3f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    # AMP setup
    amp_enabled = args.amp in {'fp16', 'bf16'}
    if args.amp == 'fp16':
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler(enabled=True, device='cuda')
    elif args.amp == 'bf16':
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler(enabled=False, device='cuda')
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler(enabled=False, device='cuda')

    # Optional compile
    if not args.no_compile:
        try:
            model = torch.compile(model, mode='max-autotune')
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")

    best_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, run_acc, seen = 0.0, 0.0, 0

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

            if args.amp == 'fp16':
                scaler.scale(loss).backward()
                if it % args.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            else:
                loss.backward()
                if it % args.accum_steps == 0:
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
        print(f"Epoch {epoch} done in {dt:.1f}s | train loss={run_loss/seen:.4f} acc={100*run_acc/seen:.2f}%")

        # Eval
        model.eval()
        test_loss_sum, test_correct, test_seen = 0.0, 0, 0
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
                    test_loss_sum += loss.item() * images.size(0)
                    test_correct  += (outputs.argmax(1) == targets).sum().item()
                    test_seen     += images.size(0)
        val_loss = test_loss_sum / max(test_seen, 1)
        val_acc  = 100.0 * test_correct / max(test_seen, 1)
        print(f"[val] epoch {epoch} | loss {val_loss:.4f} | acc {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': model.state_dict(),
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
