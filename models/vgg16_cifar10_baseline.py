"""
CIFAR-10 + VGG16 baseline training (no adversarial defense).

Defines a VGG16 variant adapted to CIFAR-10's 32x32 input resolution and
trains it with plain SGD (no adversarial training, no input preprocessing
defense, no gradient masking). Serves as the undefended baseline that
defended variants (SVD / RSR / quantization, etc.) will be compared against.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Standard VGG16 conv stack: 5 stages of conv+BN+ReLU followed by 2x2 max-pool.
# 32x32 input -> 5 pools -> 1x1 feature map with 512 channels.
VGG16_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


class VGG16Cifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = self._make_layers(VGG16_CFG)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_dataloaders(data_root: str, batch_size: int, num_workers: int):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    # persistent_workers keeps worker processes alive across epochs instead of
    # respawning them every epoch, which is what was slowly leaking memory on
    # Windows over a long run.
    persistent_workers = num_workers > 0
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += inputs.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += inputs.size(0)

    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train an undefended VGG16 baseline on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                         help="Save a periodic checkpoint every N epochs.")
    parser.add_argument("--checkpoint-dir", type=str, default="./results/checkpoints")
    parser.add_argument("--best-model-path", type=str, default="./results/best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint to resume from (accepts either a raw "
                              "model state_dict, e.g. an old checkpoint_epochN.pth, or the "
                              "richer {epoch, model_state_dict, optimizer_state_dict, "
                              "scheduler_state_dict, best_acc} dict this script now saves).")
    parser.add_argument("--start-epoch", type=int, default=0,
                         help="Epoch already completed by --resume's checkpoint. Training "
                              "continues from start-epoch+1 through --epochs. Ignored if the "
                              "checkpoint is the richer dict format, which carries its own epoch.")
    parser.add_argument("--initial-best-acc", type=float, default=0.0,
                         help="Seed value for best-so-far test accuracy when resuming, so a "
                              "worse epoch doesn't overwrite an already-better best_model.pth.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.data_root, args.batch_size, args.num_workers)

    model = VGG16Cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )

    best_acc = args.initial_best_acc
    start_epoch = args.start_epoch

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # Richer checkpoint format (this script's own periodic saves).
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if args.start_epoch == 0 and "epoch" in ckpt:
                start_epoch = ckpt["epoch"]
            if args.initial_best_acc == 0.0 and "best_acc" in ckpt:
                best_acc = ckpt["best_acc"]
        else:
            # Old format: a raw model state_dict (e.g. an earlier checkpoint_epochN.pth).
            # Optimizer/scheduler state isn't recoverable from this format, so
            # --start-epoch and --initial-best-acc must be passed explicitly.
            model.load_state_dict(ckpt)
        print(f"Resumed model weights. Continuing from epoch {start_epoch + 1}, "
              f"best_acc so far = {best_acc:.2f}%")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.resume and isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    elif start_epoch > 0:
        # No saved scheduler state (old checkpoint format): CosineAnnealingLR's
        # get_lr() is defined recursively off the previous step's LR, so it must
        # be replayed step-by-step to reach the correct value - jumping straight
        # to last_epoch=start_epoch at construction gives the wrong LR.
        for _ in range(start_epoch):
            scheduler.step()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.best_model_path) or ".", exist_ok=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% | "
            f"lr={scheduler.get_last_lr()[0]:.5f} | {elapsed:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.best_model_path)
            print(f"  -> New best test accuracy ({best_acc:.2f}%), saved to {args.best_model_path}")

        if epoch % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
            }, ckpt_path)
            print(f"  -> Saved checkpoint to {ckpt_path}")

    print(f"\nFinal test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Best test accuracy during training: {best_acc:.2f}%")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved final model weights to {args.save_path}")


if __name__ == "__main__":
    main()
