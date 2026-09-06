"""
RSR (SVD-based randomized reconstruction) defense for the CIFAR-10 VGG16 baseline.

For each input image, every channel is SVD-decomposed once (U, S, V^T). Then,
independently R times, a subset of singular-value indices is sampled without
replacement with probability proportional to the singular value's magnitude,
and the image is reconstructed using only that subset. Each of the R
reconstructions is classified, and the final prediction is the majority vote
across the R classifier outputs.

This script measures CLEAN accuracy only (no adversarial attack) on the full
CIFAR-10 test set, and compares it against the undefended VGG16 baseline
(models/vgg16_cifar10_baseline.py, 93.80% test accuracy after 100 epochs).
"""

import argparse
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from vgg16_cifar10_baseline import VGG16Cifar

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def get_raw_test_loader(data_root: str, batch_size: int, num_workers: int):
    """Test set with only ToTensor applied (pixel values in [0, 1]) - SVD
    reconstruction happens in raw pixel space, normalization is applied
    afterward, right before the images are fed to the model."""
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transforms.ToTensor()
    )
    return torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def svd_decompose(images: torch.Tensor):
    """Batched per-channel SVD. images: (N, C, H, W) in [0, 1].
    Returns U: (N, C, H, K), S: (N, C, K), Vh: (N, C, K, W), K = min(H, W)."""
    n, c, h, w = images.shape
    flat = images.reshape(n * c, h, w)
    U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
    k = S.shape[-1]
    return U.reshape(n, c, h, k), S.reshape(n, c, k), Vh.reshape(n, c, k, w)


def sample_reconstruction(U, S, Vh, keep_ratio: float, generator: torch.Generator):
    """One randomized reconstruction trial: sample a subset of singular-value
    indices per image/channel with probability proportional to the singular
    value, then rebuild the image using only those components."""
    n, c, h, k = U.shape
    w = Vh.shape[-1]
    device = U.device
    num_keep = max(1, round(k * keep_ratio))

    probs = S.reshape(n * c, k).clamp_min(1e-12).cpu()
    idx = torch.multinomial(probs, num_keep, replacement=False, generator=generator)

    mask = torch.zeros(n * c, k)
    mask.scatter_(1, idx, 1.0)
    mask = mask.reshape(n, c, k).to(device)

    S_masked = S * mask
    recon = torch.einsum("nchk,nck,nckw->nchw", U, S_masked, Vh)
    return recon.clamp(0.0, 1.0)


@torch.no_grad()
def evaluate_plain(model, loader, device):
    """Undefended forward pass, for an apples-to-apples reference number."""
    model.eval()
    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)
    correct, total = 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        normalized = (images - mean) / std
        preds = model(normalized).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_rsr(model, loader, device, R: int, keep_ratio: float, seed: int):
    """RSR defense: R independent SVD-sampled reconstructions per image,
    classified together, decided by majority vote."""
    model.eval()
    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)
    generator = torch.Generator().manual_seed(seed)

    correct, total = 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        n = images.size(0)

        U, S, Vh = svd_decompose(images)

        recons = torch.stack(
            [sample_reconstruction(U, S, Vh, keep_ratio, generator) for _ in range(R)],
            dim=0,
        )  # (R, N, C, H, W)

        batched = recons.reshape(R * n, *images.shape[1:])
        normalized = (batched - mean) / std
        logits = model(normalized)
        preds = logits.argmax(dim=1).reshape(R, n)  # (R, N)

        majority_vote = torch.mode(preds, dim=0).values  # (N,)
        correct += (majority_vote == targets).sum().item()
        total += n

    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the SVD-based RSR defense (clean accuracy, no attack)."
    )
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--baseline-acc", type=float, default=93.80,
                         help="Undefended baseline test accuracy to compare against.")
    parser.add_argument("--R", type=int, default=10, help="Number of majority-vote trials.")
    parser.add_argument("--keep-ratio", type=float, default=0.5,
                         help="Fraction of singular-value components sampled per trial.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VGG16Cifar(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    test_loader = get_raw_test_loader(args.data_root, args.batch_size, args.num_workers)

    start = time.time()
    plain_acc = evaluate_plain(model, test_loader, device)
    print(f"Plain (undefended) forward-pass accuracy on this run: {plain_acc:.2f}% "
          f"({time.time() - start:.1f}s)")

    start = time.time()
    rsr_acc = evaluate_rsr(model, test_loader, device, args.R, args.keep_ratio, args.seed)
    print(f"RSR defense (SVD sampling, R={args.R}, keep_ratio={args.keep_ratio}) "
          f"clean accuracy: {rsr_acc:.2f}% ({time.time() - start:.1f}s)")

    delta = rsr_acc - args.baseline_acc
    print(f"\nBaseline (undefended, from training run) test accuracy: {args.baseline_acc:.2f}%")
    print(f"RSR-defended clean test accuracy:                        {rsr_acc:.2f}%")
    print(f"Difference (RSR - baseline):                             {delta:+.2f}pp")


if __name__ == "__main__":
    main()
