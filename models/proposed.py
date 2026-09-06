"""
Proposed defense: SVD component-count quantization (K groups) + per-group
Bernoulli inclusion probability, combined into an R=10 majority-vote
reconstruction pipeline.

Builds on models/rsr.py (SVD decomposition, majority-vote evaluation
scaffolding) and models/quantization.py (fixed-size component-count
grouping, per-group inclusion probability design):

1. Each channel is SVD-decomposed once (U, S, V^T), as in RSR.
2. The K=5 groups are fixed-size buckets over singular-value rank (from
   quantize_by_component_count), calibrated once on a sample of TRAINING
   images to get each group's average energy share, then floored at
   `--floor` to get the per-group inclusion probability p_b'.
3. On every trial (R=10 total), each of the K groups is independently
   included via a Bernoulli(p_b') draw (per image, per channel). If a draw
   excludes every group, the highest-probability group is force-included so
   the reconstruction is never entirely zeroed out.
4. The image is reconstructed from only the included groups' singular
   components, classified by VGG16, and the R predictions are combined by
   majority vote into the final label.

This script measures CLEAN accuracy only (no adversarial attack) on the full
CIFAR-10 test set, and compares it against the undefended baseline and the
original (magnitude-proportional sampling) RSR defense.
"""

import argparse
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from vgg16_cifar10_baseline import VGG16Cifar
from rsr import CIFAR_MEAN, CIFAR_STD, get_raw_test_loader, svd_decompose, evaluate_plain, evaluate_rsr
from quantization import quantize_by_component_count, design_group_probabilities


def calibrate_group_probabilities(data_root: str, K: int, floor: float,
                                   num_calib: int, device: torch.device, seed: int = 42):
    """Estimates each group's average energy share on a sample of TRAINING
    images (kept separate from the test set used for the final accuracy
    comparison), then floors it into an inclusion probability p_b'."""
    calib_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transforms.ToTensor()
    )
    # Seeded so the same training images are sampled across a --floor sweep,
    # isolating the effect of `floor` from calibration-sample variance.
    calib_generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        calib_set, batch_size=num_calib, shuffle=True, generator=calib_generator
    )
    images, _ = next(iter(loader))
    images = images.to(device)

    _, S, _ = svd_decompose(images)
    n, c, k = S.shape
    buckets = quantize_by_component_count(k, K).to(device)

    flat_energy = (S ** 2).reshape(n * c, k)
    total = flat_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    ratios = flat_energy / total  # (n*c, k)

    group_ratio = torch.zeros(K, device=device)
    for b in range(K):
        group_ratio[b] = ratios[:, buckets == b].sum(dim=-1).mean()

    p_b, p_b_floored = design_group_probabilities(group_ratio.cpu(), floor)
    return buckets, p_b, p_b_floored


def sample_group_mask(p_b_floored: torch.Tensor, n: int, c: int, generator: torch.Generator):
    """Independent Bernoulli(p_b') draw per image/channel/group for one
    trial. Falls back to force-including the highest-probability group if a
    draw excludes every group (avoids an entirely zeroed reconstruction)."""
    K = p_b_floored.numel()
    probs = p_b_floored.view(1, 1, K).expand(n, c, K)
    mask = torch.bernoulli(probs, generator=generator).bool()
    empty = ~mask.any(dim=-1)  # (n, c)
    if empty.any():
        fallback_group = torch.argmax(p_b_floored).item()
        mask[..., fallback_group] |= empty
    return mask


def reconstruct_with_group_mask(U, S, Vh, buckets, group_mask):
    """buckets: (k,) group id per singular-value index. group_mask: (n, c, K)
    bool. Reconstructs using only the singular components whose group was
    included for that image/channel."""
    component_mask = group_mask[..., buckets].to(S.dtype)  # (n, c, k)
    S_masked = S * component_mask
    recon = torch.einsum("nchk,nck,nckw->nchw", U, S_masked, Vh)
    return recon.clamp(0.0, 1.0)


@torch.no_grad()
def evaluate_proposed(model, loader, device, R: int, buckets: torch.Tensor,
                       p_b_floored: torch.Tensor, seed: int):
    """Returns (accuracy, avg_keep_ratio) - avg_keep_ratio is the mean
    fraction of singular-value components actually used across every
    image/channel/trial's reconstruction (comparable to RSR's keep_ratio)."""
    model.eval()
    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)
    generator = torch.Generator().manual_seed(seed)
    buckets = buckets.to(device)
    p_b_floored_cpu = p_b_floored.cpu()
    k = buckets.numel()

    correct, total = 0, 0
    keep_ratio_sum, keep_ratio_count = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        n, c = images.size(0), images.size(1)

        U, S, Vh = svd_decompose(images)

        recons = []
        for _ in range(R):
            mask = sample_group_mask(p_b_floored_cpu, n, c, generator).to(device)
            component_mask = mask[..., buckets]  # (n, c, k)
            keep_ratio_sum += component_mask.float().sum().item()
            keep_ratio_count += component_mask.numel()
            recons.append(reconstruct_with_group_mask(U, S, Vh, buckets, mask))
        recons = torch.stack(recons, dim=0)  # (R, N, C, H, W)

        batched = recons.reshape(R * n, *images.shape[1:])
        normalized = (batched - mean) / std
        logits = model(normalized)
        preds = logits.argmax(dim=1).reshape(R, n)

        majority_vote = torch.mode(preds, dim=0).values
        correct += (majority_vote == targets).sum().item()
        total += n

    accuracy = 100.0 * correct / total
    avg_keep_ratio = keep_ratio_sum / keep_ratio_count
    return accuracy, avg_keep_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the proposed quantization+group-probability defense (clean accuracy)."
    )
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--R", type=int, default=10, help="Number of majority-vote trials.")
    parser.add_argument("--K", type=int, default=5, help="Number of component-count groups.")
    parser.add_argument("--floor", type=float, default=0.05, help="Minimum group inclusion probability.")
    parser.add_argument("--num-calib", type=int, default=500,
                         help="Number of training images used to estimate group energy shares.")
    parser.add_argument("--rsr-keep-ratio", type=float, default=0.5,
                         help="keep_ratio used for the original RSR comparison run.")
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

    buckets, p_b, p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, args.floor, args.num_calib, device, args.seed
    )
    print(f"\nCalibrated on {args.num_calib} training images "
          f"(group sizes={[int((buckets == b).sum()) for b in range(args.K)]}):")
    print(f"  p_b       = {[round(v, 4) for v in p_b.tolist()]}")
    print(f"  p_b_prime = {[round(v, 4) for v in p_b_floored.tolist()]} (floor={args.floor})")

    test_loader = get_raw_test_loader(args.data_root, args.batch_size, args.num_workers)

    start = time.time()
    plain_acc = evaluate_plain(model, test_loader, device)
    print(f"\nPlain (undefended) accuracy: {plain_acc:.2f}% ({time.time() - start:.1f}s)")

    start = time.time()
    rsr_acc = evaluate_rsr(model, test_loader, device, args.R, args.rsr_keep_ratio, args.seed)
    print(f"Original RSR (magnitude-proportional sampling, R={args.R}, "
          f"keep_ratio={args.rsr_keep_ratio}) accuracy: {rsr_acc:.2f}% ({time.time() - start:.1f}s)")

    start = time.time()
    proposed_acc, avg_keep_ratio = evaluate_proposed(
        model, test_loader, device, args.R, buckets, p_b_floored, args.seed
    )
    print(f"Proposed (quantized groups + Bernoulli(p_b'), R={args.R}, K={args.K}) "
          f"accuracy: {proposed_acc:.2f}%, avg keep_ratio={avg_keep_ratio:.4f} "
          f"({time.time() - start:.1f}s)")

    print("\n=== Clean accuracy comparison ===")
    print(f"{'Method':<45}{'Accuracy':>10}")
    print(f"{'Baseline (undefended)':<45}{plain_acc:>9.2f}%")
    print(f"{'Original RSR (keep_ratio={:.2f})'.format(args.rsr_keep_ratio):<45}{rsr_acc:>9.2f}%")
    print(f"{'Proposed (avg keep_ratio={:.4f})'.format(avg_keep_ratio):<45}{proposed_acc:>9.2f}%")
    print(f"\nProposed vs baseline: {proposed_acc - plain_acc:+.2f}pp")
    print(f"Proposed vs original RSR: {proposed_acc - rsr_acc:+.2f}pp")


if __name__ == "__main__":
    main()
