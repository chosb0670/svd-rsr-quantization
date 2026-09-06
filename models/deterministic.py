"""
Deterministic control for the proposed defense (models/proposed.py).

The proposed defense approximates E_{group mask ~ Bernoulli(p_b')}[softmax(f(recon(x, mask)))]
with R=10 Monte Carlo samples. For K=5 groups there are only 2^5=32 possible
inclusion/exclusion patterns, so that expectation can be computed EXACTLY:
enumerate all 32 patterns, weight each by its probability under the same
per-group Bernoulli(p_b') model, reconstruct + classify every pattern, and
take the probability-weighted average softmax as the final (still
argmax-decided) prediction.

Because every pattern's mask is a FIXED 0/1 tensor (not a random draw), the
whole pipeline - SVD reconstruction, masking, classification, weighted
average - is composed entirely of differentiable operations. Unlike the
random-sampling proposed/RSR defenses, no BPDA identity-approximation is
needed to attack this: an adversary can backpropagate through the exact
expectation directly. This script (and attacks/evaluate_deterministic.py)
exist to check whether removing the defense's randomness raises clean
accuracy but lowers adversarial robustness, since exact gradients let PGD
find adversarial examples more effectively than against a noisy, randomly
resampled defense.
"""

import argparse
import itertools

import torch
import torch.nn.functional as F

from vgg16_cifar10_baseline import VGG16Cifar
from rsr import CIFAR_MEAN, CIFAR_STD, get_raw_test_loader, svd_decompose
from proposed import calibrate_group_probabilities, reconstruct_with_group_mask


def enumerate_combinations(K: int) -> torch.Tensor:
    """All 2^K inclusion/exclusion patterns, as a (2^K, K) bool tensor."""
    patterns = list(itertools.product([0, 1], repeat=K))
    return torch.tensor(patterns, dtype=torch.bool)


def combination_probabilities(patterns: torch.Tensor, p_b_floored: torch.Tensor) -> torch.Tensor:
    """pi(pattern) = prod_b [p_b' if included else (1 - p_b')].
    patterns: (M, K) bool. Returns (M,) probabilities that sum to 1."""
    p = p_b_floored.unsqueeze(0)  # (1, K)
    term = torch.where(patterns, p, 1 - p)  # (M, K)
    return term.prod(dim=-1)


def deterministic_forward(model, images: torch.Tensor, buckets: torch.Tensor,
                           patterns: torch.Tensor, pi: torch.Tensor,
                           mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """images: (N, C, H, W) raw pixel [0, 1] - fully differentiable, no BPDA
    needed. Returns the pi-weighted average softmax, shape (N, num_classes).
    All M=2^K patterns are batched into a single (M*N) forward pass."""
    device = images.device
    n, c, h, w = images.shape
    M, K = patterns.shape

    U, S, Vh = svd_decompose(images)  # (n,c,h,k), (n,c,k), (n,c,k,w)

    group_mask_all = patterns.to(device).view(M, 1, 1, K).expand(M, n, c, K)
    component_mask_all = group_mask_all[..., buckets.to(device)].to(S.dtype)  # (M, n, c, k)

    S_masked_all = S.unsqueeze(0) * component_mask_all  # (M, n, c, k)
    U_exp = U.unsqueeze(0).expand(M, n, c, h, U.shape[-1])
    Vh_exp = Vh.unsqueeze(0).expand(M, n, c, Vh.shape[-2], w)
    recon = torch.einsum("mnchk,mnck,mnckw->mnchw", U_exp, S_masked_all, Vh_exp)
    recon = recon.clamp(0.0, 1.0)

    batched = recon.reshape(M * n, c, h, w)
    normalized = (batched - mean) / std
    logits = model(normalized)
    probs = F.softmax(logits, dim=-1).reshape(M, n, -1)

    weighted = (pi.to(device).view(M, 1, 1) * probs).sum(dim=0)  # (n, num_classes)
    return weighted


@torch.no_grad()
def evaluate_deterministic(model, loader, device, buckets, patterns, pi):
    model.eval()
    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)
    correct, total = 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        probs = deterministic_forward(model, images, buckets, patterns, pi, mean, std)
        preds = probs.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Clean-accuracy check for the deterministic (exact expectation) control."
    )
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--floor", type=float, default=0.36)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32,
                         help="Kept small since each pattern batches all 2^K reconstructions "
                              "together internally (effective batch = batch_size * 2^K).")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VGG16Cifar(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    buckets, p_b, p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, args.floor, args.num_calib, device, args.seed
    )
    patterns = enumerate_combinations(args.K)
    pi = combination_probabilities(patterns, p_b_floored)
    print(f"K={args.K} -> {patterns.size(0)} patterns, p_b_prime={[round(v, 4) for v in p_b_floored.tolist()]}")
    print(f"pi sums to {pi.sum().item():.6f} (should be 1.0)")

    test_loader = get_raw_test_loader(args.data_root, args.batch_size, args.num_workers)
    acc = evaluate_deterministic(model, test_loader, device, buckets, patterns, pi)
    print(f"\nDeterministic (exact expectation, K={args.K}, floor={args.floor}) clean accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
