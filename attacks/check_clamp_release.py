"""
Cheap check (no model inference, no attacks): for a set of candidate floors
well below B1's observed max (3.29% at floor-independent raw p_b, from the
0.05-0.36 sweep in PROGRESS.md), what fraction of the same 1000 test images
(seed=42) would actually have at least one of B1..B(K-1) NOT clamped (i.e.
raw energy share > floor) under that floor?

This recomputes each image's raw (unfloored) per-group energy share
(compute_adaptive_group_probs with floor=0.0 - the "floor" argument doesn't
affect the returned raw p_b, only the floored p_b_floored, which we ignore
here) - the same underlying quantity attacks/evaluate_adaptive_sweep.py's
sanity check already showed aggregate min/max/mean for, but this script
returns the full per-image array so multiple candidate floors can be swept
against it in one pass without recomputing SVDs per candidate.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from rsr import svd_decompose
from quantization import quantize_by_component_count
from proposed import compute_adaptive_group_probs

from evaluate import get_random_test_subset, make_loader


@torch.no_grad()
def compute_raw_p_b(loader, buckets, K, device):
    all_p_b = []
    for images, _ in loader:
        images = images.to(device)
        U, S, Vh = svd_decompose(images)
        p_b, _ = compute_adaptive_group_probs(S, buckets, 0.0)
        all_p_b.append(p_b.mean(dim=1).cpu())  # (n, K) channel-avg
        del images, U, S, Vh, p_b
    return torch.cat(all_p_b, dim=0)  # (N, K)


def main():
    parser = argparse.ArgumentParser(
        description="Check what fraction of images have B1..B(K-1) unclamped at candidate low floors."
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--floors", type=str, default="0.005,0.01,0.02,0.03")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    floors = [float(f) for f in args.floors.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print(f"Sampling {args.num_samples} random test images (seed={args.seed})...", flush=True)
    images, labels = get_random_test_subset(args.data_root, args.num_samples, args.seed)
    loader = make_loader(images, labels, args.batch_size)

    k = images.shape[-1]  # unused placeholder; buckets built from actual SVD k below
    # Determine k from one batch's SVD output.
    sample_images = images[:2].to(device)
    _, S0, _ = svd_decompose(sample_images)
    k_svd = S0.shape[-1]
    buckets = quantize_by_component_count(k_svd, args.K).to(device)
    del sample_images, S0

    p_b = compute_raw_p_b(loader, buckets, args.K, device)  # (N, K)

    print(f"\n=== Per-group raw p_b stats ({args.num_samples} images, channel-avg) ===")
    for b in range(args.K):
        vals = p_b[:, b]
        print(f"  B{b}: min={vals.min().item() * 100:8.4f}%  max={vals.max().item() * 100:8.4f}%  "
              f"mean={vals.mean().item() * 100:8.4f}%")

    print(f"\n=== Fraction of images with >=1 of B1..B{args.K - 1} UNCLAMPED (raw > floor) ===")
    print(f"{'floor':<10}{'any-unclamped %':<18}" +
          "".join(f"B{b}-unclamped %" for b in range(1, args.K)))
    results = {}
    for floor in floors:
        per_group_unclamped = []
        any_unclamped = torch.zeros(p_b.size(0), dtype=torch.bool)
        for b in range(1, args.K):
            unclamped_b = p_b[:, b] > floor
            per_group_unclamped.append(float(unclamped_b.float().mean().item()))
            any_unclamped |= unclamped_b
        any_frac = float(any_unclamped.float().mean().item())
        results[floor] = {"any_unclamped_fraction": any_frac,
                           "per_group_unclamped_fraction": per_group_unclamped}
        row = f"{floor:<10}{any_frac * 100:<18.2f}" + \
              "".join(f"{v * 100:14.2f}" for v in per_group_unclamped)
        print(row)

    print("\n(any-unclamped % = share of the 1000 images where at least one of B1..B{K-1}'s "
          "raw energy share exceeds this floor, i.e. would NOT be clamped up to the floor)")

    import json
    out_path = "./results/clamp_release_check.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
