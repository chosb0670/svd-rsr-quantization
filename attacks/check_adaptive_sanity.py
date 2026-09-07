"""
Sanity check for the adaptive-probability variant of the proposed defense
(models/proposed.py --adaptive-probs).

The original (fixed) proposed defense calibrates p_b' ONCE from 500 training
images and reuses that same vector for every test image - unlike original
RSR, whose sampling probabilities come fresh from each image's own singular
values. This script confirms the adaptive replacement actually restores that
property: it computes per-image p_b (before flooring) for a handful of test
images and checks that the vector is NOT identical across images (as it
trivially would be under the fixed/calibrated version).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from rsr import get_raw_test_loader, svd_decompose
from quantization import quantize_by_component_count
from proposed import calibrate_group_probabilities, compute_adaptive_group_probs


def main():
    parser = argparse.ArgumentParser(description="Check that adaptive p_b varies per image.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--floor", type=float, default=0.36)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = get_raw_test_loader(args.data_root, batch_size=args.num_samples, num_workers=0)
    images, labels = next(iter(loader))
    images = images[: args.num_samples].to(device)

    _, S, _ = svd_decompose(images)
    n, c, k = S.shape
    buckets = quantize_by_component_count(k, args.K).to(device)

    p_b, p_b_floored = compute_adaptive_group_probs(S, buckets, args.floor)
    # Channel-average for a compact per-image view.
    p_b_avg = p_b.mean(dim=1)  # (n, K)
    p_b_floored_avg = p_b_floored.mean(dim=1)

    print(f"\n=== Adaptive p_b (per image, channel-avg, K={args.K}) ===")
    header = "image   " + "".join(f"B{b:<9}" for b in range(args.K))
    print(header)
    for i in range(n):
        row = f"{i:<8}" + "".join(f"{v * 100:7.2f}%  " for v in p_b_avg[i].tolist())
        print(f"{row}  (label={labels[i].item()})")

    print(f"\n=== Adaptive p_b' (floored, floor={args.floor}) ===")
    print(header)
    for i in range(n):
        row = f"{i:<8}" + "".join(f"{v * 100:7.2f}%  " for v in p_b_floored_avg[i].tolist())
        print(row)

    all_same = torch.allclose(p_b_avg, p_b_avg[0:1].expand_as(p_b_avg), atol=1e-6)
    print(f"\nAll {n} images have IDENTICAL p_b vectors: {all_same} "
          f"(expected False - adaptive p_b should vary per image)")

    print(f"\n--- For reference: FIXED calibration (from {args.num_calib} training images) ---")
    _, fixed_p_b, fixed_p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, args.floor, args.num_calib, device, args.seed
    )
    print(f"  fixed p_b       = {[round(v, 4) for v in fixed_p_b.tolist()]}")
    print(f"  fixed p_b_prime = {[round(v, 4) for v in fixed_p_b_floored.tolist()]}")
    print("  (this single vector is what every test image used under the old, non-adaptive pipeline)")


if __name__ == "__main__":
    main()
