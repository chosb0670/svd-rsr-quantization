"""
Component-count bucketing ("quantization") of SVD singular-value components.

Builds on the SVD decomposition used by the RSR defense (models/rsr.py):
singular values are sorted descending (as torch.linalg.svd returns them) and
split into K groups of as-equal-as-possible *component count* (e.g. 32
components into K=5 groups of 7,7,6,6,6). Each group's energy share (sum of
sigma_i^2 in that group, over the total) is then reported separately.

An earlier version grouped by cumulative-energy quantiles instead (bucket 0 =
first 1/K of energy, etc.), but on CIFAR-10 that collapsed: raw pixel SVD
spectra are dominated by a near-constant/DC-like leading singular value that
alone captures ~90%+ of the total sigma^2 energy, so its cumulative ratio
already exceeds every earlier threshold and every single component - not
just the first - lands in the last (80-100%) bucket, leaving buckets 0-3
empty on every test image. Fixed-size, count-based grouping avoids this: it
always produces non-empty, evenly-sized groups, while the (still very
skewed) energy distribution across those groups remains visible in the
reported per-bucket energy shares.
"""

import argparse

import torch

from rsr import get_raw_test_loader, svd_decompose


def quantize_by_cumulative_energy(S: torch.Tensor, K: int = 5) -> torch.Tensor:
    """Superseded by quantize_by_component_count - see module docstring for
    why cumulative-energy quantiles collapse on this data. Kept for reference.

    S: (..., k) singular values sorted descending. Returns an integer bucket
    index in [0, K-1] per singular value, assigned by which 1/K slice of
    cumulative energy it falls into."""
    energy = S ** 2
    total = energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    cum_ratio = torch.cumsum(energy, dim=-1) / total
    bucket = (cum_ratio * K).ceil().clamp(min=1, max=K) - 1
    return bucket.long()


def quantize_by_component_count(k: int, K: int = 5) -> torch.Tensor:
    """Splits the k singular-value indices (sorted descending, so index 0 is
    the largest) into K groups of as-equal-as-possible size: the first
    (k % K) groups get one extra component. E.g. k=32, K=5 -> sizes
    [7, 7, 6, 6, 6]. Returns a (k,) tensor of bucket indices in [0, K-1]."""
    base, remainder = divmod(k, K)
    sizes = [base + 1 if b < remainder else base for b in range(K)]
    bucket = torch.empty(k, dtype=torch.long)
    start = 0
    for b, size in enumerate(sizes):
        bucket[start:start + size] = b
        start += size
    return bucket


def design_group_probabilities(avg_energy_ratios: torch.Tensor, floor: float = 0.05):
    """Turns each group's average energy share into an inclusion probability.

    p_b := avg_energy_ratios[b] (the group's average energy share across the
    sampled images). p_b' := max(p_b, floor) - a floor so that even a
    near-zero-energy group still keeps a minimum chance of being included
    (not renormalized: these are independent per-group inclusion
    probabilities, not a categorical distribution that must sum to 1)."""
    p_b = avg_energy_ratios.clone()
    p_b_floored = torch.clamp(p_b, min=floor)
    return p_b, p_b_floored


def bucket_stats(S: torch.Tensor, buckets: torch.Tensor, K: int):
    """S, buckets: (k,) for a single channel. Returns (counts, energy_ratios),
    each of length K: how many components landed in each bucket, and what
    fraction of the total energy each bucket accounts for."""
    energy = S ** 2
    total = energy.sum().clamp_min(1e-12)
    counts = torch.zeros(K, dtype=torch.long)
    energy_ratios = torch.zeros(K)
    for b in range(K):
        mask = buckets == b
        counts[b] = mask.sum()
        energy_ratios[b] = energy[mask].sum() / total
    return counts, energy_ratios


def main():
    parser = argparse.ArgumentParser(
        description="Show component-count bucketing of SVD components on sample test images."
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--floor", type=float, default=0.05,
                         help="Minimum per-group inclusion probability.")
    args = parser.parse_args()

    loader = get_raw_test_loader(args.data_root, batch_size=args.num_samples, num_workers=0)
    images, labels = next(iter(loader))
    images = images[: args.num_samples]

    _, S, _ = svd_decompose(images)  # S: (N, C, K_svd)
    n, c, k_svd = S.shape
    channel_names = ["R", "G", "B"]

    # Bucket assignment only depends on the (fixed) number of singular values,
    # not on their magnitudes, so it's computed once and reused everywhere.
    buckets = quantize_by_component_count(k_svd, args.K)
    sizes = [int((buckets == b).sum()) for b in range(args.K)]
    print(f"K={args.K} buckets, {k_svd} singular values per channel, "
          f"group sizes={sizes} ({n} sample images, {c} channels each)\n")

    overall_counts = torch.zeros(args.K)
    overall_energy = torch.zeros(args.K)
    per_image_energy = []

    for i in range(n):
        print(f"=== Test image {i} (label={labels[i].item()}) ===")
        img_counts = torch.zeros(args.K)
        img_energy = torch.zeros(args.K)
        for ch in range(c):
            counts, energy_ratios = bucket_stats(S[i, ch], buckets, args.K)
            img_counts += counts.float()
            img_energy += energy_ratios
            counts_str = ", ".join(str(v) for v in counts.tolist())
            energy_str = ", ".join(f"{v * 100:.1f}%" for v in energy_ratios.tolist())
            print(f"  channel {channel_names[ch]}: counts=[{counts_str}]  energy=[{energy_str}]")
        img_counts /= c
        img_energy /= c
        overall_counts += img_counts
        overall_energy += img_energy
        per_image_energy.append(img_energy.clone())
        counts_str = ", ".join(f"{v:.1f}" for v in img_counts.tolist())
        energy_str = ", ".join(f"{v * 100:.1f}%" for v in img_energy.tolist())
        print(f"  channel-avg: counts=[{counts_str}]  energy=[{energy_str}]\n")

    overall_counts /= n
    overall_energy /= n
    print("=== Overall average across sampled images/channels ===")
    counts_str = ", ".join(f"{v:.1f}" for v in overall_counts.tolist())
    energy_str = ", ".join(f"{v * 100:.1f}%" for v in overall_energy.tolist())
    print(f"  counts=[{counts_str}]  energy=[{energy_str}]")

    # --- Group inclusion probability design ---
    header = "image   " + "".join(f"B{b:<8}" for b in range(args.K))
    print("\n=== Per-image energy ratio by bucket (basis for p_b) ===")
    print(header)
    for i, e in enumerate(per_image_energy):
        row = f"{i:<8}" + "".join(f"{v * 100:6.1f}%  " for v in e.tolist())
        print(row)
    avg_row = f"{'avg':<8}" + "".join(f"{v * 100:6.1f}%  " for v in overall_energy.tolist())
    print(avg_row)

    p_b, p_b_floored = design_group_probabilities(overall_energy, args.floor)
    print(f"\n=== Group inclusion probability design (floor={args.floor}) ===")
    print(f"{'Bucket':<8}{'p_b':<10}{'p_b_prime':<10}")
    for b in range(args.K):
        print(f"{b:<8}{p_b[b].item():<10.4f}{p_b_floored[b].item():<10.4f}")


if __name__ == "__main__":
    main()
