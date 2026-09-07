"""
Floor sweep of the fixed-vs-adaptive group-probability comparison
(attacks/evaluate_adaptive.py) across floor in {0.05, 0.10, 0.15, 0.20, 0.36}.

Motivation: attacks/evaluate_adaptive.py (floor=0.36 only) found essentially
no difference between the fixed (training-set-calibrated) and adaptive
(per-image, computed from that image's own SVD spectrum) group-probability
variants. attacks/check_adaptive_sanity.py's sanity check suggested why: at
floor=0.36, groups B1..B(K-1) have such tiny natural energy shares (well
under 1%) that they get clamped up to the floor value for virtually every
image regardless of the image's actual spectrum - the only place per-image
variation can still show up is the top group B0, whose range across images
is under 1 percentage point. The hypothesis this sweep tests: lowering floor
should let B1..B(K-1)'s real (if small) per-image differences survive the
floor and show up in p_b', so the fixed-vs-adaptive gap should grow as floor
drops.

For each floor, on the SAME 1000-image subset (seed=42) used throughout this
project:
  1. Sanity stats: per-group (channel-averaged) raw p_b (before flooring)
     across all 1000 images - min/max/mean and the fraction of images whose
     raw share falls below that floor (and therefore gets clamped up to it).
  2. Clean/FGSM/PGD-EOT accuracy, fixed vs adaptive (same attack methodology
     as evaluate_adaptive.py: BPDA for FGSM, EOT N=10 for PGD, eps=4/255).
  3. McNemar's test (paired, same images) for fixed vs adaptive at each of
     clean/FGSM/PGD.

floor=0.36 reuses the already-computed accuracy/McNemar results from the
earlier single-floor run (2026-09-07, attacks/evaluate_adaptive.py) instead
of recrafting attacks - only its sanity stats are (cheaply) recomputed here
so the summary table has all 5 floors' sanity numbers on equal footing.

Resumable: writes results/adaptive_sweep/summary.json after every floor, and
skips any floor already present there on restart (this repo's sweeps have
hit system-memory-driven crashes before - see PROGRESS.md's 10-step sweep
section - so no floor's ~5-8 minutes of work is lost to a crash elsewhere).
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from vgg16_cifar10_baseline import VGG16Cifar
from rsr import CIFAR_MEAN, CIFAR_STD, svd_decompose
from proposed import (
    calibrate_group_probabilities,
    evaluate_proposed,
    evaluate_proposed_adaptive,
    compute_adaptive_group_probs,
)

from attacks import fgsm_attack, eot_pgd_attack
from evaluate import get_random_test_subset, make_loader, craft_adversarial
from evaluate_adaptive import (
    fixed_forward_fn,
    adaptive_forward_fn,
    per_image_correct_fixed,
    per_image_correct_adaptive,
    mcnemar_test,
    print_mcnemar,
)

# Cached from the floor=0.36 run already recorded in PROGRESS.md
# (attacks/evaluate_adaptive.py, same 1000 images/seed/K/eps/PGD settings).
CACHED_FLOOR_036 = {
    "fixed": {"clean": 89.10, "fgsm": 31.80, "pgd": 2.80},
    "adaptive": {"clean": 89.20, "fgsm": 31.90, "pgd": 2.50},
    "mcnemar": {
        "clean": {"a_only": 1, "b_only": 0, "n_discordant": 1, "p_value": 1.0},
        "fgsm": {"a_only": 2, "b_only": 1, "n_discordant": 3, "p_value": 1.0},
        "pgd": {"a_only": 0, "b_only": 3, "n_discordant": 3, "p_value": 0.25},
    },
}


def load_summary(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_summary(path, summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, path)


@torch.no_grad()
def compute_sanity_stats(loader, buckets, K, floor, device):
    """Per-group (channel-averaged) raw p_b (unfloored) across every image in
    loader: min/max/mean and the fraction of images that fall below `floor`
    (and would therefore get clamped up to it under this floor)."""
    all_p_b_img = []
    for images, _ in loader:
        images = images.to(device)
        U, S, Vh = svd_decompose(images)
        p_b, _ = compute_adaptive_group_probs(S, buckets, 0.0)
        all_p_b_img.append(p_b.mean(dim=1).cpu())  # (n, K) channel-avg
        del images, U, S, Vh, p_b
    all_p_b_img = torch.cat(all_p_b_img, dim=0)  # (N, K)

    per_group = []
    for b in range(K):
        vals = all_p_b_img[:, b]
        clamp_frac = float((vals < floor).float().mean().item())
        per_group.append({
            "group": b,
            "clamp_fraction": clamp_frac,
            "min": float(vals.min().item()),
            "max": float(vals.max().item()),
            "mean": float(vals.mean().item()),
        })
    return per_group


def print_sanity(floor, sanity):
    print(f"  Sanity stats (channel-avg raw p_b before flooring, floor={floor}):")
    for g in sanity:
        print(f"    B{g['group']}: below-floor(clamped) fraction={g['clamp_fraction'] * 100:5.1f}%  "
              f"min={g['min'] * 100:6.3f}%  max={g['max'] * 100:6.3f}%  mean={g['mean'] * 100:6.3f}%",
              flush=True)


def run_floor_full(floor, model, device, mean, std, clean_loader, eot_loader, args):
    """Full pipeline for one floor: calibration, sanity stats, clean/FGSM/
    PGD-EOT for both fixed and adaptive, McNemar's test."""
    buckets, p_b, p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, floor, args.num_calib, device, args.seed
    )
    print(f"  fixed p_b_prime = {[round(v, 4) for v in p_b_floored.tolist()]}", flush=True)

    sanity = compute_sanity_stats(clean_loader, buckets, args.K, floor, device)
    print_sanity(floor, sanity)

    gen_fixed_attack = torch.Generator().manual_seed(args.seed)
    gen_adapt_attack = torch.Generator().manual_seed(args.seed)
    fwd_fixed = fixed_forward_fn(model, mean, std, buckets, p_b_floored, gen_fixed_attack)
    fwd_adaptive = adaptive_forward_fn(model, mean, std, buckets, floor, gen_adapt_attack)

    fixed_clean_acc, _ = evaluate_proposed(model, clean_loader, device, args.R, buckets, p_b_floored, args.seed)
    adaptive_clean_acc, _ = evaluate_proposed_adaptive(model, clean_loader, device, args.R, buckets, floor, args.seed)
    print(f"  Clean: fixed={fixed_clean_acc:.2f}%  adaptive={adaptive_clean_acc:.2f}%", flush=True)

    print("  [1/4] Fixed FGSM (BPDA)...", flush=True)
    fixed_fgsm_images, fixed_fgsm_labels = craft_adversarial(fwd_fixed, clean_loader, fgsm_attack, device, eps=args.eps)
    print("  [2/4] Fixed PGD-EOT...", flush=True)
    fixed_pgd_images, fixed_pgd_labels = craft_adversarial(
        fwd_fixed, eot_loader, eot_pgd_attack, device,
        eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps, eot_samples=args.eot_samples,
    )
    print("  [3/4] Adaptive FGSM (BPDA)...", flush=True)
    adaptive_fgsm_images, adaptive_fgsm_labels = craft_adversarial(fwd_adaptive, clean_loader, fgsm_attack, device, eps=args.eps)
    print("  [4/4] Adaptive PGD-EOT...", flush=True)
    adaptive_pgd_images, adaptive_pgd_labels = craft_adversarial(
        fwd_adaptive, eot_loader, eot_pgd_attack, device,
        eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps, eot_samples=args.eot_samples,
    )

    fixed_fgsm_acc = evaluate_proposed(
        model, make_loader(fixed_fgsm_images, fixed_fgsm_labels, args.clean_batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )[0]
    fixed_pgd_acc = evaluate_proposed(
        model, make_loader(fixed_pgd_images, fixed_pgd_labels, args.clean_batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )[0]
    adaptive_fgsm_acc = evaluate_proposed_adaptive(
        model, make_loader(adaptive_fgsm_images, adaptive_fgsm_labels, args.clean_batch_size),
        device, args.R, buckets, floor, args.seed
    )[0]
    adaptive_pgd_acc = evaluate_proposed_adaptive(
        model, make_loader(adaptive_pgd_images, adaptive_pgd_labels, args.clean_batch_size),
        device, args.R, buckets, floor, args.seed
    )[0]
    print(f"  FGSM: fixed={fixed_fgsm_acc:.2f}%  adaptive={adaptive_fgsm_acc:.2f}%", flush=True)
    print(f"  PGD-EOT: fixed={fixed_pgd_acc:.2f}%  adaptive={adaptive_pgd_acc:.2f}%", flush=True)

    correct_fixed_clean = per_image_correct_fixed(model, clean_loader, device, args.R, buckets, p_b_floored, args.seed)
    correct_adaptive_clean = per_image_correct_adaptive(model, clean_loader, device, args.R, buckets, floor, args.seed)
    correct_fixed_fgsm = per_image_correct_fixed(
        model, make_loader(fixed_fgsm_images, fixed_fgsm_labels, args.clean_batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )
    correct_adaptive_fgsm = per_image_correct_adaptive(
        model, make_loader(adaptive_fgsm_images, adaptive_fgsm_labels, args.clean_batch_size),
        device, args.R, buckets, floor, args.seed
    )
    correct_fixed_pgd = per_image_correct_fixed(
        model, make_loader(fixed_pgd_images, fixed_pgd_labels, args.clean_batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )
    correct_adaptive_pgd = per_image_correct_adaptive(
        model, make_loader(adaptive_pgd_images, adaptive_pgd_labels, args.clean_batch_size),
        device, args.R, buckets, floor, args.seed
    )

    mc_clean = mcnemar_test(correct_adaptive_clean, correct_fixed_clean, "Adaptive", "Fixed")
    mc_fgsm = mcnemar_test(correct_adaptive_fgsm, correct_fixed_fgsm, "Adaptive", "Fixed")
    mc_pgd = mcnemar_test(correct_adaptive_pgd, correct_fixed_pgd, "Adaptive", "Fixed")
    print("  McNemar (Adaptive vs Fixed):")
    print_mcnemar(mc_clean)
    print_mcnemar(mc_fgsm)
    print_mcnemar(mc_pgd)

    del fixed_fgsm_images, fixed_fgsm_labels, fixed_pgd_images, fixed_pgd_labels
    del adaptive_fgsm_images, adaptive_fgsm_labels, adaptive_pgd_images, adaptive_pgd_labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    def mc_to_dict(mc):
        return {"a_only": mc["a_correct_b_wrong"], "b_only": mc["a_wrong_b_correct"],
                "n_discordant": mc["n_discordant"], "p_value": mc["p_value"]}

    return {
        "sanity": sanity,
        "fixed": {"clean": fixed_clean_acc, "fgsm": fixed_fgsm_acc, "pgd": fixed_pgd_acc},
        "adaptive": {"clean": adaptive_clean_acc, "fgsm": adaptive_fgsm_acc, "pgd": adaptive_pgd_acc},
        "mcnemar": {"clean": mc_to_dict(mc_clean), "fgsm": mc_to_dict(mc_fgsm), "pgd": mc_to_dict(mc_pgd)},
    }


def run_floor_cached(floor, device, clean_loader, args):
    """floor=0.36: reuse the already-computed accuracy/McNemar numbers, only
    (cheaply) recompute sanity stats so the summary table is complete."""
    buckets, p_b, p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, floor, args.num_calib, device, args.seed
    )
    print(f"  fixed p_b_prime = {[round(v, 4) for v in p_b_floored.tolist()]}", flush=True)
    sanity = compute_sanity_stats(clean_loader, buckets, args.K, floor, device)
    print_sanity(floor, sanity)
    print("  (reusing cached clean/FGSM/PGD-EOT/McNemar results from the earlier floor=0.36 run)", flush=True)
    return {"sanity": sanity, **CACHED_FLOOR_036}


def main():
    parser = argparse.ArgumentParser(
        description="Floor sweep of the fixed-vs-adaptive group-probability comparison."
    )
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--clean-batch-size", type=int, default=128)
    parser.add_argument("--eot-batch-size", type=int, default=25)
    parser.add_argument("--eot-samples", type=int, default=10)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--floors", type=str, default="0.05,0.10,0.15,0.20,0.36")
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-path", type=str, default="./results/adaptive_sweep/summary.json")
    parser.add_argument("--force-recompute", action="store_true",
                         help="Recompute even floors already present in the summary file (or floor=0.36 "
                              "from scratch instead of reusing cached results).")
    args = parser.parse_args()

    floors = [float(f) for f in args.floors.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = VGG16Cifar(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)

    print(f"Sampling {args.num_samples} random test images (seed={args.seed})...", flush=True)
    images, labels = get_random_test_subset(args.data_root, args.num_samples, args.seed)
    clean_loader = make_loader(images, labels, args.clean_batch_size)
    eot_loader = make_loader(images, labels, args.eot_batch_size)

    summary = load_summary(args.summary_path)

    for floor in floors:
        key = f"{floor:g}"  # e.g. 0.005 -> "0.005", 0.01 -> "0.01" - :.2f collided 0.005/0.01 into "0.01"
        if key in summary and not args.force_recompute:
            print(f"\n{'=' * 20} floor={floor} (skipping, already in summary) {'=' * 20}", flush=True)
            continue

        print(f"\n{'=' * 20} floor={floor} {'=' * 20}", flush=True)
        start = time.time()
        if abs(floor - 0.36) < 1e-9 and not args.force_recompute:
            result = run_floor_cached(floor, device, clean_loader, args)
        else:
            result = run_floor_full(floor, model, device, mean, std, clean_loader, eot_loader, args)
        print(f"  floor={floor} done in {time.time() - start:.1f}s", flush=True)

        summary[key] = result
        save_summary(args.summary_path, summary)
        print(f"  saved to {args.summary_path}", flush=True)

    print("\n\n" + "=" * 70)
    print("=== Floor sweep summary ===")
    print(f"{'floor':<8}{'fixed C/F/P':<20}{'adapt C/F/P':<20}{'McNemar p (C/F/P)':<24}")
    for floor in floors:
        key = f"{floor:g}"
        r = summary[key]
        fx, ad, mc = r["fixed"], r["adaptive"], r["mcnemar"]
        fixed_str = f"{fx['clean']:.1f}/{fx['fgsm']:.1f}/{fx['pgd']:.1f}"
        adapt_str = f"{ad['clean']:.1f}/{ad['fgsm']:.1f}/{ad['pgd']:.1f}"
        p_str = f"{mc['clean']['p_value']:.3f}/{mc['fgsm']['p_value']:.3f}/{mc['pgd']['p_value']:.3f}"
        print(f"{key:<8}{fixed_str:<20}{adapt_str:<20}{p_str:<24}")


if __name__ == "__main__":
    main()
