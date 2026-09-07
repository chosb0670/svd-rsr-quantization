"""
Fixed (training-set-calibrated) vs adaptive (per-image, computed from that
image's own SVD spectrum) group inclusion probabilities for the proposed
defense (models/proposed.py). See models/proposed.py's
compute_adaptive_group_probs docstring and attacks/check_adaptive_sanity.py
for why this variant was added: the original "proposed" pipeline calibrates
p_b' once from 500 training images and reuses that single vector for every
test image, discarding the "probabilities track this image's own spectrum"
property original RSR has. This script compares the two head-to-head on the
same 1000-image subset (seed=42), K=5, floor=0.36:
  - clean accuracy
  - FGSM (eps=4/255) and PGD-EOT (T=20, alpha=2/255, eot_samples=10) robust
    accuracy, attacked via BPDA (FGSM) / EOT (PGD) since both pipelines'
    reconstruction is a discrete, per-trial-random operation
  - McNemar's test (paired, same images) on fixed vs adaptive correctness
    for each of clean/FGSM/PGD

Memory/perf notes follow attacks/evaluate_eot.py: num_workers=0, all
accuracy/majority-vote eval under @torch.no_grad(), explicit del +
torch.cuda.empty_cache() after each crafted batch, small batch size since
EOT tiles eot_samples reconstructions per step.
"""

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from vgg16_cifar10_baseline import VGG16Cifar
from rsr import CIFAR_MEAN, CIFAR_STD, svd_decompose
from quantization import quantize_by_component_count
from proposed import (
    calibrate_group_probabilities,
    sample_group_mask,
    sample_group_mask_from_probs,
    reconstruct_with_group_mask,
    compute_adaptive_group_probs,
    evaluate_proposed,
    evaluate_proposed_adaptive,
)

from attacks import fgsm_attack, eot_fgsm_attack, eot_pgd_attack, bpda_apply
from evaluate import get_random_test_subset, make_loader, craft_adversarial


def fixed_forward_fn(model, mean, std, buckets, p_b_floored, generator):
    def fn(x):
        def recon_once(xx):
            U, S, Vh = svd_decompose(xx)
            n, c = xx.size(0), xx.size(1)
            mask = sample_group_mask(p_b_floored.cpu(), n, c, generator).to(xx.device)
            return reconstruct_with_group_mask(U, S, Vh, buckets, mask)
        recon = bpda_apply(x, recon_once)
        return model((recon - mean) / std)
    return fn


def adaptive_forward_fn(model, mean, std, buckets, floor, generator):
    """Like fixed_forward_fn, but p_b' is recomputed from the input's own
    SVD spectrum on every call, same as evaluate_proposed_adaptive."""
    def fn(x):
        def recon_once(xx):
            U, S, Vh = svd_decompose(xx)
            _, p_b_floored = compute_adaptive_group_probs(S, buckets, floor)
            mask = sample_group_mask_from_probs(p_b_floored.cpu(), generator).to(xx.device)
            return reconstruct_with_group_mask(U, S, Vh, buckets, mask)
        recon = bpda_apply(x, recon_once)
        return model((recon - mean) / std)
    return fn


@torch.no_grad()
def per_image_correct_fixed(model, loader, device, R, buckets, p_b_floored, seed):
    model.eval()
    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)
    generator = torch.Generator().manual_seed(seed)
    buckets = buckets.to(device)
    p_b_floored_cpu = p_b_floored.cpu()

    correctness = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        n, c = images.size(0), images.size(1)
        U, S, Vh = svd_decompose(images)
        recons = []
        for _ in range(R):
            mask = sample_group_mask(p_b_floored_cpu, n, c, generator).to(device)
            recons.append(reconstruct_with_group_mask(U, S, Vh, buckets, mask))
        recons = torch.stack(recons, dim=0)
        batched = recons.reshape(R * n, *images.shape[1:])
        normalized = (batched - mean) / std
        logits = model(normalized)
        preds = logits.argmax(dim=1).reshape(R, n)
        majority = torch.mode(preds, dim=0).values
        correctness.append((majority == targets).cpu())
    return torch.cat(correctness)


@torch.no_grad()
def per_image_correct_adaptive(model, loader, device, R, buckets, floor, seed):
    model.eval()
    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)
    generator = torch.Generator().manual_seed(seed)
    buckets = buckets.to(device)

    correctness = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        n, c = images.size(0), images.size(1)
        U, S, Vh = svd_decompose(images)
        _, p_b_floored = compute_adaptive_group_probs(S, buckets, floor)
        p_b_floored_cpu = p_b_floored.cpu()
        recons = []
        for _ in range(R):
            mask = sample_group_mask_from_probs(p_b_floored_cpu, generator).to(device)
            recons.append(reconstruct_with_group_mask(U, S, Vh, buckets, mask))
        recons = torch.stack(recons, dim=0)
        batched = recons.reshape(R * n, *images.shape[1:])
        normalized = (batched - mean) / std
        logits = model(normalized)
        preds = logits.argmax(dim=1).reshape(R, n)
        majority = torch.mode(preds, dim=0).values
        correctness.append((majority == targets).cpu())
    return torch.cat(correctness)


def mcnemar_test(correct_a: torch.Tensor, correct_b: torch.Tensor, label_a: str, label_b: str):
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n_disc = a_only + b_only

    if n_disc == 0:
        stat, p_value, method = 0.0, 1.0, "degenerate (no discordant pairs)"
    elif n_disc < 25:
        k = min(a_only, b_only)
        p_value = min(1.0, sum(math.comb(n_disc, i) for i in range(0, k + 1)) * 2 / (2 ** n_disc))
        stat, method = None, "exact binomial"
    else:
        stat = (abs(a_only - b_only) - 1) ** 2 / n_disc
        p_value = math.erfc(math.sqrt(stat / 2))
        method = "chi-square (continuity-corrected)"

    return {
        "label_a": label_a, "label_b": label_b,
        "a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only,
        "n_discordant": n_disc, "statistic": stat, "p_value": p_value, "method": method,
    }


def print_mcnemar(result):
    r = result
    stat_str = f"{r['statistic']:.3f}" if r["statistic"] is not None else "n/a"
    print(f"  {r['label_a']} vs {r['label_b']}:")
    print(f"    discordant pairs: {r['label_a']}-only-correct={r['a_correct_b_wrong']}, "
          f"{r['label_b']}-only-correct={r['a_wrong_b_correct']} (n_discordant={r['n_discordant']})")
    print(f"    McNemar ({r['method']}): statistic={stat_str}, p={r['p_value']:.4g}, "
          f"{'significant (p<0.05)' if r['p_value'] < 0.05 else 'NOT significant (p>=0.05)'}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Fixed vs adaptive group-probability comparison for the proposed defense."
    )
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--clean-batch-size", type=int, default=128)
    parser.add_argument("--eot-batch-size", type=int, default=25,
                         help="Kept small since EOT tiles eot_samples reconstructions per step.")
    parser.add_argument("--eot-samples", type=int, default=10)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--floor", type=float, default=0.36)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    print(f"Calibrating FIXED proposed defense (K={args.K}, floor={args.floor}) "
          f"on {args.num_calib} training images...", flush=True)
    buckets, p_b, p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, args.floor, args.num_calib, device, args.seed
    )
    print(f"  fixed p_b_prime = {[round(v, 4) for v in p_b_floored.tolist()]}", flush=True)

    gen_fixed_clean = torch.Generator().manual_seed(args.seed)
    gen_adapt_clean = torch.Generator().manual_seed(args.seed)
    gen_fixed_attack = torch.Generator().manual_seed(args.seed)
    gen_adapt_attack = torch.Generator().manual_seed(args.seed)

    fwd_fixed = fixed_forward_fn(model, mean, std, buckets, p_b_floored, gen_fixed_attack)
    fwd_adaptive = adaptive_forward_fn(model, mean, std, buckets, args.floor, gen_adapt_attack)

    print("\n=== Clean accuracy (1000 images) ===", flush=True)
    start = time.time()
    fixed_clean_acc, fixed_keep_ratio = evaluate_proposed(
        model, clean_loader, device, args.R, buckets, p_b_floored, args.seed
    )
    print(f"  Fixed:    {fixed_clean_acc:.2f}%  (avg keep_ratio={fixed_keep_ratio:.4f}) "
          f"({time.time() - start:.1f}s)", flush=True)

    start = time.time()
    adaptive_clean_acc, adaptive_keep_ratio = evaluate_proposed_adaptive(
        model, clean_loader, device, args.R, buckets, args.floor, args.seed
    )
    print(f"  Adaptive: {adaptive_clean_acc:.2f}%  (avg keep_ratio={adaptive_keep_ratio:.4f}) "
          f"({time.time() - start:.1f}s)", flush=True)

    print(f"\n--- Crafting adversarial examples at eps={args.eps:.4f} ---", flush=True)

    print("[1/4] Fixed, FGSM (BPDA)...", flush=True)
    fixed_fgsm_images, fixed_fgsm_labels = craft_adversarial(
        fwd_fixed, clean_loader, fgsm_attack, device, eps=args.eps
    )
    print("[2/4] Fixed, PGD-EOT...", flush=True)
    fixed_pgd_images, fixed_pgd_labels = craft_adversarial(
        fwd_fixed, eot_loader, eot_pgd_attack, device,
        eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps, eot_samples=args.eot_samples,
    )
    print("[3/4] Adaptive, FGSM (BPDA)...", flush=True)
    adaptive_fgsm_images, adaptive_fgsm_labels = craft_adversarial(
        fwd_adaptive, clean_loader, fgsm_attack, device, eps=args.eps
    )
    print("[4/4] Adaptive, PGD-EOT...", flush=True)
    adaptive_pgd_images, adaptive_pgd_labels = craft_adversarial(
        fwd_adaptive, eot_loader, eot_pgd_attack, device,
        eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps, eot_samples=args.eot_samples,
    )

    print("\n--- Evaluating robust accuracy (real R=10 majority-vote pipelines) ---", flush=True)
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
        device, args.R, buckets, args.floor, args.seed
    )[0]
    adaptive_pgd_acc = evaluate_proposed_adaptive(
        model, make_loader(adaptive_pgd_images, adaptive_pgd_labels, args.clean_batch_size),
        device, args.R, buckets, args.floor, args.seed
    )[0]

    print(f"\n=== Summary (num_samples={args.num_samples}, eps={args.eps:.4f}, "
          f"eot_samples={args.eot_samples}) ===")
    print(f"{'Method':<20}{'Clean':>10}{'FGSM':>10}{'PGD-EOT':>10}")
    print(f"{'Fixed':<20}{fixed_clean_acc:>9.2f}%{fixed_fgsm_acc:>9.2f}%{fixed_pgd_acc:>9.2f}%")
    print(f"{'Adaptive':<20}{adaptive_clean_acc:>9.2f}%{adaptive_fgsm_acc:>9.2f}%{adaptive_pgd_acc:>9.2f}%")

    print("\n--- Per-image correctness for McNemar's test (paired, same 1000 images) ---", flush=True)
    correct_fixed_clean = per_image_correct_fixed(model, clean_loader, device, args.R, buckets, p_b_floored, args.seed)
    correct_adaptive_clean = per_image_correct_adaptive(model, clean_loader, device, args.R, buckets, args.floor, args.seed)
    correct_fixed_fgsm = per_image_correct_fixed(
        model, make_loader(fixed_fgsm_images, fixed_fgsm_labels, args.clean_batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )
    correct_adaptive_fgsm = per_image_correct_adaptive(
        model, make_loader(adaptive_fgsm_images, adaptive_fgsm_labels, args.clean_batch_size),
        device, args.R, buckets, args.floor, args.seed
    )
    correct_fixed_pgd = per_image_correct_fixed(
        model, make_loader(fixed_pgd_images, fixed_pgd_labels, args.clean_batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )
    correct_adaptive_pgd = per_image_correct_adaptive(
        model, make_loader(adaptive_pgd_images, adaptive_pgd_labels, args.clean_batch_size),
        device, args.R, buckets, args.floor, args.seed
    )

    print(f"\n  Sanity check (should match aggregate accuracies above):")
    print(f"    Fixed:    clean={100 * correct_fixed_clean.float().mean().item():.2f}%  "
          f"FGSM={100 * correct_fixed_fgsm.float().mean().item():.2f}%  "
          f"PGD={100 * correct_fixed_pgd.float().mean().item():.2f}%")
    print(f"    Adaptive: clean={100 * correct_adaptive_clean.float().mean().item():.2f}%  "
          f"FGSM={100 * correct_adaptive_fgsm.float().mean().item():.2f}%  "
          f"PGD={100 * correct_adaptive_pgd.float().mean().item():.2f}%")

    print("\n=== McNemar's test: Adaptive vs Fixed (paired, same images) ===")
    print("\nClean:")
    print_mcnemar(mcnemar_test(correct_adaptive_clean, correct_fixed_clean, "Adaptive", "Fixed"))
    print("\nFGSM:")
    print_mcnemar(mcnemar_test(correct_adaptive_fgsm, correct_fixed_fgsm, "Adaptive", "Fixed"))
    print("\nPGD-EOT:")
    print_mcnemar(mcnemar_test(correct_adaptive_pgd, correct_fixed_pgd, "Adaptive", "Fixed"))

    if device.type == "cuda":
        print(f"\nGPU memory at end: {torch.cuda.memory_allocated() / 1e6:.1f}MB allocated, "
              f"{torch.cuda.max_memory_allocated() / 1e6:.1f}MB peak")


if __name__ == "__main__":
    main()
