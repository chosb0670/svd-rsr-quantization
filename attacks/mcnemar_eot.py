"""
Paired significance check (McNemar's test) for the eps=4/255 EOT comparisons
in PROGRESS.md, which were originally assessed with an independent two-
proportion z-test. That test assumes two independent samples, but all three
methods here (single-sample BPDA, EOT, deterministic) are evaluated on the
SAME 1000-image subset - per-image correctness across methods is PAIRED
data, so McNemar's test is the statistically appropriate tool: it looks only
at the images where the two methods disagree (one right, one wrong) rather
than treating the two accuracy numbers as independent proportions.

Recomputes, at eps=4/255, per-image correctness (not just aggregate
accuracy) for:
  - Proposed, single-sample BPDA: FGSM, PGD
  - Proposed, EOT (N=10): FGSM, PGD
  - Deterministic, exact gradient: FGSM, PGD

and runs McNemar's test on the four pairs PROGRESS.md asks about:
  Q1a: EOT-FGSM vs single-BPDA-FGSM (proposed)
  Q1b: EOT-PGD  vs single-BPDA-PGD  (proposed)
  Q2a: EOT-FGSM (proposed) vs Deterministic-FGSM
  Q2b: EOT-PGD  (proposed) vs Deterministic-PGD
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from vgg16_cifar10_baseline import VGG16Cifar
from rsr import CIFAR_MEAN, CIFAR_STD, svd_decompose
from proposed import calibrate_group_probabilities, sample_group_mask, reconstruct_with_group_mask
from deterministic import enumerate_combinations, combination_probabilities, deterministic_forward

from attacks import fgsm_attack, pgd_attack, eot_fgsm_attack, eot_pgd_attack
from evaluate import get_random_test_subset, make_loader, craft_adversarial, proposed_forward_fn
from evaluate_deterministic import deterministic_attack_forward_fn


@torch.no_grad()
def per_image_correct_proposed(model, loader, device, R, buckets, p_b_floored, seed):
    """Per-image correctness (N,) bool under the REAL R=10 majority-vote
    pipeline (same logic as proposed.evaluate_proposed, but returns the
    per-image vector instead of just the aggregate accuracy)."""
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
def per_image_correct_deterministic(model, loader, device, buckets, patterns, pi):
    """Per-image correctness (N,) bool under the exact-expectation pipeline."""
    model.eval()
    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)
    correctness = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        probs = deterministic_forward(model, images, buckets, patterns, pi, mean, std)
        preds = probs.argmax(dim=-1)
        correctness.append((preds == targets).cpu())
    return torch.cat(correctness)


def mcnemar_test(correct_a: torch.Tensor, correct_b: torch.Tensor, label_a: str, label_b: str):
    """correct_a, correct_b: (N,) bool, same image order. Returns a dict with
    the discordant counts and a two-sided p-value: exact binomial test for
    small discordant counts (<25, where the chi-square approximation is
    unreliable), continuity-corrected chi-square (1 dof) otherwise."""
    a_only = int((correct_a & ~correct_b).sum())   # a correct, b wrong
    b_only = int((~correct_a & correct_b).sum())   # a wrong, b correct
    n_disc = a_only + b_only

    if n_disc == 0:
        stat, p_value, method = 0.0, 1.0, "degenerate (no discordant pairs)"
    elif n_disc < 25:
        k = min(a_only, b_only)
        p_value = min(1.0, sum(math.comb(n_disc, i) for i in range(0, k + 1)) * 2 / (2 ** n_disc))
        stat, method = None, "exact binomial"
    else:
        stat = (abs(a_only - b_only) - 1) ** 2 / n_disc
        p_value = math.erfc(math.sqrt(stat / 2))  # chi-square(1 dof) survival function
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
          f"{'significant (p<0.05)' if r['p_value'] < 0.05 else 'NOT significant (p>=0.05)'}")


def main():
    parser = argparse.ArgumentParser(
        description="McNemar's test for paired EOT/BPDA/deterministic comparisons at a given eps."
    )
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eot-batch-size", type=int, default=25)
    parser.add_argument("--det-batch-size", type=int, default=32,
                         help="Kept small: deterministic evaluates all 2^K patterns per image "
                              "(effective batch = det_batch_size * 2^K).")
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

    print(f"Calibrating (K={args.K}, floor={args.floor}) on {args.num_calib} training images...", flush=True)
    buckets, p_b, p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, args.floor, args.num_calib, device, args.seed
    )
    patterns = enumerate_combinations(args.K)
    pi = combination_probabilities(patterns, p_b_floored)

    gen_bpda = torch.Generator().manual_seed(args.seed)
    gen_eot = torch.Generator().manual_seed(args.seed)

    proposed_loader = make_loader(images, labels, args.batch_size)
    eot_loader = make_loader(images, labels, args.eot_batch_size)
    det_loader = make_loader(images, labels, args.det_batch_size)

    fwd_bpda = proposed_forward_fn(model, mean, std, buckets, p_b_floored, gen_bpda)
    fwd_eot = proposed_forward_fn(model, mean, std, buckets, p_b_floored, gen_eot)
    fwd_det = deterministic_attack_forward_fn(model, buckets, patterns, pi, mean, std)

    print(f"\n--- Crafting adversarial examples at eps={args.eps:.4f} ---", flush=True)

    print("[1/6] Proposed, single-sample BPDA, FGSM...", flush=True)
    bpda_fgsm_images, bpda_fgsm_labels = craft_adversarial(fwd_bpda, proposed_loader, fgsm_attack, device, eps=args.eps)

    print("[2/6] Proposed, single-sample BPDA, PGD...", flush=True)
    bpda_pgd_images, bpda_pgd_labels = craft_adversarial(
        fwd_bpda, proposed_loader, pgd_attack, device, eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps
    )

    print("[3/6] Proposed, EOT, FGSM...", flush=True)
    eot_fgsm_images, eot_fgsm_labels = craft_adversarial(
        fwd_eot, eot_loader, eot_fgsm_attack, device, eps=args.eps, eot_samples=args.eot_samples
    )

    print("[4/6] Proposed, EOT, PGD...", flush=True)
    eot_pgd_images, eot_pgd_labels = craft_adversarial(
        fwd_eot, eot_loader, eot_pgd_attack, device,
        eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps, eot_samples=args.eot_samples,
    )

    print("[5/6] Deterministic, FGSM...", flush=True)
    det_fgsm_images, det_fgsm_labels = craft_adversarial(
        fwd_det, det_loader, fgsm_attack, device, eps=args.eps, loss_fn=torch.nn.functional.nll_loss
    )

    print("[6/6] Deterministic, PGD...", flush=True)
    det_pgd_images, det_pgd_labels = craft_adversarial(
        fwd_det, det_loader, pgd_attack, device,
        eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps, loss_fn=torch.nn.functional.nll_loss,
    )

    print("\n--- Evaluating per-image correctness (real pipelines) ---", flush=True)
    correct_bpda_fgsm = per_image_correct_proposed(
        model, make_loader(bpda_fgsm_images, bpda_fgsm_labels, args.batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )
    correct_bpda_pgd = per_image_correct_proposed(
        model, make_loader(bpda_pgd_images, bpda_pgd_labels, args.batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )
    correct_eot_fgsm = per_image_correct_proposed(
        model, make_loader(eot_fgsm_images, eot_fgsm_labels, args.batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )
    correct_eot_pgd = per_image_correct_proposed(
        model, make_loader(eot_pgd_images, eot_pgd_labels, args.batch_size),
        device, args.R, buckets, p_b_floored, args.seed
    )
    correct_det_fgsm = per_image_correct_deterministic(
        model, make_loader(det_fgsm_images, det_fgsm_labels, args.det_batch_size), device, buckets, patterns, pi
    )
    correct_det_pgd = per_image_correct_deterministic(
        model, make_loader(det_pgd_images, det_pgd_labels, args.det_batch_size), device, buckets, patterns, pi
    )

    print(f"\n  Sanity check (should match earlier aggregate results at eps={args.eps:.4f}):")
    print(f"    Proposed BPDA:  FGSM={100 * correct_bpda_fgsm.float().mean().item():.2f}%  "
          f"PGD={100 * correct_bpda_pgd.float().mean().item():.2f}%")
    print(f"    Proposed EOT:   FGSM={100 * correct_eot_fgsm.float().mean().item():.2f}%  "
          f"PGD={100 * correct_eot_pgd.float().mean().item():.2f}%")
    print(f"    Deterministic:  FGSM={100 * correct_det_fgsm.float().mean().item():.2f}%  "
          f"PGD={100 * correct_det_pgd.float().mean().item():.2f}%")

    print("\n=== McNemar's test (paired, same 1000 images) ===")
    print("\nQ1a: EOT vs single-sample BPDA (proposed), FGSM")
    print_mcnemar(mcnemar_test(correct_eot_fgsm, correct_bpda_fgsm, "EOT", "single-BPDA"))

    print("\nQ1b: EOT vs single-sample BPDA (proposed), PGD")
    print_mcnemar(mcnemar_test(correct_eot_pgd, correct_bpda_pgd, "EOT", "single-BPDA"))

    print("\nQ2a: Proposed-EOT vs Deterministic, FGSM")
    print_mcnemar(mcnemar_test(correct_eot_fgsm, correct_det_fgsm, "Proposed-EOT", "Deterministic"))

    print("\nQ2b: Proposed-EOT vs Deterministic, PGD")
    print_mcnemar(mcnemar_test(correct_eot_pgd, correct_det_pgd, "Proposed-EOT", "Deterministic"))


if __name__ == "__main__":
    main()
