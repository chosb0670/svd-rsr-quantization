"""
10-step hyperparameter sweep over the proposed (random, EOT) defense and its
deterministic exact-expectation control, building on attacks/evaluate_eot.py
and attacks/mcnemar_eot.py.

For every (config, method, attack) combination, saves the per-image
correctness (0/1) array under results/sweep/<section>/ as .npy, so McNemar
tests (or any other paired analysis) can be recomputed later without
rerunning any attack. A running results/sweep/summary.json also records
every aggregate accuracy as it's produced.

Sections (same 1000-image subset, seed=42, throughout):
  A) floor sweep (main): K=5, eps=4/255, floor in
     {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.36}. Proposed(EOT, N=10) and
     Deterministic, Clean/FGSM-EOT/PGD-EOT (PGD: T=20, alpha=2/255) each.
  B) K sweep: eps=4/255, floor solved analytically per K to hit
     keep_ratio~=0.5 (same formula validated in PROGRESS.md: keep_ratio =
     (size0*p_b0 + (32-size0)*floor)/32). K in {3, 8, 16}; K=5 reuses A's
     floor=0.36 point. Proposed(EOT) for all three K; Deterministic only
     for K=3 and K=8 - K=16's exact expectation needs 2^16=65536 pattern
     enumerations, which is computationally infeasible within a reasonable
     time budget even with batching (would need many hours for FGSM/PGD),
     so it is skipped per an explicit user decision.
  C) eot_samples sweep: floor=0.05, K=5, eps=4/255, eot_samples in
     {5, 20, 40}; N=10 reuses A's floor=0.05 point. Proposed only
     (clean accuracy doesn't depend on eot_samples, so only FGSM-EOT/PGD-EOT
     are measured here).
  D) eps cross-check: floor=0.05, K=5, N=10, eps=8/255 (freshly computed).
     floor=0.36/eps=8/255 reuses the existing result already in PROGRESS.md
     (Proposed EOT: clean 89.00/FGSM 14.00/PGD 0.10; Deterministic: clean
     87.90/FGSM 18.70/PGD 0.20). Proposed + Deterministic, all three metrics.
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from vgg16_cifar10_baseline import VGG16Cifar
from rsr import CIFAR_MEAN, CIFAR_STD
from proposed import calibrate_group_probabilities
from deterministic import enumerate_combinations, combination_probabilities

from attacks import fgsm_attack, pgd_attack, eot_fgsm_attack, eot_pgd_attack
from evaluate import get_random_test_subset, make_loader, craft_adversarial, proposed_forward_fn
from evaluate_deterministic import deterministic_attack_forward_fn
from mcnemar_eot import per_image_correct_proposed, per_image_correct_deterministic

SWEEP_ROOT = os.path.join("results", "sweep")
SUMMARY_PATH = os.path.join(SWEEP_ROOT, "summary.json")


def load_summary():
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    return {}


def save_summary(summary):
    os.makedirs(SWEEP_ROOT, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)


def record(summary, section, config_id, method, attack, accuracy):
    summary.setdefault(section, {}).setdefault(config_id, {}).setdefault(method, {})[attack] = accuracy
    save_summary(summary)


def already_done(summary, section, config_id, method, attack) -> bool:
    """Resume support: a prior run may have been killed (e.g. OOM) partway
    through. Skip any (section, config, method, attack) already recorded in
    summary.json instead of recrafting/re-evaluating it."""
    return attack in summary.get(section, {}).get(config_id, {}).get(method, {})


def save_correctness(section, config_id, method, attack, correct: torch.Tensor) -> float:
    out_dir = os.path.join(SWEEP_ROOT, section)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{config_id}__{method}__{attack}.npy")
    np.save(path, correct.numpy().astype(np.uint8))
    acc = 100.0 * correct.float().mean().item()
    print(f"      saved {path} (acc={acc:.2f}%)", flush=True)
    return acc


def solve_floor_for_keep_ratio(p_b0: float, size0: int, k_total: int = 32, target: float = 0.5) -> float:
    """keep_ratio = (size0*p_b0 + (k_total-size0)*floor) / k_total = target
    => floor = (target*k_total - size0*p_b0) / (k_total - size0)."""
    floor = (target * k_total - size0 * p_b0) / (k_total - size0)
    return max(1e-4, min(1.0, floor))


def run_proposed_config(model, mean, std, device, images, labels, buckets, p_b_floored,
                         eps, alpha, steps, eot_samples, R, seed,
                         batch_size, eot_batch_size, need_clean, section, config_id, summary):
    results = {}
    if need_clean:
        if already_done(summary, section, config_id, "proposed", "clean"):
            print(f"      [skip] proposed/clean already done for {config_id}", flush=True)
            results["clean"] = summary[section][config_id]["proposed"]["clean"]
        else:
            correct_clean = per_image_correct_proposed(
                model, make_loader(images, labels, batch_size), device, R, buckets, p_b_floored, seed
            )
            acc = save_correctness(section, config_id, "proposed", "clean", correct_clean)
            record(summary, section, config_id, "proposed", "clean", acc)
            results["clean"] = acc

    gen = torch.Generator().manual_seed(seed)
    fwd = proposed_forward_fn(model, mean, std, buckets, p_b_floored, gen)
    eot_loader = make_loader(images, labels, eot_batch_size)

    if already_done(summary, section, config_id, "proposed", "fgsm"):
        print(f"      [skip] proposed/fgsm already done for {config_id}", flush=True)
        results["fgsm"] = summary[section][config_id]["proposed"]["fgsm"]
    else:
        fgsm_images, fgsm_labels = craft_adversarial(
            fwd, eot_loader, eot_fgsm_attack, device, eps=eps, eot_samples=eot_samples
        )
        correct_fgsm = per_image_correct_proposed(
            model, make_loader(fgsm_images, fgsm_labels, batch_size), device, R, buckets, p_b_floored, seed
        )
        acc = save_correctness(section, config_id, "proposed", "fgsm", correct_fgsm)
        record(summary, section, config_id, "proposed", "fgsm", acc)
        results["fgsm"] = acc
        del fgsm_images, fgsm_labels

    if already_done(summary, section, config_id, "proposed", "pgd"):
        print(f"      [skip] proposed/pgd already done for {config_id}", flush=True)
        results["pgd"] = summary[section][config_id]["proposed"]["pgd"]
    else:
        pgd_images, pgd_labels = craft_adversarial(
            fwd, eot_loader, eot_pgd_attack, device, eps=eps, alpha=alpha, steps=steps, eot_samples=eot_samples
        )
        correct_pgd = per_image_correct_proposed(
            model, make_loader(pgd_images, pgd_labels, batch_size), device, R, buckets, p_b_floored, seed
        )
        acc = save_correctness(section, config_id, "proposed", "pgd", correct_pgd)
        record(summary, section, config_id, "proposed", "pgd", acc)
        results["pgd"] = acc
        del pgd_images, pgd_labels

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def run_deterministic_config(model, mean, std, device, images, labels, buckets, patterns, pi,
                              eps, alpha, steps, det_batch_size, need_clean, section, config_id, summary):
    results = {}
    det_loader = make_loader(images, labels, det_batch_size)

    if need_clean:
        if already_done(summary, section, config_id, "deterministic", "clean"):
            print(f"      [skip] deterministic/clean already done for {config_id}", flush=True)
            results["clean"] = summary[section][config_id]["deterministic"]["clean"]
        else:
            correct_clean = per_image_correct_deterministic(model, det_loader, device, buckets, patterns, pi)
            acc = save_correctness(section, config_id, "deterministic", "clean", correct_clean)
            record(summary, section, config_id, "deterministic", "clean", acc)
            results["clean"] = acc

    fwd = deterministic_attack_forward_fn(model, buckets, patterns, pi, mean, std)

    if already_done(summary, section, config_id, "deterministic", "fgsm"):
        print(f"      [skip] deterministic/fgsm already done for {config_id}", flush=True)
        results["fgsm"] = summary[section][config_id]["deterministic"]["fgsm"]
    else:
        fgsm_images, fgsm_labels = craft_adversarial(fwd, det_loader, fgsm_attack, device, eps=eps, loss_fn=F.nll_loss)
        correct_fgsm = per_image_correct_deterministic(
            model, make_loader(fgsm_images, fgsm_labels, det_batch_size), device, buckets, patterns, pi
        )
        acc = save_correctness(section, config_id, "deterministic", "fgsm", correct_fgsm)
        record(summary, section, config_id, "deterministic", "fgsm", acc)
        results["fgsm"] = acc
        del fgsm_images, fgsm_labels

    if already_done(summary, section, config_id, "deterministic", "pgd"):
        print(f"      [skip] deterministic/pgd already done for {config_id}", flush=True)
        results["pgd"] = summary[section][config_id]["deterministic"]["pgd"]
    else:
        pgd_images, pgd_labels = craft_adversarial(
            fwd, det_loader, pgd_attack, device, eps=eps, alpha=alpha, steps=steps, loss_fn=F.nll_loss
        )
        correct_pgd = per_image_correct_deterministic(
            model, make_loader(pgd_images, pgd_labels, det_batch_size), device, buckets, patterns, pi
        )
        acc = save_correctness(section, config_id, "deterministic", "pgd", correct_pgd)
        record(summary, section, config_id, "deterministic", "pgd", acc)
        results["pgd"] = acc
        del pgd_images, pgd_labels

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def calibrate_for_K(data_root, K, num_calib, device, seed):
    """Raw (unfloored) p_b only depends on K, not floor - calibrate once and
    let callers clamp with whatever floor they need."""
    buckets, p_b, _ = calibrate_group_probabilities(data_root, K, 0.05, num_calib, device, seed)
    sizes = [int((buckets == b).sum()) for b in range(K)]
    return buckets, p_b, sizes


def section_A(model, mean, std, device, images, labels, args, summary):
    print("\n" + "=" * 60, flush=True)
    print("=== SECTION A: floor sweep (K=5, eps=4/255) ===", flush=True)
    print("=" * 60, flush=True)
    K = 5
    buckets, p_b, sizes = calibrate_for_K(args.data_root, K, args.num_calib, device, args.seed)
    patterns = enumerate_combinations(K)
    det_batch_size = max(1, 256 // (2 ** K))

    floors = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.36]
    for floor in floors:
        config_id = f"floor{floor:.2f}"
        print(f"\n--- A: {config_id} (K={K}, eps=4/255) ---", flush=True)
        p_b_floored = torch.clamp(p_b, min=floor)
        pi = combination_probabilities(patterns, p_b_floored)

        t0 = time.time()
        try:
            print("  [proposed EOT]", flush=True)
            run_proposed_config(
                model, mean, std, device, images, labels, buckets, p_b_floored,
                args.eps, args.pgd_alpha, args.pgd_steps, args.eot_samples, args.R, args.seed,
                args.batch_size, args.eot_batch_size, True, "A", config_id, summary,
            )
        except Exception:
            print(f"  !! proposed config {config_id} FAILED:\n{traceback.format_exc()}", flush=True)

        try:
            print("  [deterministic]", flush=True)
            run_deterministic_config(
                model, mean, std, device, images, labels, buckets, patterns, pi,
                args.eps, args.pgd_alpha, args.pgd_steps, det_batch_size, True, "A", config_id, summary,
            )
        except Exception:
            print(f"  !! deterministic config {config_id} FAILED:\n{traceback.format_exc()}", flush=True)

        print(f"  ({time.time() - t0:.1f}s)", flush=True)

    print("\n=== SECTION A DONE ===", flush=True)


def section_B(model, mean, std, device, images, labels, args, summary):
    print("\n" + "=" * 60, flush=True)
    print("=== SECTION B: K sweep (eps=4/255, keep_ratio~=0.5) ===", flush=True)
    print("=" * 60, flush=True)

    for K in [3, 8, 16]:
        config_id = f"K{K}"
        print(f"\n--- B: {config_id} ---", flush=True)
        buckets, p_b, sizes = calibrate_for_K(args.data_root, K, args.num_calib, device, args.seed)
        floor = solve_floor_for_keep_ratio(p_b[0].item(), sizes[0])
        p_b_floored = torch.clamp(p_b, min=floor)
        print(f"  sizes={sizes}, p_b0={p_b[0].item():.4f}, solved floor={floor:.4f}", flush=True)

        t0 = time.time()
        try:
            print("  [proposed EOT]", flush=True)
            run_proposed_config(
                model, mean, std, device, images, labels, buckets, p_b_floored,
                args.eps, args.pgd_alpha, args.pgd_steps, args.eot_samples, args.R, args.seed,
                args.batch_size, args.eot_batch_size, True, "B", config_id, summary,
            )
        except Exception:
            print(f"  !! proposed config {config_id} FAILED:\n{traceback.format_exc()}", flush=True)

        if K == 16:
            print("  [deterministic] SKIPPED - 2^16=65536 patterns is computationally "
                  "infeasible for FGSM/PGD within a reasonable time budget (user decision).", flush=True)
        else:
            det_batch_size = max(1, 256 // (2 ** K))
            patterns = enumerate_combinations(K)
            pi = combination_probabilities(patterns, p_b_floored)
            try:
                print("  [deterministic]", flush=True)
                run_deterministic_config(
                    model, mean, std, device, images, labels, buckets, patterns, pi,
                    args.eps, args.pgd_alpha, args.pgd_steps, det_batch_size, True, "B", config_id, summary,
                )
            except Exception:
                print(f"  !! deterministic config {config_id} FAILED:\n{traceback.format_exc()}", flush=True)

        print(f"  ({time.time() - t0:.1f}s)", flush=True)

    print("\n=== SECTION B DONE ===", flush=True)


def section_C(model, mean, std, device, images, labels, args, summary):
    print("\n" + "=" * 60, flush=True)
    print("=== SECTION C: eot_samples sweep (floor=0.05, K=5, eps=4/255) ===", flush=True)
    print("=" * 60, flush=True)
    K = 5
    floor = 0.05
    buckets, p_b, sizes = calibrate_for_K(args.data_root, K, args.num_calib, device, args.seed)
    p_b_floored = torch.clamp(p_b, min=floor)

    for eot_samples in [5, 20, 40]:
        config_id = f"eot{eot_samples}"
        print(f"\n--- C: {config_id} ---", flush=True)
        t0 = time.time()
        try:
            run_proposed_config(
                model, mean, std, device, images, labels, buckets, p_b_floored,
                args.eps, args.pgd_alpha, args.pgd_steps, eot_samples, args.R, args.seed,
                args.batch_size, args.eot_batch_size, False, "C", config_id, summary,
            )
        except Exception:
            print(f"  !! config {config_id} FAILED:\n{traceback.format_exc()}", flush=True)
        print(f"  ({time.time() - t0:.1f}s)", flush=True)

    print("\n=== SECTION C DONE ===", flush=True)


def section_D(model, mean, std, device, images, labels, args, summary):
    print("\n" + "=" * 60, flush=True)
    print("=== SECTION D: eps cross-check (floor=0.05, K=5, N=10, eps=8/255) ===", flush=True)
    print("=" * 60, flush=True)
    K = 5
    floor = 0.05
    eps_8 = 8 / 255
    config_id = "floor0.05_eps8"

    buckets, p_b, sizes = calibrate_for_K(args.data_root, K, args.num_calib, device, args.seed)
    p_b_floored = torch.clamp(p_b, min=floor)
    patterns = enumerate_combinations(K)
    pi = combination_probabilities(patterns, p_b_floored)
    det_batch_size = max(1, 256 // (2 ** K))

    t0 = time.time()
    try:
        print("  [proposed EOT]", flush=True)
        run_proposed_config(
            model, mean, std, device, images, labels, buckets, p_b_floored,
            eps_8, args.pgd_alpha, args.pgd_steps, args.eot_samples, args.R, args.seed,
            args.batch_size, args.eot_batch_size, True, "D", config_id, summary,
        )
    except Exception:
        print(f"  !! proposed config {config_id} FAILED:\n{traceback.format_exc()}", flush=True)

    try:
        print("  [deterministic]", flush=True)
        run_deterministic_config(
            model, mean, std, device, images, labels, buckets, patterns, pi,
            eps_8, args.pgd_alpha, args.pgd_steps, det_batch_size, True, "D", config_id, summary,
        )
    except Exception:
        print(f"  !! deterministic config {config_id} FAILED:\n{traceback.format_exc()}", flush=True)

    print(f"  ({time.time() - t0:.1f}s)", flush=True)
    print("\n=== SECTION D DONE ===", flush=True)


def main():
    parser = argparse.ArgumentParser(description="10-step hyperparameter sweep (A/B/C/D).")
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eot-batch-size", type=int, default=25)
    parser.add_argument("--eot-samples", type=int, default=10)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sections", type=str, default="A,B,C,D",
                         help="Comma-separated subset of sections to run.")
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

    summary = load_summary()
    sections = args.sections.split(",")

    overall_start = time.time()
    if "A" in sections:
        section_A(model, mean, std, device, images, labels, args, summary)
    if "B" in sections:
        section_B(model, mean, std, device, images, labels, args, summary)
    if "C" in sections:
        section_C(model, mean, std, device, images, labels, args, summary)
    if "D" in sections:
        section_D(model, mean, std, device, images, labels, args, summary)

    print(f"\n=== SWEEP COMPLETE ({time.time() - overall_start:.1f}s total) ===", flush=True)
    print(f"Summary written to {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
