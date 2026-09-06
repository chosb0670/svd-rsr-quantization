"""
Compares the proposed defense's random R=10 Monte Carlo sampling
(models/proposed.py) against its deterministic exact-expectation control
(models/deterministic.py) under FGSM and PGD, on the same random 1000-image
CIFAR-10 test subset used in attacks/evaluate.py (seed=42).

The random version needs BPDA to attack (its reconstruction is a
non-differentiable random draw). The deterministic version enumerates all
2^K group-inclusion patterns and averages their softmax outputs by exact
probability - a finite weighted sum of differentiable branches, so it is
fully differentiable end-to-end and is attacked with EXACT gradients, no
BPDA approximation needed.

Goal: check whether removing the defense's randomness (a) raises clean
accuracy and (b) lowers adversarial robustness, since an adversary can now
compute exact gradients instead of the noisy BPDA surrogate.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from vgg16_cifar10_baseline import VGG16Cifar
from rsr import CIFAR_MEAN, CIFAR_STD
from proposed import calibrate_group_probabilities, evaluate_proposed
from deterministic import enumerate_combinations, combination_probabilities, deterministic_forward, evaluate_deterministic

from attacks import fgsm_attack, pgd_attack
from evaluate import get_random_test_subset, make_loader, craft_adversarial, proposed_forward_fn


def deterministic_attack_forward_fn(model, buckets, patterns, pi, mean, std):
    """Returns log-probabilities (not raw logits) since deterministic_forward
    already outputs a normalized probability distribution - use with
    loss_fn=F.nll_loss when crafting attacks."""
    def fn(x):
        probs = deterministic_forward(model, x, buckets, patterns, pi, mean, std)
        return torch.log(probs.clamp_min(1e-12))
    return fn


def main():
    parser = argparse.ArgumentParser(
        description="Compare proposed (random R=10) vs deterministic (exact expectation) under FGSM/PGD."
    )
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100,
                         help="Batch size for the random (proposed) method.")
    parser.add_argument("--det-batch-size", type=int, default=32,
                         help="Batch size for the deterministic method (effective batch = this * 2^K).")
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--floor", type=float, default=0.36)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VGG16Cifar(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)

    print(f"Sampling {args.num_samples} random test images (seed={args.seed})...")
    images, labels = get_random_test_subset(args.data_root, args.num_samples, args.seed)

    print(f"Calibrating (K={args.K}, floor={args.floor}) on {args.num_calib} training images...")
    buckets, p_b, p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, args.floor, args.num_calib, device, args.seed
    )
    patterns = enumerate_combinations(args.K)
    pi = combination_probabilities(patterns, p_b_floored)
    print(f"  p_b_prime = {[round(v, 4) for v in p_b_floored.tolist()]}, "
          f"{patterns.size(0)} patterns, pi sums to {pi.sum().item():.6f}")

    gen = torch.Generator().manual_seed(args.seed)

    # --- Proposed (random, R=10): needs BPDA, evaluated at batch_size ---
    proposed_loader = make_loader(images, labels, args.batch_size)
    proposed_fwd = proposed_forward_fn(model, mean, std, buckets, p_b_floored, gen)

    def eval_proposed(loader):
        return evaluate_proposed(model, loader, device, args.R, buckets, p_b_floored, args.seed)[0]

    # --- Deterministic (exact expectation): fully differentiable, exact grad, smaller batch ---
    det_loader = make_loader(images, labels, args.det_batch_size)
    det_fwd = deterministic_attack_forward_fn(model, buckets, patterns, pi, mean, std)

    def eval_deterministic(loader):
        return evaluate_deterministic(model, loader, device, buckets, patterns, pi)

    results = {}

    print("\n=== Proposed (random sampling, R=10, BPDA attack) ===")
    clean_acc = eval_proposed(proposed_loader)
    print(f"  Clean accuracy: {clean_acc:.2f}%")
    fgsm_images, fgsm_labels = craft_adversarial(proposed_fwd, proposed_loader, fgsm_attack, device, eps=args.eps)
    fgsm_acc = eval_proposed(make_loader(fgsm_images, fgsm_labels, args.batch_size))
    print(f"  FGSM (eps={args.eps:.4f}) robust accuracy: {fgsm_acc:.2f}%")
    pgd_images, pgd_labels = craft_adversarial(
        proposed_fwd, proposed_loader, pgd_attack, device,
        eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps,
    )
    pgd_acc = eval_proposed(make_loader(pgd_images, pgd_labels, args.batch_size))
    print(f"  PGD (T={args.pgd_steps}, BPDA) robust accuracy: {pgd_acc:.2f}%")
    results["Proposed (random, R=10, BPDA)"] = (clean_acc, fgsm_acc, pgd_acc)

    print("\n=== Deterministic (exact expectation over 2^K patterns, exact gradient) ===")
    clean_acc_d = eval_deterministic(det_loader)
    print(f"  Clean accuracy: {clean_acc_d:.2f}%")
    fgsm_images_d, fgsm_labels_d = craft_adversarial(
        det_fwd, det_loader, fgsm_attack, device, eps=args.eps, loss_fn=F.nll_loss
    )
    fgsm_acc_d = eval_deterministic(make_loader(fgsm_images_d, fgsm_labels_d, args.det_batch_size))
    print(f"  FGSM (eps={args.eps:.4f}) robust accuracy: {fgsm_acc_d:.2f}%")
    pgd_images_d, pgd_labels_d = craft_adversarial(
        det_fwd, det_loader, pgd_attack, device,
        eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps, loss_fn=F.nll_loss,
    )
    pgd_acc_d = eval_deterministic(make_loader(pgd_images_d, pgd_labels_d, args.det_batch_size))
    print(f"  PGD (T={args.pgd_steps}, exact gradient) robust accuracy: {pgd_acc_d:.2f}%")
    results["Deterministic (exact expectation)"] = (clean_acc_d, fgsm_acc_d, pgd_acc_d)

    print(f"\n=== Summary (num_samples={args.num_samples}, eps={args.eps:.4f}) ===")
    print(f"{'Method':<38}{'Clean':>10}{'FGSM':>10}{'PGD':>10}")
    for name, (c, f, p) in results.items():
        print(f"{name:<38}{c:>9.2f}%{f:>9.2f}%{p:>9.2f}%")


if __name__ == "__main__":
    main()
