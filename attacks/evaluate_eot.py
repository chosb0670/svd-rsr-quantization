"""
Proper EOT (Expectation over Transformation) attack against the proposed
(random, R=10) defense, compared against the deterministic exact-expectation
control.

Plain BPDA (attacks/evaluate.py, attacks/evaluate_deterministic.py) computes
each PGD step's gradient from a SINGLE stochastic reconstruction sample -
a valid but weaker/noisier estimate of the gradient the attacker actually
wants: the gradient of the defense's *expected* loss under its own
randomness. EOT (Athalye et al.) averages the gradient over `--eot-samples`
independent stochastic reconstructions per step, which is the methodologically
correct way to attack a randomized defense and should find (at least as)
strong adversarial examples as plain single-sample BPDA.

The deterministic control has no randomness to average over - EOT is
mathematically identical to the exact gradient it already uses, so its
numbers are the ones already computed in attacks/evaluate_deterministic.py
(at the same eps) rather than recomputed here.

Memory/perf notes: DataLoaders here use num_workers=0 (no worker processes
to leak), all accuracy/majority-vote evaluation runs under @torch.no_grad()
(models/proposed.py), and both the EOT gradient helper and the attack loops
explicitly `del` intermediate tensors and call torch.cuda.empty_cache() after
each step/batch. EOT tiles `eot_samples` reconstructions into one batched
forward+backward pass per step (not a Python loop), so batch_size is kept
small (default 25) to bound the effective batch (batch_size * eot_samples).
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from vgg16_cifar10_baseline import VGG16Cifar
from rsr import CIFAR_MEAN, CIFAR_STD
from proposed import calibrate_group_probabilities, evaluate_proposed

from attacks import eot_fgsm_attack, eot_pgd_attack
from evaluate import get_random_test_subset, make_loader, proposed_forward_fn


def craft_adversarial_verbose(forward_fn, loader, attack, device, label, **attack_kwargs):
    """Same as evaluate.craft_adversarial but prints per-batch progress
    (flushed immediately) so a background run can be checked on mid-flight."""
    adv_images, labels = [], []
    num_batches = len(loader)
    start = time.time()
    for i, (images, targets) in enumerate(loader, 1):
        images, targets = images.to(device), targets.to(device)
        x_adv = attack(forward_fn, images, targets, **attack_kwargs)
        adv_images.append(x_adv.detach().cpu())
        labels.append(targets.cpu())
        del images, targets, x_adv
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{label}] batch {i}/{num_batches} done ({time.time() - start:.1f}s elapsed)", flush=True)
    return torch.cat(adv_images), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(
        description="EOT-attack the proposed (random) defense; compare against the deterministic control."
    )
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=25,
                         help="Kept small since EOT tiles eot_samples reconstructions per step "
                              "(effective batch = batch_size * eot_samples).")
    parser.add_argument("--eot-samples", type=int, default=10)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--floor", type=float, default=0.36)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    # Deterministic control's numbers at eps=8/255, from attacks/evaluate_deterministic.py -
    # not recomputed here since EOT is identical to its already-exact gradient.
    parser.add_argument("--det-clean-acc", type=float, default=87.90)
    parser.add_argument("--det-fgsm-acc", type=float, default=18.70)
    parser.add_argument("--det-pgd-acc", type=float, default=0.20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU memory before start: "
              f"{torch.cuda.memory_allocated() / 1e6:.1f}MB allocated", flush=True)

    model = VGG16Cifar(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)

    print(f"Sampling {args.num_samples} random test images (seed={args.seed})...", flush=True)
    images, labels = get_random_test_subset(args.data_root, args.num_samples, args.seed)
    loader = make_loader(images, labels, args.batch_size)

    print(f"Calibrating proposed defense (K={args.K}, floor={args.floor}) "
          f"on {args.num_calib} training images...", flush=True)
    buckets, p_b, p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, args.floor, args.num_calib, device, args.seed
    )
    print(f"  p_b_prime = {[round(v, 4) for v in p_b_floored.tolist()]}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    fwd = proposed_forward_fn(model, mean, std, buckets, p_b_floored, gen)

    def eval_proposed(loader):
        return evaluate_proposed(model, loader, device, args.R, buckets, p_b_floored, args.seed)[0]

    print("\n=== Proposed (random, R=10) under EOT ===", flush=True)
    clean_acc = eval_proposed(loader)
    print(f"  Clean accuracy: {clean_acc:.2f}%", flush=True)

    fgsm_images, fgsm_labels = craft_adversarial_verbose(
        fwd, loader, eot_fgsm_attack, device, "FGSM-EOT", eps=args.eps, eot_samples=args.eot_samples
    )
    fgsm_acc = eval_proposed(make_loader(fgsm_images, fgsm_labels, args.batch_size))
    print(f"  FGSM-EOT (eps={args.eps:.4f}, N={args.eot_samples}) robust accuracy: {fgsm_acc:.2f}%", flush=True)
    del fgsm_images, fgsm_labels
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pgd_images, pgd_labels = craft_adversarial_verbose(
        fwd, loader, eot_pgd_attack, device, "PGD-EOT",
        eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps, eot_samples=args.eot_samples,
    )
    pgd_acc = eval_proposed(make_loader(pgd_images, pgd_labels, args.batch_size))
    print(f"  PGD-EOT (eps={args.eps:.4f}, alpha={args.pgd_alpha:.4f}, steps={args.pgd_steps}, "
          f"N={args.eot_samples}) robust accuracy: {pgd_acc:.2f}%", flush=True)
    del pgd_images, pgd_labels
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n=== Summary (num_samples={args.num_samples}, eps={args.eps:.4f}, eot_samples={args.eot_samples}) ===")
    print(f"{'Method':<42}{'Clean':>10}{'FGSM':>10}{'PGD':>10}")
    print(f"{'Proposed (random, R=10, EOT N=' + str(args.eot_samples) + ')':<42}"
          f"{clean_acc:>9.2f}%{fgsm_acc:>9.2f}%{pgd_acc:>9.2f}%")
    print(f"{'Deterministic (exact, no EOT needed)':<42}"
          f"{args.det_clean_acc:>9.2f}%{args.det_fgsm_acc:>9.2f}%{args.det_pgd_acc:>9.2f}%")

    if device.type == "cuda":
        print(f"\nGPU memory at end: {torch.cuda.memory_allocated() / 1e6:.1f}MB allocated, "
              f"{torch.cuda.max_memory_allocated() / 1e6:.1f}MB peak")


if __name__ == "__main__":
    main()
