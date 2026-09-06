"""
Adversarial robustness evaluation: FGSM and PGD against the undefended
baseline, the original RSR defense, and the proposed quantized-group defense
(models/rsr.py, models/proposed.py).

RSR and the proposed defense reconstruct the input via SVD before
classification; that reconstruction involves random sampling and discrete
masks, so it isn't meaningfully differentiable. Attacks against them use BPDA
(attacks.py): the real (stochastic) reconstruction still runs on the forward
pass, but its backward pass is approximated as the identity, so a usable
gradient w.r.t. the raw pixel input exists for crafting the attack. The
undefended baseline is directly differentiable and needs no such trick.

After crafting adversarial images (with the BPDA surrogate used only for
gradients), robust accuracy is measured with each defense's REAL evaluation
pipeline (RSR/proposed: R=10 stochastic reconstructions + majority vote).

Runs on a random 1000-image subset of the CIFAR-10 test set (not the full
10,000) to keep the iterative PGD attack fast.
"""

import argparse
import os
import sys

import torch
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from vgg16_cifar10_baseline import VGG16Cifar
from rsr import CIFAR_MEAN, CIFAR_STD, svd_decompose, sample_reconstruction, evaluate_plain, evaluate_rsr
from proposed import calibrate_group_probabilities, sample_group_mask, reconstruct_with_group_mask, evaluate_proposed

from attacks import fgsm_attack, pgd_attack, bpda_apply


def get_random_test_subset(data_root: str, num_samples: int, seed: int):
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(test_set), generator=g)[:num_samples]
    images = torch.stack([test_set[i][0] for i in indices])
    labels = torch.tensor([test_set[i][1] for i in indices])
    return images, labels


def make_loader(images: torch.Tensor, labels: torch.Tensor, batch_size: int):
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def baseline_forward_fn(model, mean, std):
    def fn(x):
        return model((x - mean) / std)
    return fn


def rsr_forward_fn(model, mean, std, keep_ratio: float, generator: torch.Generator):
    """One stochastic RSR reconstruction per call, wrapped in BPDA so the
    attack gets a gradient even though the sampling itself isn't
    differentiable. Actual robustness is measured separately with the real
    R=10 majority-vote pipeline (evaluate_rsr)."""
    def fn(x):
        def recon_once(xx):
            U, S, Vh = svd_decompose(xx)
            return sample_reconstruction(U, S, Vh, keep_ratio, generator)
        recon = bpda_apply(x, recon_once)
        return model((recon - mean) / std)
    return fn


def proposed_forward_fn(model, mean, std, buckets, p_b_floored, generator: torch.Generator):
    """One stochastic proposed-defense reconstruction per call, wrapped in
    BPDA. Actual robustness is measured with evaluate_proposed's real R=10
    majority-vote pipeline."""
    def fn(x):
        def recon_once(xx):
            U, S, Vh = svd_decompose(xx)
            n, c = xx.size(0), xx.size(1)
            mask = sample_group_mask(p_b_floored.cpu(), n, c, generator).to(xx.device)
            return reconstruct_with_group_mask(U, S, Vh, buckets, mask)
        recon = bpda_apply(x, recon_once)
        return model((recon - mean) / std)
    return fn


def craft_adversarial(forward_fn, loader, attack, device, **attack_kwargs):
    adv_images, labels = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        x_adv = attack(forward_fn, images, targets, **attack_kwargs)
        adv_images.append(x_adv.detach().cpu())
        labels.append(targets.cpu())
        # Explicit cleanup after each batch: the GPU-resident images/targets/
        # x_adv aren't needed once copied to CPU, and PGD/EOT steps can leave
        # sizeable intermediate tensors (reconstructions, gradients) around.
        del images, targets, x_adv
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(adv_images), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FGSM/PGD robustness of the baseline, RSR, and the proposed defense."
    )
    parser.add_argument("--checkpoint", type=str, default="./results/vgg16_cifar10_baseline.pth")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--rsr-keep-ratio", type=float, default=0.5)
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
    clean_loader = make_loader(images, labels, args.batch_size)

    print(f"Calibrating proposed defense (K={args.K}, floor={args.floor}) on {args.num_calib} training images...")
    buckets, p_b, p_b_floored = calibrate_group_probabilities(
        args.data_root, args.K, args.floor, args.num_calib, device, args.seed
    )
    print(f"  p_b_prime = {[round(v, 4) for v in p_b_floored.tolist()]}")

    gen = torch.Generator().manual_seed(args.seed)

    forward_fns = {
        "No defense": baseline_forward_fn(model, mean, std),
        "Original RSR": rsr_forward_fn(model, mean, std, args.rsr_keep_ratio, gen),
        f"Proposed (K={args.K}, floor={args.floor})": proposed_forward_fn(
            model, mean, std, buckets, p_b_floored, gen
        ),
    }

    def eval_accuracy(name, loader):
        """Real (stochastic, majority-vote where applicable) evaluation - not
        the BPDA surrogate, which is only for gradient crafting."""
        if name == "No defense":
            return evaluate_plain(model, loader, device)
        elif name == "Original RSR":
            return evaluate_rsr(model, loader, device, args.R, args.rsr_keep_ratio, args.seed)
        else:
            return evaluate_proposed(model, loader, device, args.R, buckets, p_b_floored, args.seed)[0]

    results = {}
    for name, forward_fn in forward_fns.items():
        print(f"\n=== {name} ===")
        clean_acc = eval_accuracy(name, clean_loader)
        print(f"  Clean accuracy: {clean_acc:.2f}%")

        fgsm_images, fgsm_labels = craft_adversarial(
            forward_fn, clean_loader, fgsm_attack, device, eps=args.eps
        )
        fgsm_acc = eval_accuracy(name, make_loader(fgsm_images, fgsm_labels, args.batch_size))
        print(f"  FGSM (eps={args.eps:.4f}) robust accuracy: {fgsm_acc:.2f}%")

        pgd_images, pgd_labels = craft_adversarial(
            forward_fn, clean_loader, pgd_attack, device,
            eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps,
        )
        pgd_acc = eval_accuracy(name, make_loader(pgd_images, pgd_labels, args.batch_size))
        print(f"  PGD (eps={args.eps:.4f}, alpha={args.pgd_alpha:.4f}, steps={args.pgd_steps}) "
              f"robust accuracy: {pgd_acc:.2f}%")

        results[name] = (clean_acc, fgsm_acc, pgd_acc)

    print(f"\n=== Summary (num_samples={args.num_samples}, eps={args.eps:.4f}) ===")
    print(f"{'Method':<35}{'Clean':>10}{'FGSM':>10}{'PGD':>10}")
    for name, (c, f, p) in results.items():
        print(f"{name:<35}{c:>9.2f}%{f:>9.2f}%{p:>9.2f}%")


if __name__ == "__main__":
    main()
