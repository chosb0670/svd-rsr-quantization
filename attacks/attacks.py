"""
FGSM and PGD adversarial attacks, with BPDA (Backward Pass Differentiable
Approximation) support for defenses whose input-reconstruction step is not
meaningfully differentiable (RSR, the proposed quantized-group defense both
involve random sampling / discrete masks before classification).

BPDA (Athalye et al., "Obfuscated Gradients Give a False Sense of Security"):
the true reconstruction still runs on the forward pass (so what the attacker
"sees" matches what the defense actually does), but its backward pass is
approximated as the identity function, so gradients can flow straight through
to the raw pixel input for attack crafting even though the real reconstruction
isn't differentiable.
"""

import torch
import torch.nn.functional as F


class BPDAIdentity(torch.autograd.Function):
    """Wraps a non-differentiable (or discretely-random) function `fn` so
    autograd treats it as the identity on the backward pass, while the
    forward pass still runs the real `fn`."""

    @staticmethod
    def forward(ctx, x, fn):
        with torch.no_grad():
            return fn(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def bpda_apply(x: torch.Tensor, fn) -> torch.Tensor:
    return BPDAIdentity.apply(x, fn)


def fgsm_attack(forward_fn, x: torch.Tensor, y: torch.Tensor, eps: float,
                 loss_fn=F.cross_entropy) -> torch.Tensor:
    """x' = x + eps * sign(grad_x L(x, y)). x: raw pixel images in [0, 1].
    forward_fn(x) -> whatever `loss_fn` expects (raw logits by default; may
    internally use BPDA for non-differentiable preprocessing). Pass
    loss_fn=F.nll_loss with a forward_fn returning log-probabilities for a
    model whose output is already a normalized distribution (e.g. a
    probability-weighted mixture) rather than raw logits."""
    x = x.clone().detach().requires_grad_(True)
    output = forward_fn(x)
    loss = loss_fn(output, y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x.detach() + eps * grad.sign()
    return x_adv.clamp(0.0, 1.0)


def pgd_attack(forward_fn, x: torch.Tensor, y: torch.Tensor, eps: float,
               alpha: float, steps: int, loss_fn=F.cross_entropy) -> torch.Tensor:
    """Iterative FGSM with per-step projection back into the eps-ball around
    the original image (L-infinity) and into the valid [0, 1] pixel range."""
    x_orig = x.clone().detach()
    x_adv = x_orig.clone()
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        output = forward_fn(x_adv)
        loss = loss_fn(output, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv
