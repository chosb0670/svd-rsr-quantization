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


def _empty_cache_if_cuda(x: torch.Tensor):
    if x.is_cuda:
        torch.cuda.empty_cache()


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
    x_adv = x_adv.clamp(0.0, 1.0)
    del x, output, loss, grad
    return x_adv


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
        del output, loss, grad
    _empty_cache_if_cuda(x_orig)
    return x_adv


def eot_grad(forward_fn, x: torch.Tensor, y: torch.Tensor, eot_samples: int,
             loss_fn=F.cross_entropy) -> torch.Tensor:
    """Expectation-over-Transformation gradient estimate: averages the
    gradient over `eot_samples` independent stochastic evaluations of
    forward_fn at the SAME x, instead of the single sample plain BPDA uses.
    This is the correct gradient of the defense's *expected* loss under its
    own randomness (Athalye et al.), and is what an adaptive attacker against
    a randomized defense should actually use.

    Vectorized: x is tiled into an (eot_samples * n) batch in one shot rather
    than looped, so forward_fn's internal random sampling draws a fresh,
    independent reconstruction per tile; a single backward() call then gives
    the properly averaged per-image gradient (autograd sums gradient
    contributions across the tiled/expanded copies back into the original
    x, and cross_entropy's batch-mean averaging divides by eot_samples)."""
    n = x.size(0)
    x_req = x.clone().detach().requires_grad_(True)
    x_tiled = x_req.unsqueeze(0).expand(eot_samples, *x_req.shape).reshape(
        eot_samples * n, *x_req.shape[1:]
    )
    y_tiled = y.unsqueeze(0).expand(eot_samples, n).reshape(eot_samples * n)

    output = forward_fn(x_tiled)
    loss = loss_fn(output, y_tiled)
    grad = torch.autograd.grad(loss, x_req)[0]

    del x_tiled, y_tiled, output, loss
    return grad


def eot_fgsm_attack(forward_fn, x: torch.Tensor, y: torch.Tensor, eps: float,
                     eot_samples: int = 10, loss_fn=F.cross_entropy) -> torch.Tensor:
    x = x.clone().detach()
    grad = eot_grad(forward_fn, x, y, eot_samples, loss_fn)
    x_adv = (x + eps * grad.sign()).clamp(0.0, 1.0)
    del grad
    _empty_cache_if_cuda(x)
    return x_adv


def eot_pgd_attack(forward_fn, x: torch.Tensor, y: torch.Tensor, eps: float,
                    alpha: float, steps: int, eot_samples: int = 10,
                    loss_fn=F.cross_entropy) -> torch.Tensor:
    """PGD where each step's gradient is an EOT average over `eot_samples`
    stochastic reconstructions, rather than plain BPDA's single sample."""
    x_orig = x.clone().detach()
    x_adv = x_orig.clone()
    for _ in range(steps):
        grad = eot_grad(forward_fn, x_adv, y, eot_samples, loss_fn)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0.0, 1.0)
        del grad
        _empty_cache_if_cuda(x_orig)
    return x_adv
