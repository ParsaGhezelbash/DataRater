# attack.py
import torch
import torch.nn as nn
from torch.autograd import grad

__all__ = ["fgsm_attack", "pgd_attack", "apgd_attack"]


from torch.func import functional_call
# Differentiable inner loop (manual SGD, no optim.step) ---
def call_with_fast(model, fast_params, x):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    # replace only params with fast ones; keep real buffers
    merged = {**buffers, **params}
    merged.update(fast_params)
    return functional_call(model, merged, (x,), strict=False)


# FGSM: Fast Gradient Sign Method
def fgsm_attack(model, inputs, labels, loss_fn, eps, params=None):
    r"""
    Perform single-step FGSM attack.
    Args:
        model: forward-callable model or function (x -> logits)
        inputs (Tensor): clean input batch (requires_grad=False)
        labels (Tensor): true targets
        loss_fn: loss function (e.g., nn.CrossEntropyLoss())
        eps (float): perturbation strength in same normalization scale
        params (dict): optional fast params (for stateless functional_call)
    """
    x = inputs.clone().detach()
    
    x = x.requires_grad_(True)
    logits = model(x) if params is None else call_with_fast(model, params, x)
    loss = loss_fn(logits, labels)
    g = torch.autograd.grad(loss, x, only_inputs=True)[0]
    x_adv = x + eps * g.sign()
    return torch.clamp(x_adv, 0.0, 1.0).detach()


# PGD: Projected Gradient Descent
def pgd_attack(model, inputs, labels, loss_fn, eps, step_size, steps, 
               random_start=True, params=None):
    r"""
    Multi-step adversarial attack (L∞ PGD).
    """
    x = inputs.clone().detach()

    if random_start:
        x = x + torch.empty_like(x).uniform_(-eps, eps)
        x = torch.clamp(x, 0.0, 1.0)

    for _ in range(steps):
        x.requires_grad_(True)
        logits = model(x) if params is None else call_with_fast(model, params, x)
        loss = loss_fn(logits, labels)
        g = torch.autograd.grad(loss, x, only_inputs=True)[0]

        x = x + step_size * g.sign()
        delta = torch.clamp(x - inputs, -eps, eps)
        x = torch.clamp(inputs + delta, 0.0, 1.0).detach()

    return x


# APGD: Auto-Projected Gradient Descent (simplified)
def apgd_attack(model, inputs, labels, loss_fn,
                eps=8/255, steps=10, beta=1.5, params=None):
    r"""
    Simplified APGD (Auto-PGD) — adaptive step control version of PGD.
    Source: Croce & Hein (2020)
    """
    x = inputs.clone().detach()
    x = x + torch.empty_like(x).uniform_(-eps, eps)
    x = torch.clamp(x, 0.0, 1.0)

    step_size = 2 * eps / steps
    best_adv, best_loss = x, torch.full((x.size(0),), -1e9, device=x.device)

    # running momentum-style step sizes
    for t in range(steps):
        x.requires_grad_(True)
        logits = model(x) if params is None else call_with_fast(model, params, x)
        loss = loss_fn(logits, labels)
        g = torch.autograd.grad(loss, x, only_inputs=True)[0]

        x = x + step_size * g.sign()
        delta = torch.clamp(x - inputs, -eps, eps)
        x = torch.clamp(inputs + delta, 0, 1).detach()

        # track best adversarial examples
        with torch.no_grad():
            new_logits = model(x) if params is None else call_with_fast(x, params)
            new_loss = loss_fn(new_logits, labels).detach()
            replace = new_loss > best_loss
            best_adv[replace] = x[replace]
            best_loss[replace] = new_loss[replace]

        # adapt step size if loss stagnates (simple heuristic)
        if (t + 1) % (steps // 3) == 0:
            step_size *= beta

    return best_adv.detach()
