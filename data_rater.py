import os
import csv
import uuid
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy import stats

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

import matplotlib

from attack import pgd_attack
matplotlib.use("Agg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt

from models import construct_model
from config import DataRaterConfig
from datasets import get_dataset_loaders

@dataclass
class LoggingContext:
    run_id: str
    outer_loss: list[float]
    outer_loss_clean: list[float]

from torch.func import functional_call

# Differentiable inner loop (manual SGD, no optim.step) ---
def call_with_fast(model, fast_params, x):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    # replace only params with fast ones; keep real buffers
    merged = {**buffers, **params}
    merged.update(fast_params)
    return functional_call(model, merged, (x,), strict=False)

def inner_unroll_differentiable(
    config: DataRaterConfig,
    inner_model: nn.Module,
    data_rater: nn.Module,
    train_iterator,
    train_loader,
    T: int,
):
    """
    Returns the 'fast' params after T differentiable SGD steps
    starting from the current inner_model params.
    """
    inner_model.train()
    data_rater.train()

    # Start from the model's current (base) parameters
    fast_params = {name: p.clone().detach().requires_grad_(True)
                   for name, p in inner_model.named_parameters()}

    tau = getattr(config, "rater_softmax_temperature", 1.0)  # optional config; default=1.0

    for _ in range(T):
        # Get next batch (wrap iterator if needed)
        try:
            inner_samples, inner_labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            inner_samples, inner_labels = next(train_iterator)

        inner_samples = inner_samples.to(config.device)
        inner_labels = inner_labels.to(config.device)

        # Ratings *with* grad, softmax across batch (dim=0). Optional temperature.
        ratings = data_rater(inner_samples)                         # [B] or [B,1]
        tau = 2.0 # temperature
        ratings = ratings - ratings.mean(dim=0)     # shift-invariant but stabilizes scale a bit
        weights = torch.softmax(ratings / tau, dim=0)
        # Forward with fast params
        logits = call_with_fast(inner_model, fast_params, inner_samples)

        # Per-sample loss
        if config.loss_type == 'mse':
            per = nn.functional.mse_loss(logits, inner_labels, reduction='none')  # [B, ...]
            if per.ndim > 1:
                per = per.view(per.size(0), -1).mean(dim=1)
        elif config.loss_type == 'cross_entropy':
            per = nn.functional.cross_entropy(logits, inner_labels, reduction='none')  # [B]
        else:
            raise ValueError(f"Loss type {config.loss_type} not supported")

        # Weight across the batch; use sum (weights already sum to 1)
        inner_loss = (per * weights).sum()

        # Compute grads wrt fast params (keep graph to backprop into η via weights)
        grads = torch.autograd.grad(inner_loss, list(fast_params.values()),
                                    create_graph=True, retain_graph=True)

        # Optional grad clipping (by value to keep it simple here)
        max_norm = getattr(config, "grad_clip_norm", None)
        if max_norm is not None and max_norm > 0:
            # simple per-tensor clamp; for true norm clip you'd need to compute norms
            clipped_grads = []
            for g in grads:
                clipped_grads.append(torch.clamp(g, -max_norm, max_norm))
            grads = clipped_grads

        # Manual SGD update
        lr = config.inner_lr
        fast_params = {
            name: p - lr * g
            for (name, p), g in zip(fast_params.items(), grads)
        }

    return fast_params, train_iterator


# --- Outer step using differentiable inner unroll ---
def outer_loop_step(config: DataRaterConfig,
                    logging_context: LoggingContext,
                    data_rater,
                    outer_optimizer,
                    inner_models,
                    train_iterator,
                    val_iterator,
                    train_loader,
                    val_loader):
    """
    Performs one meta-update step using a POPULATION of inner models with
    differentiable inner updates.
    """
    data_rater.train()
    for m in inner_models:
        m.train()  # we will use training stats/buffers during inner unroll

    # Get one outer batch shared by all models
    try:
        outer_samples, outer_labels = next(val_iterator)
    except StopIteration:
        val_iterator = iter(val_loader)
        outer_samples, outer_labels = next(val_iterator)

    outer_samples = outer_samples.to(config.device)
    outer_labels = outer_labels.to(config.device)

    # Run inner unroll for each model -> fast params, then eval outer loss
    outer_losses = []
    outer_losses_clean = []
    for model in inner_models:
        fast_params, train_iterator = inner_unroll_differentiable(
            config, model, data_rater, train_iterator, train_loader, config.inner_steps
        )

        # Attack on outer samples with inner model
        model.eval()
        adv_outer_samples = pgd_attack(
            model, outer_samples, outer_labels,
            loss_fn=nn.CrossEntropyLoss(),
            eps=config.attack_eps,
            step_size=config.attack_step_size,
            steps=config.attack_steps,
            random_start=True,
            params=fast_params
        )

        # Evaluate on outer batch using the fast params
        model.eval()
        outer_logits = call_with_fast(model, fast_params, adv_outer_samples)

        if config.loss_type == 'mse':
            outer_loss = nn.functional.mse_loss(outer_logits, outer_labels)
        elif config.loss_type == 'cross_entropy':
            outer_loss = nn.functional.cross_entropy(outer_logits, outer_labels)
        else:
            raise ValueError(f"Loss type {config.loss_type} not supported")

        outer_losses.append(outer_loss)

        outer_logits_clean = call_with_fast(model, fast_params, outer_samples)
        if config.loss_type == 'mse':
            outer_loss_clean = nn.functional.mse_loss(outer_logits_clean, outer_labels)
        elif config.loss_type == 'cross_entropy':
            outer_loss_clean = nn.functional.cross_entropy(outer_logits_clean, outer_labels)
        else:
            raise ValueError(f"Loss type {config.loss_type} not supported")

        outer_losses_clean.append(outer_loss_clean)

        # Inner model update
        if config.model_update:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(fast_params[name])

    average_outer_loss = torch.mean(torch.stack(outer_losses))
    average_outer_loss_clean = torch.mean(torch.stack(outer_losses_clean))

    # Meta-step on η
    outer_optimizer.zero_grad()
    for i, outer_loss in enumerate(outer_losses):
        (outer_loss / len(outer_losses)).backward()  # average grads
    torch.nn.utils.clip_grad_norm_(data_rater.parameters(), config.grad_clip_norm)
    outer_optimizer.step()

    logging_context.outer_loss.append(average_outer_loss.item())
    logging_context.outer_loss_clean.append(average_outer_loss_clean.item())
    return average_outer_loss.item(), average_outer_loss_clean.item(), train_iterator, val_iterator


def save_regression_plot(corruption_levels, weights, out_dir, tag, x_label, slope=None, intercept=None, r_value=None):
    """
    Saves a scatter + linear fit plot to <out_dir>/plots/{tag}.png.
    """
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Scatter
    plt.scatter(corruption_levels, weights, alpha=0.3, label="Individual Image Score")
    
    if slope is not None:
        r_squared = r_value ** 2
        # Fit line
        x_line = np.array([min(corruption_levels), max(corruption_levels)])
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, linewidth=2, color="red",
                label=f"Linear Regression (R² = {r_squared:.3f})")

    plt.title("DataRater Score vs. Individual Image", fontsize=16)
    plt.xlabel(x_label)
    plt.ylabel("Raw Score (Rating)")
    plt.grid(True, linestyle=":")
    plt.legend()

    out_path = os.path.join(out_dir, "plots", f"{tag}.png")  # <-- unique filename via tag
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def compute_regression_coefficient(config: DataRaterConfig, dataset_handler, data_rater, test_loader):
    """
    Computes the regression coefficient of the data rater and 
    corruption level of a given sample.
    """

    data_rater.eval()

    weights = []
    corruption_levels = []

    for i, (batch_samples, batch_labels) in enumerate(test_loader):
        # samples in the batch
        batch_samples, batch_labels = batch_samples.to(config.device), batch_labels.to(config.device)
        batch_size = batch_samples.size(0)
        individually_corrupted_batch = torch.zeros_like(batch_samples)
        fractions_for_this_batch = []
        
        for j in range(batch_size):
            # determine the corruption level for this sample
            frac = np.random.uniform(0.0, 1.0)

            # append the corruption level for this sample
            fractions_for_this_batch.append(frac)

            # corrupt the sample
            original_sample = batch_samples[j:j+1]
            corrupted_sample = dataset_handler.corrupt_samples(original_sample, frac)
            individually_corrupted_batch[j] = corrupted_sample
            
        with torch.no_grad():
            scores = data_rater(individually_corrupted_batch).squeeze(-1)  # [B]
            # raw, uncoupled:
            weights.extend(scores.cpu().numpy())
            corruption_levels.extend(fractions_for_this_batch)

    slope, intercept, r_value, p_value, std_err = stats.linregress(corruption_levels, weights)
    return slope, intercept, r_value, p_value, std_err, corruption_levels, weights


def compute_rate(config: DataRaterConfig, model, data_rater, test_loader):
    """
    Computes the regression coefficient of the data rater and loss on clean samples.
    """

    data_rater.eval()

    weights = []
    loss_values = []
    accuracy_values = []

    for i, (batch_samples, batch_labels) in enumerate(test_loader):
        # samples in the batch
        batch_samples, batch_labels = batch_samples.to(config.device), batch_labels.to(config.device)

        with torch.no_grad():
            scores = data_rater(batch_samples).squeeze(-1)
            weights.extend(scores.cpu().numpy())

            loss_fn = nn.CrossEntropyLoss() if config.loss_type == 'cross_entropy' else nn.MSELoss()
            loss_values.extend(loss_fn(model(batch_samples), batch_labels).cpu().numpy().flatten())
            
            preds = model(batch_samples).argmax(dim=-1)
            accuracy_values.extend((preds == batch_labels).cpu().numpy().flatten())

    # slope, intercept, r_value, p_value, std_err = stats.linregress(loss_values, weights)
    return loss_values, weights, accuracy_values


def compute_rate_adv(config: DataRaterConfig, model, data_rater, test_loader):
    """
    Computes the regression coefficient of the data rater and loss on clean samples.
    """

    data_rater.eval()

    weights = []
    loss_values = []
    accuracy_values = []

    for i, (batch_samples, batch_labels) in enumerate(test_loader):
        # samples in the batch
        batch_samples, batch_labels = batch_samples.to(config.device), batch_labels.to(config.device)

        with torch.no_grad():
            scores = data_rater(batch_samples).squeeze(-1)
            weights.extend(scores.cpu().numpy())

            adv_samples = pgd_attack(
                model, batch_samples, batch_labels,
                loss_fn=nn.CrossEntropyLoss(),
                eps=config.attack_eps,
                step_size=config.attack_step_size,
                steps=config.attack_steps,
                random_start=True
            )
            loss_fn = nn.CrossEntropyLoss() if config.loss_type == 'cross_entropy' else nn.MSELoss()
            loss_values.extend(loss_fn(model(adv_samples), batch_labels).cpu().numpy().flatten())

            preds = model(adv_samples).argmax(dim=-1)
            accuracy_values.extend((preds == batch_labels).cpu().numpy().flatten())   

    # slope, intercept, r_value, p_value, std_err = stats.linregress(loss_values, weights)
    return loss_values, weights, accuracy_values


def run_meta_training(config: DataRaterConfig):
    """
    Initializes models and orchestrates the main meta-training loop.
    Manages a population of inner models and refreshes them periodically.
    """
    # Generate run_id with dataset name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"{config.dataset_name}_{timestamp}_{str(uuid.uuid4())[:8]}"

    print(f"Run ID: {run_id}")
    logging_context = LoggingContext(run_id=run_id, outer_loss=[], outer_loss_clean=[])

    run_dir = os.path.join("experiments", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # --- Initial Setup ---
    data_rater = construct_model(config.data_rater_model_class).to(config.device)
    outer_optimizer = optim.Adam(data_rater.parameters(), lr=config.outer_lr)

    dataset_handler, (train_loader, val_loader, test_loader) = get_dataset_loaders(config)
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)

    inner_models = None

    print(
        f"Starting meta-training with a population of {config.num_inner_models} models, "
        f"refreshing every {config.meta_refresh_steps} steps."
    )

    for meta_step in tqdm(range(config.meta_steps), desc="Meta-Training"):
        if meta_step % config.meta_refresh_steps == 0:
            tqdm.write(f"\n[Meta-Step {meta_step}] Refreshing inner model population...")
            inner_models = [
                construct_model(config.inner_model_class).to(config.device)
                for _ in range(config.num_inner_models)
            ]

        outer_loss, outer_loss_clean, train_iterator, val_iterator = outer_loop_step(
            config,
            logging_context,
            data_rater, outer_optimizer,
            inner_models,
            train_iterator, val_iterator,
            train_loader, val_loader
        )

        # total = 0.0
        # for n,p in data_rater.named_parameters():
        #     if p.grad is not None:
        #         g = p.grad.detach()
        #         total += g.abs().sum().item()
        # print(f"[η grad sum] {total:.3e}")

        if (meta_step + 1) % 10 == 0:
            tqdm.write(f"  [Meta-Step {meta_step + 1}/{config.meta_steps}] Outer Loss (Adv): {outer_loss:.4f}, Outer Loss (Clean): {outer_loss_clean:.4f}")

        if (meta_step) % 100 == 0:
            slope, intercept, r_value, p_value, std_err, corruption_levels, weights = compute_regression_coefficient(
                config, dataset_handler, data_rater, test_loader
            )
            tag = f"regression_step_{meta_step + 1:06d}"
            save_regression_plot(corruption_levels, weights, run_dir, tag, "corruption", slope, intercept, r_value)

            print(f"Iteration {meta_step + 1} Regression coefficient: "
                  f"Slope: {slope:.4f}, Intercept: {intercept:.4f}, "
                  f"R-value: {r_value:.4f}, P-value: {p_value:.4f}, Std-err: {std_err:.4f}")

            # Rate vs clean loss plots
            loss_values, weights, accuracy_values = compute_rate(
                config, inner_models[0], data_rater, test_loader
            )
            tag = f"regression_clean_step_{meta_step + 1:06d}"
            save_regression_plot(loss_values, weights, run_dir, tag, "clean loss")

            # Rate vs adv loss plots
            loss_values_adv, weights_adv, accuracy_values_adv = compute_rate_adv(
                config, inner_models[0], data_rater, test_loader
            )
            tag = f"regression_adv_step_{meta_step + 1:06d}"
            save_regression_plot(loss_values_adv, weights_adv, run_dir, tag, "adv loss")

            # Plot accuracy vs clean loss
            plt.figure(figsize=(10,6))
            plt.plot(accuracy_values, label='Accuracy (Clean)', color='orange')
            plt.xlabel('Meta Step')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Over Meta Steps')
            plt.legend()
            plt.grid(True, linestyle=":")
            acc_plot_path = os.path.join(run_dir, "plots", f"accuracy_curve_step_{meta_step + 1:06d}.png")
            plt.savefig(acc_plot_path, bbox_inches="tight", dpi=150)
            plt.close()

            # Plot accuracy vs adv loss
            plt.figure(figsize=(10,6))
            plt.plot(accuracy_values_adv, label='Accuracy (Adv)', color='blue')
            plt.xlabel('Meta Step')
            plt.ylabel('Accuracy')
            plt.title('Adversarial Accuracy Over Meta Steps')
            plt.legend()
            plt.grid(True, linestyle=":")
            acc_adv_plot_path = os.path.join(run_dir, "plots", f"accuracy_adv_curve_step_{meta_step + 1:06d}.png")
            plt.savefig(acc_adv_plot_path, bbox_inches="tight", dpi=150)
            plt.close()

            # plot outer loss curve so far
            plt.figure(figsize=(10,6))
            plt.plot(logging_context.outer_loss, label='Loss (Adv)', color='blue')
            plt.plot(logging_context.outer_loss_clean, label='Loss (Clean)', color='orange')
            plt.xlabel('Meta Step')
            plt.ylabel('Loss')
            plt.title('Loss Over Meta Steps')
            plt.legend()
            plt.grid(True, linestyle=":")
            loss_plot_path = os.path.join(run_dir, "plots", f"loss_curve_step_{meta_step + 1:06d}.png")
            plt.savefig(loss_plot_path, bbox_inches="tight", dpi=150)
            plt.close()

    if config.save_data_rater_checkpoint:
        # Save the data rater model checkpoint
        checkpoint_path = os.path.join(run_dir, "data_rater.pt")
        torch.save(data_rater.state_dict(), checkpoint_path)

    if config.log:
        # Save the outer loss log as a CSV
        outer_loss_csv_path = os.path.join(run_dir, "outer_loss.csv")
        with open(outer_loss_csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["meta_step", "outer_loss"])
            for i, loss in enumerate(logging_context.outer_loss):
                writer.writerow([i, loss])

        outer_loss_clean_csv_path = os.path.join(run_dir, "outer_loss_clean.csv")
        with open(outer_loss_clean_csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["meta_step", "outer_loss_clean"])
            for i, loss in enumerate(logging_context.outer_loss_clean):
                writer.writerow([i, loss])

    slope, intercept, r_value, p_value, std_err, _, _ = \
        compute_regression_coefficient(config, dataset_handler, data_rater, test_loader)
    print(f"Regression coefficient: "
          f"Slope: {slope:.4f}, "
          f"Intercept: {intercept:.4f}, "
          f"R-value: {r_value:.4f}, "
          f"P-value: {p_value:.4f}, "
          f"Std-err: {std_err:.4f}")

    print("\nTraining complete!")
    return data_rater
