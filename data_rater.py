import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from scipy import stats  # Import the stats module from SciPy
import uuid
from dataclasses import dataclass
from datetime import datetime

from models import construct_model
from config import DataRaterConfig
from datasets import get_dataset_loaders

@dataclass
class LoggingContext:
    run_id: str
    outer_loss: list[float]

def inner_loop_step(config: DataRaterConfig,
                    inner_model, 
                    inner_optimizer, 
                    data_rater, 
                    inner_batch):
    """
    Performs a single training step for the inner model.
    """
    inner_model.train()
    data_rater.train()

    inner_samples, inner_labels = inner_batch
    inner_samples, inner_labels = inner_samples.to(
        config.device), inner_labels.to(config.device)

    # Zero gradients for the inner optimizer
    inner_optimizer.zero_grad()

    # Get ratings and compute weights for the current batch
    with torch.no_grad():  # Don't track gradients for data rater here
        ratings = data_rater(inner_samples)
        weights = torch.softmax(ratings, dim=0)

    # Get predictions from the inner model
    logits = inner_model(inner_samples)

    # Calculate per-sample cross-entropy loss and apply weights
    if config.loss_type == 'mse':
        per_sample_losses = nn.functional.mse_loss(logits, inner_labels, reduction='none')
    elif config.loss_type == 'cross_entropy':
        per_sample_losses = nn.functional.cross_entropy(
            logits, inner_labels, reduction='none')
    else:
        raise ValueError(f"Loss type {config.loss_type} not supported")
    weighted_loss = (per_sample_losses * weights).mean()
    # Update the inner model
    weighted_loss.backward()
    torch.nn.utils.clip_grad_norm_(inner_model.parameters(), config.grad_clip_norm)
    inner_optimizer.step()


def outer_loop_step(config: DataRaterConfig,
                    logging_context: LoggingContext,
                    data_rater, 
                    outer_optimizer, 
                    inner_models, 
                    inner_optimizers, 
                    train_iterator, 
                    val_iterator,
                    train_loader,
                    val_loader):
    """
    Performs one full meta-update step using a provided POPULATION of inner models.
    """
    # --- 1. Run the inner loop for each model in the provided population ---
    for model, optimizer in zip(inner_models, inner_optimizers):
        for _ in range(config.inner_steps):
            try:
                inner_batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                inner_batch = next(train_iterator)

            inner_loop_step(config, model, optimizer, data_rater, inner_batch)

    # --- 2. Perform the outer update on the DataRater ---
    data_rater.train()
    for model in inner_models:
        model.eval()

    # Get a single outer batch to evaluate all models
    try:
        outer_samples, outer_labels = next(val_iterator)
    except StopIteration:
        val_iterator = iter(val_loader)
        outer_samples, outer_labels = next(val_iterator)
    outer_samples, outer_labels = outer_samples.to(
        config.device), outer_labels.to(config.device)

    # --- 3. Calculate and average the outer loss across the population ---
    outer_losses = []
    for model in inner_models:
        outer_logits = model(outer_samples)
        if config.loss_type == 'mse':
            outer_loss = nn.functional.mse_loss(outer_logits, outer_labels)
        elif config.loss_type == 'cross_entropy':
            outer_loss = nn.functional.cross_entropy(outer_logits, outer_labels)
        else:
            raise ValueError(f"Loss type {config.loss_type} not supported")
        outer_losses.append(outer_loss)

    average_outer_loss = torch.mean(torch.stack(outer_losses))

    # --- 4. Update the DataRater using the averaged loss ---
    outer_optimizer.zero_grad()
    average_outer_loss.backward()
    torch.nn.utils.clip_grad_norm_(data_rater.parameters(), config.grad_clip_norm)
    outer_optimizer.step()

    logging_context.outer_loss.append(average_outer_loss.item())

    return average_outer_loss.item()

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
            scores = data_rater(individually_corrupted_batch)
            scores_softmax = torch.softmax(scores, dim=0)
            weights.extend(scores_softmax.cpu().numpy())
            corruption_levels.extend(fractions_for_this_batch)

    slope, intercept, r_value, p_value, std_err = stats.linregress(corruption_levels, weights)
    return slope, intercept, r_value, p_value, std_err

def run_meta_training(config: DataRaterConfig):
    """
    Initializes models and orchestrates the main meta-training loop.
    Manages a population of inner models and refreshes them periodically.
    """
    # Generate run_id with dataset name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"{config.dataset_name}_{timestamp}_{str(uuid.uuid4())[:8]}"

    print(f"Run ID: {run_id}")
    logging_context = LoggingContext(run_id=run_id, outer_loss=[])
    # --- Initial Setup ---
    data_rater = construct_model(config.data_rater_model_class).to(config.device)
    outer_optimizer = optim.Adam(data_rater.parameters(), lr=config.outer_lr)

    dataset_handler, (train_loader, val_loader, test_loader) = get_dataset_loaders(config)
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)

    # Declare lists for the population of inner models and optimizers
    inner_models, inner_optimizers = None, None

    print(
        f"Starting meta-training with a population of {config.num_inner_models} models,\
             refreshing every {config.meta_refresh_steps} steps.")

    # --- The Main Training Loop ---
    for meta_step in tqdm(range(config.meta_steps), desc="Meta-Training"):
        # Periodically refresh the entire population of inner models
        if meta_step % config.meta_refresh_steps == 0:
            tqdm.write(
                f"\n[Meta-Step {meta_step}] Refreshing inner model population...")
            inner_models = [construct_model(config.inner_model_class).to(config.device)
                            for _ in range(config.num_inner_models)]
            inner_optimizers = [optim.Adam(
                model.parameters(), lr=config.inner_lr) for model in inner_models]

        # Pass the current population of models and optimizers to the step function
        outer_loss = outer_loop_step(
            config,
            logging_context,
            data_rater, outer_optimizer,
            inner_models, inner_optimizers,  # Pass the lists
            train_iterator, val_iterator,
            train_loader, val_loader
        )

        if (meta_step + 1) % 10 == 0:
            tqdm.write(
                f"  [Meta-Step {meta_step + 1}/{config.meta_steps}] Outer Loss: {outer_loss:.4f}")

    if config.save_data_rater_checkpoint:
        run_dir = os.path.join("experiments", run_id)
        os.makedirs(run_dir, exist_ok=True)

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


    slope, intercept, r_value, p_value, std_err = compute_regression_coefficient(config, dataset_handler, data_rater, test_loader)
    print(f"Regression coefficient: "
          f"Slope: {slope:.4f}, "
          f"Intercept: {intercept:.4f}, "
          f"R-value: {r_value:.4f}, "
          f"P-value: {p_value:.4f}, "
          f"Std-err: {std_err:.4f}")

    print("\n✅ Training complete!")
    return data_rater
