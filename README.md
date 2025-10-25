# DataRater for Robustness: Meta-Learned Dataset Curation to improve Model Robustness

**Implementation of the DataRater (Calian et. al.) paper: https://arxiv.org/abs/2505.17895**

DataRater is a meta-learning framework that learns to assess data quality and reweight training samples to improve model performance. It uses a two-level optimization approach where an outer "DataRater" model learns to score data samples while inner models are trained on the reweighted data.

## Overview

The framework consists of:
- **Inner Models**: Task-specific models (e.g., CNN classifiers) trained on reweighted data
- **DataRater Model**: Meta-learner that assigns quality scores to training samples
- **Meta-Training Loop**: Alternates between inner model training and DataRater optimization

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
sh runbin/mnist_v1.sh
```

## Setting Up a New Dataset

To add support for a new dataset, create a class inheriting from `DataRaterDataset`:

```python
from datasets import DataRaterDataset, DataCorruptionConfig
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MyDataset(DataRaterDataset):
    def corrupt_samples(self, samples: torch.Tensor, corruption_fraction: float) -> torch.Tensor:
        """Apply corruption to samples (e.g., noise, occlusion)"""
        if corruption_fraction == 0.0:
            return samples
        
        corrupted = samples.clone()
        # Implement your corruption logic here
        # Example: add gaussian noise
        noise = torch.randn_like(samples) * corruption_fraction
        corrupted = samples + noise
        return torch.clamp(corrupted, -1, 1)
    
    def get_loaders(self, batch_size: int, train_split_ratio: float, 
                   train_corruption_config: DataCorruptionConfig) -> tuple:
        """Create train/val/test data loaders"""
        # Load your dataset
        # Apply transforms
        # Create train/validation split
        # Wrap training data with CorruptedSubset for on-the-fly corruption
        
        #...
        # Your dataset loading logic here
        train_loader = DataLoader(corrupted_train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
```

Register your dataset in `datasets.py`:

```python
def get_dataset_loaders(config: DataRaterConfig) -> tuple:
    if config.dataset_name == "mnist":
        dataset_handler = MNISTDataRaterDataset()
    elif config.dataset_name == "my_dataset":
        dataset_handler = MyDataset()
    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported.")
```

## Creating Custom Models

### Inner Model (Task Model)

Create models that inherit from `nn.Module`:

```python
import torch.nn as nn

class MyTaskModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyTaskModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)
```

### DataRater Model

Create a model that outputs quality scores for input samples:

```python
class MyDataRater(nn.Module):
    def __init__(self):
        super(MyDataRater, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.scorer = nn.Linear(64 * 4 * 4, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.scorer(features).squeeze(-1)
```

Register your models in `models.py`:

```python
def construct_model(model_class):
    if model_class == 'ToyCNN':
        return ToyCNN()
    elif model_class == 'DataRater':
        return DataRater()
    elif model_class == 'MyTaskModel':
        return MyTaskModel()
    elif model_class == 'MyDataRater':
        return MyDataRater()
    else:
        raise ValueError(f"Model {model_class} not found")
```

## Running Meta-Training Loop

### Basic Configuration

```python
from config import DataRaterConfig
from data_rater import run_meta_training

args = parse_args()

config = DataRaterConfig(
    dataset_name=args.dataset_name,
    inner_model_class=args.inner_model_name,
    data_rater_model_class=args.data_rater_model_name,
    batch_size=args.batch_size,
    train_split_ratio=args.train_split_ratio,
    inner_lr=args.inner_lr,
    outer_lr=args.outer_lr,
    meta_steps=args.meta_steps,
    inner_steps=args.inner_steps,
    meta_refresh_steps=args.meta_refresh_steps,
    grad_clip_norm=args.grad_clip_norm,
    num_inner_models=args.num_inner_models,
    device=args.device,
    loss_type=args.loss_type,
    save_data_rater_checkpoint=args.save_data_rater_checkpoint,
    log=args.log,
    eval_frequency=args.eval_frequency,
    attack=args.attack,
    attack_eps=args.attack_eps,
    attack_steps=args.attack_steps,
    attack_eval_steps=args.attack_eval_steps,
    attack_step_size=args.attack_step_size,
    model_update=args.model_update,
)
run_meta_training(config)
```

## Key Parameters

- `dataset_name`: Name of the dataset to use for training and evaluation.
- `inner_model_class`: Model class name for the inner (task) model.
- `data_rater_model_class`: Model class name for the DataRater (data weighting) model.
- `batch_size`: Number of samples per batch for both inner and outer loops.
- `train_split_ratio`: Fraction of data used for training (rest for validation).
- `inner_lr`: Learning rate for inner model updates during the inner loop.
- `outer_lr`: Learning rate for DataRater updates during the outer loop.
- `meta_steps`: Total number of meta-training (outer loop) steps to run.
- `inner_steps`: Number of gradient steps each inner model takes per meta-step.
- `meta_refresh_steps`: Frequency (in meta-steps) to reinitialize the inner model population.
- `grad_clip_norm`: Maximum norm for gradient clipping during meta-optimization.
- `num_inner_models`: Number of inner models in the population (improves stability).
- `device`: Device to use for training (e.g., "cuda" or "cpu").
- `loss_type`: Loss function to use for training (e.g., "mse" or "cross_entropy").
- `save_data_rater_checkpoint`: Whether to save the trained DataRater model checkpoint.
- `log`: Whether to log training metrics and save logs to disk.
- `eval_frequency`: Frequency of evaluation during training (in meta-steps).
- `attack`: Whether to use adversarial attacks (default: False).
- `attack_eps`: Attack perturbation strength (default: 8/255).
- `attack_steps`: Number of attack steps (default: 10).
- `attack_eval_steps`: Number of attack evaluation steps (default: 20).
- `attack_step_size`: Attack step size (default: 2/255).
- `model_update`: Whether to update the model (default: False).

## Architecture Details

### Meta-Training Process

1. **Population Management**: Maintains multiple inner models, refreshed periodically
2. **Inner Loop**: Each inner model trains on reweighted data using DataRater scores
3. **Outer Loop**: DataRater optimized based on inner models' validation performance
4. **Data Weighting**: DataRater assigns quality scores, converted to sample weights via softmax


## MNIST Experiment

```bash
python data_rater_main.py \
  --dataset_name=mnist \
  --inner_model_name=ToyCNN \
  --data_rater_model_name=DataRater \
  --train_split_ratio=0.8 \
  --batch_size=128 \
  --inner_lr=1e-3 \
  --outer_lr=3e-4 \
  --meta_steps=1000 \
  --inner_steps=2 \
  --meta_refresh_steps=150 \
  --grad_clip_norm=5.0 \
  --num_inner_models=8 \
  --loss_type=cross_entropy \
  --save_data_rater_checkpoint=True \
  --log=True \
  --eval_frequency=100 \
  --attack=True \
  --attack_eps=0.031 \
  --attack_steps=10 \
  --attack_eval_steps=20 \
  --attack_step_size=0.007 \
  --model_update=True
```

You can find the saved DataRater model checkpoint (`data_rater.pt`) in the `mnist_20250920_1037_a11efc10/`. The checkpoint and associated data are useful for further analysis, reproducibility, or resuming training.

### Downstream Comparison Results

We compared three training strategies on corrupted MNIST data:

- **Baseline** – standard training, no dropping.  
- **Filtered** – drop the bottom 10% of samples per batch using a trained *DataRater*.  
- **Random-Drop** – drop the bottom 10% at random (control).  

Each experiment was repeated **5 times with different seeds** to account for randomness.

| Method       | Test Accuracy (mean ± std) |
|--------------|-----------------------------|
| Baseline     | **0.9708 ± 0.0030** |
| Filtered     | **0.9732 ± 0.0036** |
| Random-Drop  | **0.9699 ± 0.0033** |


## Contributing

When adding new datasets or models:
1. Follow the abstract base class interfaces
2. Register new components in the factory functions
3. Test with a simple experiment script
4. Add corruption strategies appropriate for your data type
