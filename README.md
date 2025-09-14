# DataRater: Meta-Learning for Data Quality Assessment

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
pip install torch torchvision tqdm matplotlib numpy scipy
```

### Basic Usage

```bash
python data_rater_main.py --dataset_name=mnist --meta_steps=1000
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
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
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

config = DataRaterConfig(
    dataset_name="mnist",
    inner_model_class="ToyCNN",
    data_rater_model_class="DataRater",
    batch_size=128,
    inner_lr=1e-3,
    outer_lr=1e-3,
    meta_steps=1000,
    inner_steps=5,
    num_inner_models=4
)

trained_data_rater = run_meta_training(config)
```

### Advanced Configuration

```python
config = DataRaterConfig(
    dataset_name="my_dataset",
    inner_model_class="MyTaskModel", 
    data_rater_model_class="MyDataRater",
    batch_size=64,
    train_split_ratio=0.8,
    inner_lr=5e-4,
    outer_lr=1e-4,
    meta_steps=2000,
    inner_steps=10,
    meta_refresh_steps=20,  # Refresh inner models every 20 steps
    grad_clip_norm=0.5,
    num_inner_models=8,     # Population of 8 inner models
    device="cuda",
    save_data_rater_checkpoint=True,
    log=True
)

trained_data_rater = run_meta_training(config)
```

### Command Line Usage

```bash
python data_rater_main.py \
    --dataset_name=mnist \
    --inner_model_name=ToyCNN \
    --data_rater_model_name=DataRater \
    --batch_size=128 \
    --inner_lr=1e-3 \
    --outer_lr=1e-3 \
    --meta_steps=1000 \
    --inner_steps=5 \
    --num_inner_models=4 \
    --save_data_rater_checkpoint=True \
    --log=True
```

## Key Parameters

- `meta_steps`: Number of outer loop optimization steps
- `inner_steps`: Number of gradient steps for each inner model per meta-step  
- `num_inner_models`: Population size of inner models (helps with stability)
- `meta_refresh_steps`: How often to reinitialize the inner model population
- `inner_lr`/`outer_lr`: Learning rates for inner models and DataRater respectively
- `grad_clip_norm`: Gradient clipping threshold

## Architecture Details

### Meta-Training Process

1. **Population Management**: Maintains multiple inner models, refreshed periodically
2. **Inner Loop**: Each inner model trains on reweighted data using DataRater scores
3. **Outer Loop**: DataRater optimized based on inner models' validation performance
4. **Data Weighting**: DataRater assigns quality scores, converted to sample weights via softmax

### Data Corruption

The framework includes built-in data corruption for robustness training:

```python
from datasets import DataCorruptionConfig

corruption_config = DataCorruptionConfig(
    corruption_probability=0.25,        # 25% of training samples corrupted
    corruption_fraction_range=(0.1, 0.9) # 10-90% of pixels corrupted when applied
)
```

## Example: MNIST Experiment

```bash
# Run the included MNIST experiment
bash experiments/mnist_v1.sh
```

This trains a DataRater on corrupted MNIST data, learning to identify and downweight corrupted samples while prioritizing clean, informative examples.

## File Structure

```
datarater/
 README.md
 config.py              # Configuration class
 data_rater_main.py     # Main training script
 data_rater.py          # Core meta-training logic
 datasets.py            # Dataset implementations
 models.py              # Model definitions
 experiments/           # Experiment scripts
 data/                  # Dataset storage
```

## Contributing

When adding new datasets or models:
1. Follow the abstract base class interfaces
2. Register new components in the factory functions
3. Test with a simple experiment script
4. Add corruption strategies appropriate for your data type
