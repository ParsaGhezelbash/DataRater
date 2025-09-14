import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from dataclasses import dataclass

from config import DataRaterConfig

# --- 1. Configuration (Using a dataclass for clarity) ---
@dataclass
class DataCorruptionConfig:
    """Configuration for applying corruption to the training dataset."""
    corruption_probability: float = 0.25 # Chance an image in the train set gets corrupted
    corruption_fraction_range: tuple = (0.1, 0.9) # Min/Max corruption fraction if applied


# --- 2. Abstract Base Class for Datasets ---
class DataRaterDataset(ABC):
    """
    Abstract base class for creating datasets compatible with the DataRater.
    It defines a standard interface for getting data loaders and applying corruption.
    """

    @abstractmethod
    def corrupt_samples(self, samples: torch.Tensor, corruption_fraction: float) -> torch.Tensor:
        """
        Applies a specified level of corruption to a batch of samples.

        Args:
            samples: A batch of samples to corrupt.
            corruption_fraction: The fraction of pixels to alter.

        Returns:
            A new tensor containing the corrupted samples.
        """
        pass

    @abstractmethod
    def get_loaders(self, batch_size: int, train_split_ratio: float, train_corruption_config: DataCorruptionConfig) -> tuple:
        """
        Creates and returns the training, validation, and test data loaders.
        The training loader should yield a mix of clean and corrupted data.

        Args:
            batch_size: The number of samples per batch.
            train_split_ratio: The fraction of the training data to use for the inner loop.
            train_corruption_config: Configuration for corrupting the training data.

        Returns:
            A tuple containing (train_loader, val_loader, test_loader).
        """
        pass

# --- 3. Custom Dataset Wrapper for On-the-Fly Corruption ---

class CorruptedSubset(Dataset):
    """
    A wrapper for a PyTorch Subset that applies corruption to a portion of the
    training data on-the-fly.
    """
    def __init__(self, original_subset: Subset, corruption_fn, config: DataCorruptionConfig):
        self.original_subset = original_subset
        self.corruption_fn = corruption_fn
        self.config = config

    def __len__(self):
        return len(self.original_subset)

    def __getitem__(self, index):
        image, label = self.original_subset[index]

        # Probabilistically decide whether to corrupt the image
        if torch.rand(1).item() < self.config.corruption_probability:
            # Apply a random level of corruption within the specified range
            corruption_fraction = torch.FloatTensor(1).uniform_(
                self.config.corruption_fraction_range[0],
                self.config.corruption_fraction_range[1]
            ).item()
            # The corruption function expects a batch, so we add a dimension and remove it
            image = self.corruption_fn(image.unsqueeze(0), corruption_fraction).squeeze(0)

        return image, label


# --- 4. Concrete Implementation for MNIST ---

class MNISTDataRaterDataset(DataRaterDataset):
    """
    An implementation of DataRaterDataset for the MNIST dataset.
    - The training data loader provides a mix of clean and corrupted images.
    - The validation and test loaders provide clean images.
    """

    def corrupt_samples(self, samples: torch.Tensor, corruption_fraction: float) -> torch.Tensor:
        """
        Corrupts a batch of images by REPLACING a fraction of pixels with random noise.
        """
        if corruption_fraction == 0.0:
            return samples

        corrupted_images = samples.clone()
        # Ensure it works for single images (B, C, H, W) or batches
        if len(samples.shape) == 3:
            images = images.unsqueeze(0) # Add batch dimension if missing

        batch_size, _, height, width = samples.shape
        num_pixels_to_corrupt = int(corruption_fraction * (height * width))

        for i in range(batch_size):
            indices = torch.randperm(height * width, device=samples.device)[:num_pixels_to_corrupt]
            row_indices, col_indices = indices // width, indices % width
            random_pixels = torch.rand(num_pixels_to_corrupt, device=samples.device) * 2 - 1
            corrupted_images[i, 0, row_indices, col_indices] = random_pixels

        return corrupted_images


    def get_loaders(self, batch_size: int, train_split_ratio: float, train_corruption_config: DataCorruptionConfig) -> tuple:
        """Prepares the correctly configured MNIST data loaders."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Create a train/validation split from the ORIGINAL clean dataset
        num_train = int(len(full_train_dataset) * train_split_ratio)
        num_val = len(full_train_dataset) - num_train
        train_subset, val_subset = random_split(full_train_dataset, [num_train, num_val])

        # ** CRITICAL STEP: Wrap the training subset to apply corruption on-the-fly **
        corrupted_train_dataset = CorruptedSubset(
            original_subset=train_subset,
            corruption_fn=self.corrupt_samples,
            config=train_corruption_config
        )

        print(f"Train set: {len(corrupted_train_dataset)} images (probabilistically corrupted)")
        print(f"Validation set: {len(val_subset)} images (clean)")
        print(f"Test set: {len(test_dataset)} images (clean)")

        # Create DataLoaders
        train_loader = DataLoader(corrupted_train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False) # Validation data is CLEAN
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Test data is CLEAN

        return train_loader, val_loader, test_loader

# --- 5. Updated Factory Function ---

def get_dataset_loaders(config: DataRaterConfig) -> tuple:
    """
    Factory function to get the data loaders based on the configuration.

    Args:
        config: A DataRaterConfig object specifying the dataset and parameters.

    Returns:
        A tuple containing (train_loader, val_loader, test_loader).
    """
    if config.dataset_name == "mnist":
        dataset_handler = MNISTDataRaterDataset()
    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported.")

    return dataset_handler.get_loaders(
        config.batch_size,
        config.train_split_ratio,
        DataCorruptionConfig()
    )