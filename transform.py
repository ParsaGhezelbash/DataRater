from inspect import ismethod
import torch

from attack import fgsm_attack

def corrupt(samples: torch.Tensor, corruption_probability: float=0.1, corruption_fraction_range: tuple = (0.1, 0.9)) -> torch.Tensor:
    """
    Corrupts the input tensor by randomly setting a fraction of its elements to zero.

    Args:
        x (torch.Tensor): The input tensor to be corrupted.
        corruption_level (float): The fraction of elements to be set to zero (between 0 and 1).

    Returns:
        torch.Tensor: The corrupted tensor.
    """
    if torch.rand(1).item() < corruption_probability:
        corruption_fraction = torch.FloatTensor(1).uniform_(
            corruption_fraction_range[0],
            corruption_fraction_range[1]
        ).item()

        if corruption_fraction == 0.0:
            return samples

        corrupted_images = samples.clone()
        # Ensure it works for single images (B, C, H, W) or batches
        if len(samples.shape) == 3:
            images = images.unsqueeze(0)

        batch_size, _, height, width = samples.shape
        num_pixels_to_corrupt = int(corruption_fraction * (height * width))

        for i in range(batch_size):
            indices = torch.randperm(height * width, device=samples.device)[:num_pixels_to_corrupt]
            row_indices, col_indices = indices // width, indices % width
            random_pixels = torch.rand(num_pixels_to_corrupt, device=samples.device) * 2 - 1
            corrupted_images[i, 0, row_indices, col_indices] = random_pixels

        return corrupted_images
    
    
def identity(samples: torch.Tensor) -> torch.Tensor:
    """
    Returns the input tensor unchanged.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The unchanged input tensor.
    """
    return samples


def noise(samples: torch.Tensor, noise_level: float=0.1) -> torch.Tensor:
    """
    Adds Gaussian noise to the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        noise_level (float): The standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: The tensor with added Gaussian noise.
    """
    noise = torch.randn_like(samples) * noise_level
    return samples + noise


def fgsm(samples: torch.Tensor, labels: torch.Tensor, model, loss_fn, epsilon) -> torch.Tensor:
    """
    Applies the Fast Gradient Sign Method (FGSM) to perturb the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        epsilon (float): The magnitude of the perturbation.
        gradient (torch.Tensor): The gradient of the loss with respect to the input tensor.

    Returns:
        torch.Tensor: The perturbed tensor.
    """
    perturbed_samples = fgsm_attack(model, samples, labels, loss_fn, epsilon)
    return perturbed_samples


def get_transform(name: str):
    """
    Retrieves a transformation function by name.

    Args:
        name (str): The name of the transformation function.

    Returns:
        function: The corresponding transformation function.

    Raises:
        ValueError: If the transformation name is not recognized.
    """
    transforms = {
        "corrupt": corrupt,
        "identity": identity,
        "noise": noise,
        "fgsm": fgsm
    }
    if name not in transforms:
        raise ValueError(f"Transform '{name}' not recognized. Available transforms: {list(transforms.keys())}")
    return transforms[name]