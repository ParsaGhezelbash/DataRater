import torch
import torch.nn as nn
import torch.optim as optim


class ToyCNN(nn.Module):
    """The inner model that gets trained on re-weighted data."""

    def __init__(self):
        super(ToyCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=32*5*5, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        return self.layers(x)


class DataRater(nn.Module):
    """The outer model (meta-learner) that learns to rate data."""

    def __init__(self, temperature=1.0):
        super(DataRater, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head = nn.Linear(400, 1)
        self.temperature = temperature

    def forward(self, x):
        features = self.layers(x)
        return self.head(features).squeeze(-1)


def construct_model(model_class):
    if model_class == 'ToyCNN':
        return ToyCNN()
    elif model_class == 'DataRater':
        return DataRater()
    else:
        raise ValueError(f"Model {model_class} not found")