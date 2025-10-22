import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18


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
    

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=None)    # or weights="IMAGENET1K_V1" for pretrained
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)  # smaller kernel
        self.model.maxpool = nn.Identity()      # remove downsampling
        self.model.fc = nn.Linear(512, 10)      # CIFAR-10 = 10 classes

    def forward(self, x):
        return self.model(x)
    

class ResNetMnist(nn.Module):
    def __init__(self):
        super(ResNetMnist, self).__init__()
        self.model = resnet18(weights=None)    # or weights="IMAGENET1K_V1" for pretrained
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)  # adapt for 1-channel input
        self.model.maxpool = nn.Identity()      # remove downsampling
        self.model.fc = nn.Linear(512, 10)      # MNIST = 10 classes

    def forward(self, x):
        return self.model(x)


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


class DataRaterResNet(nn.Module):
    def __init__(self, in_channels=3, pretrained=False, temperature=1.0):
        super().__init__()
        resnet = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # Adapt for smaller images or 1‑channel input
        if in_channels == 1:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # Avoid excessive downsampling for 32×32 inputs
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))  # remove fc
        self.head = nn.Linear(512, 1)
        self.temperature = temperature

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.head(f).squeeze(-1)
    

class ToyMLP(nn.Module):
    """Simple 2-layer MLP for regression tasks."""
    
    def __init__(self, input_dim=10, hidden_dim=64):
        super(ToyMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single output for regression
        )
    
    def forward(self, x):
        return self.layers(x).squeeze(-1)  # Remove last dimension to get shape (batch_size,)

class RegressionDataRater(nn.Module):
    """The outer model (meta-learner) that learns to rate regression data."""
    
    def __init__(self, input_dim=10, hidden_dim=64, temperature=1.0):
        super(RegressionDataRater, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Using Tanh like in the CNN version
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = temperature
    
    def forward(self, x):
        return self.layers(x).squeeze(-1)  # Remove last dimension


def construct_model(model_class):
    if model_class == 'ToyCNN':
        return ToyCNN()
    elif model_class == 'ResNet18':
        return ResNet18()
    elif model_class == 'ResNetMnist':
        return ResNetMnist()
    elif model_class == 'DataRater':
        return DataRater()
    elif model_class == 'DataRaterResNet':
        return DataRaterResNet()
    elif model_class == 'ToyMLP':
        return ToyMLP()
    elif model_class == 'RegressionDataRater':
        return RegressionDataRater()
    else:
        raise ValueError(f"Model {model_class} not found")