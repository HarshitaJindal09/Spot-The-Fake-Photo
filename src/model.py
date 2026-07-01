import torch
import torch.nn as nn
from torchvision import models


class ScreenDetector(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)