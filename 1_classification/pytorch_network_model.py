
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class MODEL(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet18(pretrained=False)
        self.classifier = nn.Sequential(
            nn.Dropout()
            , nn.Linear(1000, num_classes)
            , nn.Sigmoid()
        )
    def forward(self, x):
        x = self.network(x)
        return self.classifier(x)