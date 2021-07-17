import torch
from torch import nn
from torchvision import models


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, pretrained, features_fixed):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.model.features.requires_grad_(not features_fixed)
        self.model.classifier[1] = torch.nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
