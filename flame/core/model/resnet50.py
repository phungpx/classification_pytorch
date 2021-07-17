import torch.nn as nn
from torchvision import models


class Resnet50(nn.Module):
    def __init__(self, num_classes, pretrained, features_fixed):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(progress=True, pretrained=pretrained)
        for parameters in self.model.parameters():
            parameters.requires_grad = not features_fixed
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
