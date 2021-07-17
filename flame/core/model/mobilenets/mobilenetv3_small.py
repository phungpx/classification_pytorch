from torch import nn
from .mobilenetv3 import mobilenet_v3_small


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes, pretrained, features_fixed):
        super(MobileNetV3Small, self).__init__()
        self.model = mobilenet_v3_small(pretrained=pretrained)
        self.model.features.requires_grad_(not features_fixed)
        self.model.classifier[-1] = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
