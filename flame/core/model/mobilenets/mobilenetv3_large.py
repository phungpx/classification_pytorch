from torch import nn
from .mobilenetv3 import mobilenet_v3_large


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes, pretrained, features_fixed):
        super(MobileNetV3Large, self).__init__()
        self.model = mobilenet_v3_large(pretrained=pretrained)
        self.model.features.requires_grad_(not features_fixed)
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
