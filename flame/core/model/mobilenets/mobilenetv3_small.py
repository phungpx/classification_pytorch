from torch import nn, Tensor
from .mobilenetv3 import mobilenet_v3_small


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained: bool = False, features_fixed: bool = False) -> None:
        super(MobileNetV3Small, self).__init__()
        self.model = mobilenet_v3_small(pretrained=pretrained)
        self.model.features.requires_grad_(not features_fixed)
        self.model.classifier[-1] = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MobileNetV3Small(num_classes=1000).to(device)
    dummy_input = torch.rand(size=[8, 3, 224, 224], dtype=torch.float32, device=device)
    output = model(dummy_input)

    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Number of parameters: {sum((p.numel() for p in model.parameters() if p.requires_grad))}")

