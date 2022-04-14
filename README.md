# Classification Task
Implement models for multi-classes, multi-labels classification tasks

# 1. Architectures
- [ ] [MobileNet V1: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
- [x] [MobileNet V2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
- [x] [MobileNet V3: Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
- [x] [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)
- [x] [PP-LCNet: : A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf)

```python3
from mobilenets.mobilenetv3 import MobileNetV3Small, MobileNetV3Large

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MobileNetV3Small(num_classes=10, pretrained=True).to(device)
# model = MobileNetV3Large(num_classes=10, pretrained=True).to(device)
dummy_input = torch.rand(size=[8, 3, 224, 224], dtype=torch.float32, device=device)
output = model(dummy_input)

print(f"Input Shape: {dummy_input.shape}")
print(f"Output Shape: {output.shape}")
print(f"Number of parameters: {sum((p.numel() for p in model.parameters() if p.requires_grad))}")

```

# 2. Criteria and Metric
- [x] Cross entropy
- [x] Focal Loss 
- [x] Accuracy, Recall, Precision, F1-Score
- [ ] FAR (False Acception Rate)
- [ ] FRR (False Rejection Rate)
