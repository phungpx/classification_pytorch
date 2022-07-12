# Classification Task
Implement models for multi-classes, multi-labels classification tasks

# Note. Run TensorBoard on Server
You have to create a ssh connection using port forwarding:
```bash
ssh -L 16006:127.0.0.1:6006 user@host
```
Then you run the tensorboard command:

```bash
tensorboard --logdir=/path/to/logs
```
Then you can easily access the tensorboard in your browser under:
```bash
localhost:16006/
```

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

# Techniques
## 1. Enhancing Neural Networks with Mixup
1.0 References
[1] [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf)
[2] [mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10)
[3] [Enhancing Neural Networks with Mixup in Pytorch](https://towardsdatascience.com/enhancing-neural-networks-with-mixup-in-pytorch-5129d261bc4a)

*Randomly mixing up images, and it works better?*
1.1 What is Mixup?
![image](https://user-images.githubusercontent.com/61035926/167535065-3e054115-937f-4acc-99bf-4657d1f265fc.png)
* The concept of mixup mathematically:
```bash
image = lambda * image1 + (1 - lambda) * image2
label = lambda * label1 + (1 - lambda) * label2

given:
- lambda: a random number from a given beta distribution.
- image1 (label1 with one-hot encoding of label2) and image2 (with one-hot encoding of label2).
```
* Beta distribution: very high probability of lambda being close to 0 or 1.
![image](https://user-images.githubusercontent.com/61035926/167536347-81a687d0-efc0-47e9-9adb-27430fadea84.png)

1.2 

## 2. Pruning Techniques
## 3. Semi-supervised Learning
## 4. Unsupervised Learning
## 5. Self-supervised Learning
