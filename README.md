# Classification Task

Implement models for multi-classes, multi-labels classification tasks

# 1. Architectures

- [ ] [MobileNet V1: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
- [X] [MobileNet V2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
- [X] [MobileNet V3: Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
- [X] [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)
- [X] [PP-LCNet: : A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf)
- [ ] [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803v2.pdf)
- [ ] [An image is worth 16X16 words: transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929v2.pdf)

# 2. Criteria and Metric

- [X] Cross entropy
- [X] Focal Loss
- [ ] [PolyLoss: A Polynomial Expansion Perspective of Loss Functions](https://arxiv.org/pdf/2204.12511.pdf)
- [X] Accuracy, Recall, Precision, F1-Score
- [ ] FAR (False Acception Rate)
- [ ] FRR (False Rejection Rate)

# 3. Usage

### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py configs/MNIST/training.yaml --num-epochs 20 --gpu-indices 0,1,2,3
```

### Testing

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py configs/MNIST/testing.yaml --gpu-indices 0,1,2,3 --checkpoint-path <str>
```

### Resume

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config config/MNIST/training.yaml --num-epoch 20 --num-gpus 0,1,2,3 --resume-path <str>
```

### Note. Run TensorBoard on Server

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

# 4. Techniques

1. Enhance performace by applying `mixup`

   References:
	- [1][mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf)
	- [2] [mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10)
	- [3] [Enhancing Neural Networks with Mixup in Pytorch](https://towardsdatascience.com/enhancing-neural-networks-with-mixup-in-pytorch-5129d261bc4a)

3. Pruning Techniques
4. Semi-supervised Learning
5. Unsupervised Learning
6. Self-supervised Learning
