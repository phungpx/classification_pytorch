import math
import torch
from torch import nn


class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvSiLU, self).__init__()
        self.conv_silu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            SiLU(),)

    def forward(self, x):
        return self.conv_silu(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.SEnet = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_dim, kernel_size=1),  # C x 1 x 1 -> C/r x 1 x 1
            SiLU(),  # in original using ReLU
            nn.Conv2d(in_channels=reduced_dim, out_channels=in_channels, kernel_size=1),  # C/r x 1 x 1 -> C x 1 x 1
            nn.Sigmoid())

    def forward(self, x):
        return x * self.SEnet(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio,
                 reduction=4,  # r in squeeze-and-excitation optimization
                 survival_probability=0.8,):  # survival_probability of stochastic depth
        super(InvertedResidualBlock, self).__init__()
        self.survival_probability = survival_probability
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = math.floor(in_channels / reduction)

        if self.expand:
            self.expand_conv = ConvSiLU(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(
            ConvSiLU(in_channels=hidden_dim, out_channels=hidden_dim,
                     kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim),
            SqueezeExcitation(in_channels=hidden_dim, reduced_dim=reduced_dim),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels))

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_probability
        return torch.div(x, self.survival_probability) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        # (compound_coefficient, resolution, dropout_rate)
        self.efficient_parameters = {"b0": (0, 224, 0.2),
                                     "b1": (0.5, 240, 0.2),
                                     "b2": (1, 260, 0.3),
                                     "b3": (2, 300, 0.3),
                                     "b4": (3, 380, 0.4),
                                     "b5": (4, 456, 0.4),
                                     "b6": (5, 528, 0.5),
                                     "b7": (6, 600, 0.5)}

        width_factor, depth_factor, dropout_rate = self._calculate_factors(version)

        self.baseline_params = [['Conv', 32, 2, 3, 1],  # channels=32, stride=2, kernel_size=3, padding=1
                                ['MBConv', 1, 16, 1, 1, 3],  # expand_ratio=1, channels=16, layers=1, stride=1, kernel_size=3
                                ['MBConv', 6, 24, 2, 2, 3],  # expand_ratio=6, channels=24, layers=2, stride=2, kernel_size=3
                                ['MBConv', 6, 40, 2, 2, 5],  # expand_ratio=6, channels=40, layers=2, stride=2, kernel_size=5
                                ['MBConv', 6, 80, 3, 2, 3],  # expand_ratio=6, channels=80, layers=3, stride=2, kernel_size=3
                                ['MBConv', 6, 112, 3, 1, 5],  # expand_ratio=6, channels=112, layers=3, stride=1, kernel_size=5
                                ['MBConv', 6, 192, 4, 2, 5],  # expand_ratio=6, channels=192, layers=4, stride=2, kernel_size=5
                                ['MBConv', 6, 320, 1, 1, 3],  # expand_ratio=6, channels=320, layers=1, stride=1, kernel_size=3
                                ['Conv', 1280, 1, 1, 0],  # channels=1280, stride=1, kernel_size=1, padding=0
                                ['AvgPool', 1],  # output_size=1
                                ['Flatten', 1, -1],  # start_dim=1, end_dim=-1
                                ['Dropout', dropout_rate],  # dropout_rate=dropout_rate
                                ['Linear', num_classes]]  # out_features=num_classes

        self.features = self._create_network(width_factor=width_factor, depth_factor=depth_factor)

    def _calculate_factors(self, version, alpha=1.2, beta=1.1):
        compound_coefficient, resolution_factor, dropout_rate = self.efficient_parameters[version]
        depth_factor = alpha ** compound_coefficient
        width_factor = beta ** compound_coefficient
        return width_factor, depth_factor, dropout_rate

    def _create_network(self, width_factor, depth_factor):
        features = []
        in_channels = 3
        for params in self.baseline_params:
            if params[0] == 'Conv':
                channels, stride, kernel_size, padding = params[1:]
                channels = math.floor(channels * width_factor) if in_channels == 3 else math.ceil(channels * width_factor)
                features.append(ConvSiLU(in_channels=in_channels, out_channels=channels,
                                         kernel_size=kernel_size, stride=stride, padding=padding))
                in_channels = channels
            elif params[0] == 'MBConv':
                expand_ratio, channels, repeats, stride, kernel_size = params[1:]
                channels = 4 * math.ceil(math.floor(channels * width_factor) / 4)
                repeats = math.ceil(repeats * depth_factor)
                for layer in range(repeats):
                    padding = kernel_size // 2
                    stride = stride if layer == 0 else 1
                    features.append(InvertedResidualBlock(in_channels=in_channels, out_channels=channels,
                                                          expand_ratio=expand_ratio, stride=stride, kernel_size=kernel_size, padding=padding))
                    in_channels = channels
            elif params[0] == 'AvgPool':
                features.append(nn.AdaptiveAvgPool2d(output_size=params[1]))
            elif params[0] == 'Flatten':
                features.append(nn.Flatten(start_dim=params[1], end_dim=params[2]))
            elif params[0] == 'Dropout':
                features.append(nn.Dropout(p=params[1]))
            elif params[0] == 'Linear':
                features.append(nn.Linear(in_features=in_channels, out_features=params[1]))

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":
    version = "b0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EfficientNet(version=version, num_classes=1000).to(device)
    _, resolution, _ = model.efficient_parameters[version]
    x = torch.randn((4, 3, resolution, resolution)).to(device)
    num_params = sum([param.numel() for param in model.parameters() if param.requires_grad])

    print(f'Efficient Net Version: {version}')
    print(f'\tInput Shape: {x.shape}')
    print(f'\tOuput Shape: {model(x).shape}')  # (num_examples, num_classes)
    print(f'\tNumber of Parameters: {num_params}')
