import torch
import torchvision
import torch.nn as nn


class SpoofingNet(nn.Module):
    def __init__(
        self,
        num_classes,
        num_samples,
        mb_pretrained,
        mb_fixed,
        mb_out_channels,
        feat_out_channels,
        pool_out,
    ):
        super(SpoofingNet, self).__init__()

        self.MBConv = torchvision.models.mobilenet_v2(pretrained=mb_pretrained).features
        self.MBConv[-1] = torchvision.models.mobilenet.ConvBNReLU(
            in_planes=320, out_planes=mb_out_channels, kernel_size=1, stride=1
        )
        self.MBConv.requires_grad_(not mb_fixed)
        self.Avgpool = nn.AdaptiveAvgPool2d(output_size=pool_out)
        self.Conv1x1 = nn.Conv2d(
            in_channels=mb_out_channels * num_samples,
            out_channels=feat_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.Linear = nn.Linear(in_features=feat_out_channels, out_features=num_classes)

    def forward(self, x):
        """
        Input: x: List[Tensor], len(x)=num_samples, list of tensors which have different sizes
               s: Tensor, shape [N, C, H, W]
        Output: Tensor: [N, num_classes]
        """
        x = [self.MBConv(s) for s in x]
        x = [self.Avgpool(s) for s in x]

        x = torch.cat(x, dim=1)

        x = self.Conv1x1(x)
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = x.view(x.shape[0], -1)
        x = self.Linear(x)

        return x
