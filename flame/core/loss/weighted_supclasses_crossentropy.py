import torch

from .loss import LossBase


class WeightedSupclassesCrossEntropy(LossBase):
    def __init__(self, weight, smooth=1e-6, output_transform=lambda x: x):
        super(WeightedSupclassesCrossEntropy, self).__init__(output_transform)
        self.weight = weight
        self.smooth = smooth

    def forward(self, pred, target, supclass_idx):
        weight = self.weight.gather(dim=0, index=supclass_idx).unsqueeze(1)
        output = torch.nn.functional.softmax(pred, dim=1)
        output = torch.log(output + self.smooth)
        target = torch.zeros_like(output).scatter(dim=1, index=target.unsqueeze(1), value=1)
        return - torch.sum(weight * output * target) / weight.sum()
