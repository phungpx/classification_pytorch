import torch
from typing import Callable

from .loss import LossBase


class WeightedSupclassesCrossEntropy(LossBase):
    def __init__(self, weight: torch.Tensor, smooth: float = 1e-6, output_transform: Callable = lambda x: x):
        super(WeightedSupclassesCrossEntropy, self).__init__(output_transform)
        self.weight = weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor, supclass_idx: torch.Tensor) -> torch.Tensor:
        pred: torch.Tensor = pred.softmax(dim=1)  # prob: batch_size x num_classes
        target: torch.Tensor = torch.zeros_like(pred).scatter(dim=1, index=target.unsqueeze(1), value=1)  # one-hot: batch_size x num_classes
        # cross entropy loss
        loss = - target * torch.log(pred + self.smooth)  # batch_size x num_classes
        # weight for each sample in batch
        weight: torch.Tensor = self.weight.gather(dim=0, index=supclass_idx).unsqueeze(dim=1)  # batch_size x 1
        weight: torch.Tensor = weight / weight.sum()
        # final loss
        loss = torch.sum(weight * loss)

        return loss
