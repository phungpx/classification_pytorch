from typing import Callable
import torch

from . import loss


class FocalLoss(loss.LossBase):
    '''Focal Loss'''
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = 'mean',
        output_transform: Callable = lambda x: x
    ) -> None:
        super(FocalLoss, self).__init__(output_transform)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert target.dtype == torch.int64, f'labels dtype must be torch.int64, got: {target.dtype}'
        assert pred.shape[0] == target.shape[0], f'expected target batch {target.shape[1]} to match target batch {pred.shape[0]}'

        pred: torch.Tensor = pred.softmax(dim=1)
        target: torch.Tensor = torch.zero_like(pred).scatter(dim=1, index=target.unsqueeze(dim=1), value=1)  # one-hot

        focal: torch.Tensor = -self.alpha * (1. - pred).pow(self.gamma) * torch.log(pred)
        loss: torch.Tensor = torch.sum(target * focal, dim=1)

        if self.reduction == 'mean':
            loss: torch.Tensor = loss.mean()
        elif self.reduction == 'sum':
            loss: torch.Tensor = loss.sum()
        elif self.reduction == 'none':
            loss: torch.Tensor = loss
        else:
            raise NotImplementedError(f'invalid reduction mode: {self.reduction}')

        return loss
