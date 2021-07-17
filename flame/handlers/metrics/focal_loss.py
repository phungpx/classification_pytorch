import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def _onehot_encode(self, labels: torch.Tensor, n_classes: int = 0,
                       device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
        sizes: torch.Size = labels.shape
        onehot_tensor: torch.Tensor = torch.zeros(size=(sizes[0], n_classes, *sizes[1:]), dtype=dtype, device=device)
        onehot_tensor: torch.Tensor = onehot_tensor.scatter_(1, labels.unsqueeze(dim=1), 1.)

        return onehot_tensor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert target.dtype == torch.int64, f'labels dtype must be torch.int64, got: {target.dtype}'
        assert pred.shape[0] == target.shape[0], f'expected target batch {target.shape[1]} to match target batch {pred.shape[0]}'

        pred: torch.Tensor = nn.Softmax(dim=1)(pred)
        target: torch.Tensor = self._onehot_encode(labels=target, n_classes=pred.shape[1],
                                                   device=pred.device, dtype=pred.dtype)

        loss: torch.Tensor = -self.alpha * (1. - pred).pow(self.gamma) * torch.log(pred)
        loss: torch.Tensor = torch.sum(target * loss, dim=1)

        if self.reduction == 'mean':
            loss: torch.Tensor = loss.mean()
        elif self.reduction == 'sum':
            loss: torch.Tensor = loss.sum()
        elif self.reduction == 'none':
            loss: torch.Tensor = loss
        else:
            raise NotImplementedError(f"invalid reduction mode: {self.reduction}")

        return loss
