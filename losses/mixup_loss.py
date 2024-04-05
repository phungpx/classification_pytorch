import torch
import torch.nn as nn

from . import loss


class MixupLoss(loss.LossBase):
    def __init__(
        self,
        criterion: Union[nn.CrossEntropyLoss, nn.BCEWithLogitsLoss] = None,
        output_transform: Callable = lambda x: x
    ) -> None:
        super(MixupLoss, self).__init__(output_transform)
        self.criterion = criterion

    def forward(
        self,
        pred: torch.Tensor,
        target1: torch.Tensor,
        target2: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        '''
        Args:
            pred: floatTensor B x N, output from ConvNet
            target1: Tensor.Int64 B
            target2: Tensor.Int64 B
            alpha: Tensor.Float32, beta distribution
        Output:
            loss: Tensor.Float32, scalar for backward.
        '''
        assert alpha.dtype == torch.float32, f'alpha dtype must be torch.float32, got: {alpha.dtype}'
        assert target1.dtype == torch.int64, f'labels dtype must be torch.int64, got: {target1.dtype}'
        assert target2.dtype == torch.int64, f'labels dtype must be torch.int64, got: {target2.dtype}'
        assert pred.shape[0] == target1.shape[0], f'expected target batch {target1.shape[1]} to match target batch {pred.shape[0]}'
        assert pred.shape[0] == target2.shape[0], f'expected target batch {target2.shape[1]} to match target batch {pred.shape[0]}'

        loss = alpha * self.criterion(pred, target1) + (1 - alpha) * self.criterion(pred, target2)

        return loss
