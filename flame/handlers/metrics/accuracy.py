import torch
from sklearn.metrics import f1_score
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class Accuracy(Metric):
    def __init__(self, is_multilabel=False, output_transform=lambda x: x):
        super(Accuracy, self).__init__(output_transform)
        if is_multilabel:
            self.accuracy_fn = self._is_multilabel
        else:
            self.accuracy_fn = self._is_multiclass

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        average_accuracy = self.accuracy_fn(*output)

        if len(average_accuracy.shape) != 0:
            raise ValueError('accuracy_fn did not return the average accuracy.')

        N = output[0].shape[0]
        self._sum += average_accuracy.item() * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples

    def _is_multilabel(self, preds, targets):
        # x = torch.where(preds * targets >= 1, torch.ones_like(targets), torch.zeros_like(targets)).sum(dim=1)
        # y = torch.where((preds + targets) >= 1, torch.ones_like(targets), torch.zeros_like(targets)).sum(dim=1)
        # accuracy = (1 / targets.shape[0]) * torch.sum(x / y)
        # return accuracy
        return f1_score(y_true=targets.to(torch.int).cpu().numpy(),
                        y_pred=preds.to(torch.int).cpu().numpy(),
                        average='samples')

    def _is_multiclass(self, preds, targets):
        indices = torch.argmax(preds, dim=1)
        correct = torch.eq(indices, targets).view(-1)
        accuracy = torch.sum(correct) / correct.shape[0]
        return accuracy
