from pathlib import Path
from typing import Any, Callable, Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from handlers.evaluator import MetricBase

__all__ = ["Accuracy", "ConfusionMatrix"]


def stats(preds, targets, num_classes):
    """Compute TP, FP, FN, TN.
    Args:
        preds: (tensor) model outputs sized [N,].
        targets: (tensor) labels targets sized [N,].
    Returns:
        (tensor): TP, FP, FN, TN, sized [num_classes,].
    """
    tp = torch.empty(num_classes)
    fp = torch.empty(num_classes)
    fn = torch.empty(num_classes)
    tn = torch.empty(num_classes)
    for i in range(num_classes):
        tp[i] = ((preds == i) & (targets == i)).sum().item()
        fp[i] = ((preds == i) & (targets != i)).sum().item()
        fn[i] = ((preds != i) & (targets == i)).sum().item()
        tn[i] = ((preds != i) & (targets != i)).sum().item()

    return tp, fp, fn, tn


class Metric(MetricBase):
    def __init__(self, metric_fn: Any, output_transform: Callable = lambda x: x):
        super(Metric, self).__init__(output_transform)
        self.metric_fn = metric_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output: Any) -> float:
        value = self.metric_fn(*output)

        if len(value.shape) != 0:
            raise ValueError("metric_fn did not return the average accuracy.")

        N = output[0].shape[0]
        self._sum += value.item() * N
        self._num_examples += N

        return value.item()

    def compute(self):
        if self._num_examples == 0:
            raise ValueError(
                "Loss must have at least one example before it can be computed."
            )
        return self._sum / self._num_examples


class Accuracy(nn.Module):
    def __init__(self, num_classes: int):
        super(Accuracy, self).__init__()
        self.num_classes = num_classes

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(preds, dim=1)
        correct = (preds == targets).sum()
        return torch.true_divide(correct, targets.shape[0])


class F1Score(nn.Module):
    def __init__(self, num_classes):
        super(F1Score, self).__init__()
        self.num_classes = num_classes

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(preds, dim=1)
        tp, fp, fn, _ = stats(preds, targets, self.num_classes)
        precision = torch.true_divide(tp.sum(), (tp.sum().item() + fp.sum().item()))
        recall = torch.true_divide(tp.sum(), (tp.sum().item() + fn.sum().item()))
        return torch.true_divide(
            2 * (precision * recall), (precision.item() + recall.item())
        )


class ConfusionMatrix(MetricBase):
    def __init__(
        self,
        save_dir: str,
        classes: Union[int, Dict[int, str]],
        output_transform: Callable = lambda x: x,
    ):
        super(ConfusionMatrix, self).__init__(output_transform)
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.num_classes = len(classes) if isinstance(classes, dict) else classes
        self.class_names = (
            list(classes.values())
            if isinstance(classes, dict)
            else [str(class_id) for class_id in range(classes)]
        )

    def reset(self):
        self.confusion_matrix = torch.zeros(
            size=(self.num_classes, self.num_classes), dtype=torch.int16
        )

    def update(self, output: Any) -> None:
        preds, targets = output
        preds = torch.argmax(preds, dim=1)
        for target, pred in zip(targets.view(-1), preds.view(-1)):
            self.confusion_matrix[target.long(), pred.long()] += 1

    def compute(self):
        plt.figure(figsize=(15, 10))
        df_cm = pd.DataFrame(
            self.confusion_matrix, index=self.class_names, columns=self.class_names
        ).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(
            heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=15
        )
        heatmap.xaxis.set_ticklabels(
            heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=15
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(str(self.save_dir.joinpath("confusion_matrix.png")))
        print(f"ConfusionMatrix: {self.save_dir}")

        return self.confusion_matrix


class Precision(MetricBase):
    def __init__(self, num_classes: int, output_transform: Callable = lambda x: x):
        super(Precision, self).__init__(output_transform)
        self.num_classes = num_classes

    def reset(self) -> None:
        self.true_pos = torch.zeros((self.num_classes))
        self.false_pos = torch.zeros((self.num_classes))

    def update(self, output: Any) -> None:
        preds, targets = output
        preds = torch.argmax(preds, dim=1)
        tp, fp, _, _ = stats(preds, targets, self.num_classes)
        for i in range(self.num_classes):
            self.true_pos[i] += tp[i]
            self.false_pos[i] += fp[i]

    def compute(self):
        eps = 1e-8
        precision = []
        for i in range(self.num_classes):
            precision.append(
                self.true_pos[i].sum().item()
                / (self.true_pos[i].sum().item() + self.false_pos[i].sum().item() + eps)
            )
        return sum(precision) / len(precision)


class Recall(MetricBase):
    def __init__(self, num_classes: int, output_transform: Callable = lambda x: x):
        super(Recall, self).__init__(output_transform)
        self.num_classes = num_classes

    def reset(self) -> None:
        self.true_pos = torch.zeros((self.num_classes))
        self.false_neg = torch.zeros((self.num_classes))

    def update(self, output: Any) -> None:
        preds, targets = output
        preds = torch.argmax(preds, dim=1)
        tp, _, fn, _ = stats(preds, targets, self.num_classes)
        for i in range(self.num_classes):
            self.true_pos[i] += tp[i]
            self.false_neg[i] += fn[i]

    def compute(self):
        recall = []
        eps = 1e-8
        for i in range(self.num_classes):
            recall.append(
                self.true_pos[i].sum().item()
                / (self.true_pos[i].sum().item() + self.false_neg[i].sum().item() + eps)
            )
        return sum(recall) / len(recall)
