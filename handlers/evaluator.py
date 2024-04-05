from abc import ABC, abstractmethod
from typing import Callable, Dict, Any


__all__ = ["MetricBase", "Metrics"]


class MetricBase(ABC):
    def __init__(self, output_transform: Callable = lambda x: x):
        super(MetricBase, self).__init__()
        self.output_transform = output_transform

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self, output: Any) -> Any:
        pass

    @abstractmethod
    def compute(self) -> Any:
        pass

    def started(self) -> None:
        self.reset()

    def iteration_completed(self, output: Any) -> Any:
        output = self.output_transform(output)
        output = self.update(output)
        return output

    def epoch_completed(self) -> Any:
        return self.compute()


class Metrics:
    def __init__(self, metrics: Dict[str, MetricBase]) -> None:
        super(Metrics, self).__init__()
        self.metrics = metrics

    def started(self, evaluator_name: str) -> None:
        self.evaluator_metric = dict()
        for metric_name, metric_fn in self.metrics.items():
            metric_fn.started()
            self.evaluator_metric[f"{evaluator_name}_{metric_name}"] = metric_fn

    def iteration_completed(self, output: Any) -> Dict[str, Any]:
        iteration_metric = dict()
        for metric_name, metric_fn in self.evaluator_metric.items():
            iteration_metric[metric_name] = metric_fn.iteration_completed(output=output)

        return iteration_metric

    def epoch_completed(self) -> Dict[str, Any]:
        epoch_metric = dict()
        for metric_name, metric_fn in self.evaluator_metric.items():
            epoch_metric[metric_name] = metric_fn.epoch_completed()

        return epoch_metric
