from typing import Dict, Any


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        evaluator_name: str = "valid",
        patience: int = 7,
        delta: float = 0,
        mode: str = "min",
        score_name: str = "loss",
    ) -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            mode (str): 'min' or 'max' - 'min' applied to loss and 'max' applied to accuracy, for example. Default: min
            score_name (str): 'loss' or 'accuracy' to take value in metrics
            evaluator_name (str): 'train' or 'train_eval' or 'valid' ... to get value.
        """
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.score_name = score_name
        self.evaluator_name = evaluator_name

        # initialize value
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metrics: Dict[str, Any]) -> None:
        score = metrics[f"{self.evaluator_name}_{self.score_name}"]
        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"Early Stopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
