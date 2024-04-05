import time
from typing import Callable, Dict, Optional, List

import torch
import torch.nn as nn
from tqdm import tqdm


class Evaluator(nn.Module):
    def __init__(
        self,
        data: nn.Module,
        model: nn.Module,
        metric: Callable,
    ):
        super(Evaluator, self).__init__()
        self.data = data
        self.model = model
        self.metric = metric

    def eval_epoch(
        self, evaluator_name: str, dataloader: nn.Module
    ) -> Dict[str, float]:
        self.model.eval()
        self.metric.started(evaluator_name)
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                params = [
                    param.to(self.device) if torch.is_tensor(param) else param
                    for param in batch
                ]
                params[0] = self.model(params[0])

                _ = self.metric.iteration_completed(output=params)

        return self.metric.epoch_completed()

    def __call__(
        self,
        checkpoint_path: Optional[str] = None,
        gpu_indices: List[int] = [],
    ):
        # Set Device for Model: prepare for (multi-device) GPU training
        if gpu_indices is None:
            gpu_indices = []

        self.device = torch.device(
            f"cuda:{min(gpu_indices)}" if len(gpu_indices) > 0 else "cpu"
        )

        # Load weight
        if checkpoint_path is None:
            raise ValueError("No checkpoint to load.")

        self.model = self.model.to(self.device)
        if len(gpu_indices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_indices)

        # Load weight
        state_dict = torch.load(f=checkpoint_path, map_location=self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state_dict=state_dict)
        else:
            self.model.load_state_dict(state_dict=state_dict)

        # Start to evaluate
        print(f"{time.asctime()} - STARTED")
        metrics = self.eval_epoch(evaluator_name="test", dataloader=self.data)
        messages = [
            f"\n* {metric_name}:\n{metric_value}\n"
            for metric_name, metric_value in metrics.items()
        ]
        print(f"[INFO] {''.join(messages)}")
        print(f"{time.asctime()} - COMPLETED")
