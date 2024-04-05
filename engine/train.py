import os
import time
from enum import Enum
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Evaluator(Enum):
    TRAIN = "train"
    TRAIN_EVAL = "train_eval"
    VALID = "valid"
    TEST = "test"


class Trainer:
    def __init__(
        self,
        project_name: str,
        model: nn.Module,
        data: Dict[str, Callable],
        loss: nn.Module,
        optim: nn.Module,
        metric: Callable,
        lr_scheduler: nn.Module,
        early_stopping: Callable,
        logger: Callable,
        writer: Callable,
        plotter: Callable,
        save_dir: str,
        model_inspection: Callable = None,
    ):
        super(Trainer, self).__init__()
        self.logger = logger.get_logger(log_name=project_name)
        self.model = model
        self.data = data
        self.loss = loss
        self.optim = optim
        self.metric = metric
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping

        # Logger and Tensorboard
        self.writer = writer
        self.plotter = plotter

        # get model info
        if model_inspection is not None:
            model_inspection(self.model, self.logger)

        # Save Directory for Checkpoint and Backup
        self.save_dir = Path(save_dir) / datetime.now().strftime(r"%y%m%d%H%M")
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

        # training information
        self.iteration_counters: Dict[str, int] = defaultdict(int)
        self.training_info: Dict[str, Any] = dict()

    def train_epoch(
        self, evaluator_name: str, dataloader: nn.Module
    ) -> Dict[str, float]:
        self.model.train()
        self.metric.started(evaluator_name)
        for batch in tqdm(dataloader, total=len(dataloader)):
            self.optim.zero_grad()
            params = [
                param.to(self.device) if torch.is_tensor(param) else param
                for param in batch
            ]
            params[0] = self.model(params[0])
            loss = self.loss(*params)
            loss.backward()
            self.optim.step()

            # log learning_rate
            self.writer.add_scalar(
                name="learning_rate",
                value=self.optim.param_groups[0]["lr"],
                step=self.iteration_counters[evaluator_name],
            )

            iteration_metric = self.metric.iteration_completed(output=params)

            for metric_name, metric_value in iteration_metric.items():
                self.writer.add_scalar(
                    name=metric_name,
                    value=metric_value,
                    step=self.iteration_counters[evaluator_name],
                )

            self.iteration_counters[evaluator_name] += 1

        return self.metric.epoch_completed()

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

                iteration_metric = self.metric.iteration_completed(output=params)

                for metric_name, metric_value in iteration_metric.items():
                    self.writer.add_scalar(
                        name=metric_name,
                        value=metric_value,
                        step=self.iteration_counters[evaluator_name],
                    )

                self.iteration_counters[evaluator_name] += 1

        return self.metric.epoch_completed()

    def verbose(self, message: str, _print: bool = True) -> None:
        self.logger.info(message)
        if _print:
            print(message)

    def verbose_metric(self, metric: Dict[str, Any], _print: bool = True) -> None:
        messages = []
        for metric_name, metric_value in metric.items():
            if isinstance(metric_value, float):
                messages.append(f"{metric_name}: {metric_value:.5f}")
                # save metric value to plotter
                self.plotter.add_scalar(metric_name, metric_value)

        message = " - ".join(messages)
        self.verbose(message=f"{message}", _print=_print)

    def test(self) -> None:
        self.verbose(message=f"\n** TESTING **")
        # Load weight
        checkpoint_path = self.training_info["checkpoint_path"]
        state_dict = torch.load(f=checkpoint_path, map_location=self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state_dict=state_dict)
        else:
            self.model.load_state_dict(state_dict=state_dict)

        # Start to evaluate
        self.verbose(message=f"{time.asctime()} - STARTED")
        metrics = self.eval_epoch(
            evaluator_name=Evaluator.TEST.value,
            dataloader=self.data[Evaluator.TEST.value],
        )
        messages = [
            f"{metric_name}: {metric_value:.5f}"
            for metric_name, metric_value in metrics.items()
        ]
        self.verbose(message=f"[Info] {' - '.join(messages)}")
        self.verbose(message=f"{time.asctime()} - COMPLETED")

    def train(
        self,
        num_epochs: int,
        resume_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        gpu_indices: List[int] = [],
    ) -> None:
        # Set Device for Model: prepare for (multi-device) GPU training
        if gpu_indices is None:
            gpu_indices = []

        self.device = torch.device(
            f"cuda:{min(gpu_indices)}" if len(gpu_indices) > 0 else "cpu"
        )

        self.verbose(message=f"** TRAINING - DEVICES: {self.device} {gpu_indices} **")

        # Load pretrained weight
        if checkpoint_path is not None:
            state_dict = torch.load(f=checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict=state_dict)

        # Resume Mode
        if resume_path is not None:
            self.verbose(message=f"{time.asctime()} - RESUME")
            checkpoint = torch.load(f=resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optim.load_state_dict(checkpoint["optim"])
            start_epoch = checkpoint["epoch"] + 1
            best_score = checkpoint["best_score"]
            score_name = checkpoint["score_name"]
            mode = self.early_stopping.mode
        else:
            start_epoch = 0
            mode = self.early_stopping.mode  # TODO: how to get `mode`
            score_name = self.early_stopping.score_name  # TODO: how to get `score name`
            best_score = -np.Inf if mode == "min" else 0

        # Multi GPUs
        self.model = self.model.to(self.device)
        if len(gpu_indices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_indices)

        # Initialize checkpoint path for saving checkpoint
        _checkpoint_path = (
            self.save_dir / f"best_model_{start_epoch}_{score_name}_{best_score}.pth"
        )

        # Start to train
        self.verbose(message=f"{time.asctime()} - STARTED")
        for epoch in range(start_epoch, num_epochs + start_epoch):
            self.verbose(message=f"\nEpoch #{epoch} - {time.asctime()}")
            train_metrics = self.train_epoch(
                evaluator_name=Evaluator.TRAIN.value,
                dataloader=self.data[Evaluator.TRAIN.value],
            )
            train_eval_metrics = self.eval_epoch(
                evaluator_name=Evaluator.TRAIN_EVAL.value,
                dataloader=self.data[Evaluator.TRAIN_EVAL.value],
            )
            valid_metrics = self.eval_epoch(
                evaluator_name=Evaluator.VALID.value,
                dataloader=self.data[Evaluator.VALID.value],
            )

            # update learning scheduler
            self.lr_scheduler._step(valid_metrics)

            # update early stopping
            self.early_stopping(valid_metrics)
            # check early stopping flag
            if self.early_stopping.early_stop:
                self.verbose(message=f"{time.asctime()} - EARLY STOPPING.")
                break

            # export training information
            self.verbose_metric(train_metrics)
            self.verbose_metric(train_eval_metrics)
            self.verbose_metric(valid_metrics)

            # save backup checkpoint for resume mode
            if self.save_dir.joinpath(f"backup_epoch_{epoch - 1}.pth").exists():
                os.remove(str(self.save_dir.joinpath(f"backup_epoch_{epoch - 1}.pth")))

            if isinstance(self.model, torch.nn.DataParallel):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()

            backup_checkpoint = {
                "epoch": epoch,
                "best_score": best_score,
                "score_name": score_name,
                "model": model_state_dict,
                "optim": self.optim.state_dict(),
            }

            backup_checkpoint_path = self.save_dir / f"backup_epoch_{epoch}.pth"
            torch.save(obj=backup_checkpoint, f=str(backup_checkpoint_path))
            self.verbose(
                message=f"__Saving Backup Checkpoint__ {str(backup_checkpoint_path)}",
                _print=False,
            )

            score = (
                -valid_metrics[f"valid_{score_name}"]
                if mode == "min"
                else valid_metrics[f"valid_{score_name}"]
            )

            if score > best_score:
                best_score = score
                if _checkpoint_path.exists():
                    os.remove(str(_checkpoint_path))
                _checkpoint_path = (
                    self.save_dir / f"best_model_{epoch}_{score_name}_{best_score}.pth"
                )
                torch.save(obj=model_state_dict, f=str(_checkpoint_path))
                self.verbose(
                    message=f"__Saving Checkpoint__ {str(_checkpoint_path)}",
                    _print=False,
                )

                self.training_info["checkpoint_path"] = str(_checkpoint_path)
                self.training_info["best_score"] = best_score
                self.training_info["epoch"] = epoch

            self.plotter.draw()

        self.training_info["score_name"] = score_name
        self.training_info["backup_checkpoint_path"] = str(backup_checkpoint_path)

        self.verbose(message=f"{time.asctime()} - COMPLETED")

    def __call__(
        self,
        num_epochs: int = 1,
        resume_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        gpu_indices: List[int] = [],
    ) -> None:
        self.train(num_epochs, resume_path, checkpoint_path, gpu_indices)
        if self.data.get("test", None) is not None:
            self.test()
