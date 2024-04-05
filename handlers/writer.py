from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, save_dir: str):
        tb_dir = Path(save_dir) / datetime.now().strftime(r"%y%m%d%H%M") / "tensorboard"
        if not tb_dir.exists():
            tb_dir.mkdir(parents=True)

        self.writer = SummaryWriter(str(tb_dir))

    def add_scalar(self, name: str, value: float, step: int = 0) -> None:
        self.writer.add_scalar(name, value, global_step=step)

    def add_image(self, name: str, data: torch.Tensor, step: int = 0):
        img_data = torchvision.utils.make_grid(data)
        self.writer.add_image(name, img_data, global_step=step)

    def add_embedding(self, data: torch.Tensor, targets: torch.Tensor, step: int = 0):
        features = data.reshape(data.shape[0], -1)
        class_labels = [label.item() for label in targets]
        self.writer.add_embedding(
            features, metadata=class_labels, label_img=data, global_step=step
        )
