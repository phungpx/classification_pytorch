import logging
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime


class Logger:
    def __init__(
        self,
        save_dir,
        mode=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ):
        self.log_dir = Path(save_dir) / datetime.now().strftime(r"%y%m%d%H%M") / "log"
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        self.mode = mode
        self.format = format

    def get_logger(self, log_name):
        logger = logging.getLogger(log_name)
        logger.setLevel(self.mode)
        handler = logging.FileHandler(str(self.log_dir.joinpath(f"{log_name}.log")))
        formatter = logging.Formatter(self.format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
