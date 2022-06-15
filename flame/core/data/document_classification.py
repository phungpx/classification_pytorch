import cv2
import torch
import random
import numpy as np

from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from torch.utils.data import Dataset


class DocumentClassification(Dataset):
    def __init__(
        self,
        datadirs: List[str],
        classes: Dict[str, int],
        image_patterns: List[str],
        image_size: Tuple[int, int],
        inner_size: int,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        required_transforms: Optional[list] = None,
        optional_transforms: Optional[list] = None,
        max_transforms: int = 5,
        opencv_threads: int = 4
    ):
        super(DocumentClassification, self).__init__()
        cv2.setNumThreads(opencv_threads)
        self.classes = classes
        self.image_size = image_size
        self.inner_size = inner_size
        self.required_transforms = required_transforms if required_transforms else []
        self.optional_transforms = optional_transforms if optional_transforms else []
        self.max_transforms = min(max_transforms, len(self.optional_transforms))

        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1) if std is not None else None

        data_stats: Dict[str, list] = defaultdict(list)
        for datadir in datadirs:
            for class_name in classes:
                class_dir = Path(datadir).joinpath(class_name)
                if not class_dir.exists():
                    # print(f"Warning: {str(class_dir)} doesn't exist.")
                    continue

                for image_pattern in image_patterns:
                    data_stats[class_name].extend(class_dir.glob(image_pattern))

        self.image_paths: List[Tuple[Path, str]] = []  # List[Tuple[image_path, class_name]]
        for class_name, class_image_paths in data_stats.items():
            self.image_paths.extend([(image_path, class_name) for image_path in class_image_paths])

        print('--' * 20)
        print(f'[___Dataset Info___]: {Path(datadirs[0]).stem} set - {len(self.image_paths)} files.')
        for class_name, image_paths in data_stats.items():
            print(f'\t {class_name}: {len(image_paths)} images.')
        print('--' * 20)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, class_name = self.image_paths[idx]

        sample = cv2.imread(str(image_path))
        sample = self._resize(sample, self.inner_size)

        if (self.mean is not None) and (self.std is not None):
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        for transform in random.sample(self.required_transforms, k=len(self.required_transforms)):
            sample = transform(image=sample)
        for transform in random.sample(self.optional_transforms, k=random.randint(0, self.max_transforms)):
            sample = transform(image=sample)

        sample = cv2.resize(sample, dsize=self.image_size)
        sample = np.ascontiguousarray(sample)
        sample = torch.from_numpy(sample)
        sample = sample.permute(2, 0, 1).to(torch.float)

        if (self.mean is not None) and (self.std is not None):
            sample = (sample.div(255.) - self.mean) / self.std
        else:
            sample = (sample - sample.mean()) / sample.std() if not (sample == sample.mean()).all() else torch.zeros_like(sample)

        target = self.classes[class_name]

        return sample, target, str(image_path)

    def _resize(self, image, size):
        ratio = size / min(image.shape[:2])
        image = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)

        return image
