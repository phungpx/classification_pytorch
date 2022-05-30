import cv2
import torch
import random
import numpy as np

from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset


class DocumentDirection(Dataset):
    def __init__(
        self,
        data_dirs: List[str],
        classes: Dict[str, int],
        image_patterns: List[str],
        image_size: Tuple[int, int],
        inner_size: int,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        transforms: Optional[list] = None,
        max_transforms: int = 5,
        opencv_threads: int = 4
    ):
        super(DocumentDirection, self).__init__()
        cv2.setNumThreads(opencv_threads)
        self.classes = classes
        self.image_size = image_size
        self.inner_size = inner_size
        self.transforms = transforms if transforms else []
        self.max_transforms = min(max_transforms, len(self.transforms))

        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1) if std is not None else None

        for data_dir in data_dirs:
            for class_name in classes:
                if not Path(data_dir).joinpath(class_name).exists():
                    raise FileNotFoundError(f'Folder {class_name} does not exist.')

        self.image_paths: List[Tuple[List[Path], str]] = []
        for class_name in classes:
            class_image_paths = []
            for data_dir in data_dirs:
                for image_pattern in image_patterns:
                    class_image_paths.extend(list(Path(data_dir).joinpath(class_name).glob(image_pattern)))

            self.image_paths.append((class_image_paths, class_name))

        self.image_paths = [(image_path, class_name) for _image_paths, class_name in self.image_paths for image_path in _image_paths]
        self.image_paths = [(image_path, i, class_name) for image_path, class_name in self.image_paths for i in range(4)]

        print(f"{', '.join([Path(data_dir).stem for data_dir in data_dirs])} - {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, target, class_name = self.image_paths[idx]

        supclass_idx = self.classes[class_name]

        sample = cv2.imread(str(image_path))
        sample = self._resize(sample, self.inner_size)

        if (self.mean is not None) and (self.std is not None):
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        for transform in random.sample(self.transforms, k=random.randint(0, self.max_transforms)):
            sample = transform(image=sample)

        sample = np.rot90(sample, k=target)
        sample = cv2.resize(sample, dsize=self.image_size)
        sample = np.ascontiguousarray(sample)
        sample = torch.from_numpy(sample)
        sample = sample.permute(2, 0, 1).contiguous().float()

        if (self.mean is not None) and (self.std is not None):
            sample = (sample.div(255.) - self.mean) / self.std
        else:
            sample = (sample - sample.mean()) / sample.std()

        return sample, target, supclass_idx, str(image_path)

    def _resize(self, image, size):
        ratio = size / min(image.shape[:2])
        image = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)

        return image
