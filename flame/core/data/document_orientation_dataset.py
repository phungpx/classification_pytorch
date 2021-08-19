import cv2
import torch
import random
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


class OrientationDataset(Dataset):
    def __init__(self, datadir, classes, image_pattern, image_size, inner_size, transforms=None, max_transforms=5, opencv_threads=4):
        super(OrientationDataset, self).__init__()
        cv2.setNumThreads(opencv_threads)
        datadir = Path(datadir)
        self.classes = classes
        self.image_size = image_size
        self.inner_size = inner_size
        self.transforms = transforms if transforms else []
        self.max_transforms = min(max_transforms, len(self.transforms))

        for class_ in classes:
            if not datadir.joinpath(class_).exists():
                raise FileNotFoundError(f'Folder {class_} does not exist.')

        self.image_paths = [(datadir.joinpath(class_).glob(image_pattern), class_) for class_ in classes]
        self.image_paths = [(image_path, class_) for path_gen, class_ in self.image_paths for image_path in path_gen]
        self.image_paths = [(image_path, i, class_) for image_path, class_ in self.image_paths for i in range(4)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, target, class_ = self.image_paths[idx]

        supclass_idx = self.classes[class_]

        sample = cv2.imread(str(image_path))
        sample = self._resize(sample, self.inner_size)

        for transform in random.sample(self.transforms, k=random.randint(0, self.max_transforms)):
            sample = transform(image=sample)

        sample = np.rot90(sample, k=target)
        sample = cv2.resize(sample, dsize=self.image_size)
        sample = np.ascontiguousarray(sample)
        sample = torch.from_numpy(sample)
        sample = sample.permute(2, 0, 1).to(torch.float)
        sample = (sample - sample.mean()) / sample.std()
        return sample, target, supclass_idx, str(image_path)

    def _resize(self, image, size):
        ratio = size / min(image.shape[:2])
        image = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
        return image
