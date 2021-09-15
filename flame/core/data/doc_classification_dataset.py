import cv2
import torch
import random
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


class DocClassificationDataset(Dataset):
    def __init__(self, datadir, classes, image_patterns, image_size, inner_size, required_transforms=None, optional_transforms=None, max_transforms=5, opencv_threads=4):
        super(DocClassificationDataset, self).__init__()
        cv2.setNumThreads(opencv_threads)
        datadir = Path(datadir)
        self.classes = classes
        self.image_size = image_size
        self.inner_size = inner_size
        self.required_transforms = required_transforms if required_transforms else []
        self.optional_transforms = optional_transforms if optional_transforms else []
        self.max_transforms = min(max_transforms, len(self.optional_transforms))

        for class_ in classes:
            if not datadir.joinpath(class_).exists():
                raise FileNotFoundError(f'Folder {class_} does not exist.')

        self.image_paths = [(datadir.joinpath(class_).glob(image_pattern), class_) for class_ in classes for image_pattern in image_patterns]
        self.image_paths = [(image_path, class_) for path_gen, class_ in self.image_paths for image_path in path_gen]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, class_ = self.image_paths[idx]

        sample = cv2.imread(str(image_path))
        sample = self._resize(sample, self.inner_size)

        for transform in random.sample(self.required_transforms, k=len(self.required_transforms)):
            sample = transform(image=sample)
        for transform in random.sample(self.optional_transforms, k=random.randint(0, self.max_transforms)):
            sample = transform(image=sample)

        sample = cv2.resize(sample, dsize=self.image_size)
        sample = np.ascontiguousarray(sample)
        sample = torch.from_numpy(sample)
        sample = sample.permute(2, 0, 1).to(torch.float)

        if (sample == sample.mean()).all():
            sample = torch.zeros_like(sample)
        else:
            sample = (sample - sample.mean()) / sample.std()

        target = self.classes[class_]
        return sample, target, str(image_path)

    def _resize(self, image, size):
        ratio = size / min(image.shape[:2])
        image = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
        return image
