import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Callable, List, Tuple


class TextRotationDataset(Dataset):
    def __init__(
        self,
        datadirs: List[str] = None,
        image_pattern: str = '*.jpg',
        image_height: int = 32,  # fix heigh of text line image, width = image_ratio * image_height
        image_ratio: int = 20,  # ensure ratio of image (w / h)
        inner_size: int = 48,  # fixed dimension for min dimension of image
        transforms: Optional[List[Callable]] = None,
        max_transform: int = 5,
    ):
        super(TextRotationDataset, self).__init__()
        self.inner_size = inner_size
        self.image_ratio = image_ratio
        self.image_height = image_height
        self.transforms = transforms if transforms else []
        self.max_transform = min(max_transform, len(self.transforms))

        image_paths = []
        [image_paths.extend(Path(datadir).glob(image_pattern)) for datadir in datadirs]

        # each text image at 0 degree, we synthesize with 180 degree image.
        self.image_paths = [(image_path, i) for image_path in image_paths for i in range(2)]

        # after cropping or padding for ratio of text image, scaling text image to fixed height size.
        self.image_size = (int(round(image_height * image_ratio)), image_height)

        # ensure ratio (w / h) for text line, crop width of too long text line.
        self.crop_to_ratio = iaa.CropToAspectRatio(image_ratio, position='right-bottom')

        # ensure ratio (w / h) for text line, padding for too short text line.
        self.pad_to_ratio = iaa.PadToAspectRatio(image_ratio, position='right-bottom')

        print(f'- {Path(datadirs[0]).parent.stem}: {len(self.image_paths)}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, target = self.image_paths[idx]

        image = cv2.imread(str(image_path))
        image = self.resize(image)  # narrow down image size

        for transform in random.sample(self.transforms, k=random.randint(0, self.max_transform)):
            image = transform(image=image)

        image = np.rot90(image, k=target * 2)  # rotate image 0 or 180 degree
        sample = image.astype(np.float)

        # crop appropriate ratio and then normalize image to mean=0 and std=1
        if sample.shape[1] / sample.shape[0] > self.image_ratio:
            sample = self.crop_to_ratio(image=sample)
            sample = (sample - sample.mean()) / sample.std() if sample.std() else np.zeros_like(sample)
        # normalize image to mean=0, std=1 and then pad to appropriate ratio
        else:
            sample = (sample - sample.mean()) / sample.std() if sample.std() else np.zeros_like(sample)
            sample = self.pad_to_ratio(image=sample)

        # resize to fixed size and then convert to tensor (C x H x W)
        sample = cv2.resize(sample, dsize=self.image_size)
        sample = np.ascontiguousarray(sample)
        sample = torch.from_numpy(sample).float()
        sample = sample.permute(2, 0, 1).contiguous()

        target = torch.tensor([target], dtype=torch.float)

        return sample, target, str(image_path)

    def resize(self, image):
        ratio = self.inner_size / min(image.shape[:2])
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

        return image
