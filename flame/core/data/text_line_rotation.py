import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from typing import Optional, Callable, List, Tuple
from torch.utils.data import Dataset


class TextLineRotationDataset(Dataset):
    def __init__(self, datadir: Optional[str] = None,
                 image_pattern: str = '*.jpg',
                 image_height: int = 32,
                 image_ratio: int = 20,
                 inner_size: int = 48,
                 transforms: Optional[List[Callable]] = None,
                 max_transform: int = 5,
                 dataset_len: Optional[int] = None):
        super(TextLineRotationDataset, self).__init__()
        self.inner_size = inner_size  # fixed dimension for min dimension of image, and then scale remain dimension with ratio inner_size / min(image.shape[:2])
        self.image_ratio = image_ratio  # ensure ratio of image (w / h)
        self.image_height = image_height  # fix heigh of text line image, width = image_ratio * image_height
        self.transforms = transforms if transforms else []
        self.max_transform = min(max_transform, len(self.transforms))

        self.image_paths = Path(datadir).glob(image_pattern)
        self.image_paths = [(image_path, i) for image_path in self.image_paths for i in range(2)]  # each text image at 0 degree, we synthesize with 180 degree image.
        self.dataset_len = dataset_len

        self.image_size = (int(round(image_height * image_ratio)), image_height)  # after cropping or padding for ratio of text image, scaling text image to fixed height size.
        self.ratio_crop = iaa.CropToAspectRatio(image_ratio, position='right-bottom')  # ensure ratio (w / h) for text line, crop width of too long text line.
        self.ratio_pad = iaa.PadToAspectRatio(image_ratio, position='right-bottom')  # ensure ratio (w / h) for text line, padding for too short text line.

    def __len__(self):
        if self.dataset_len is None:
            return len(self.image_paths)
        return self.dataset_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_path, target = self.image_paths[idx]

        image = cv2.imread(str(image_path))
        image = self.resize(image)

        for transform in random.sample(self.transforms, k=random.randint(0, self.max_transform)):
            image = transform(image=image)

        image = np.rot90(image, k=target * 2)  # rotate image 0 or 180 degree
        image = image.astype(np.float)

        if image.shape[1] / image.shape[0] > self.image_ratio:
            image = self.ratio_crop(image=image)
            image = (image - image.mean()) / image.std() if image.std() else np.zeros_like(image)
        else:
            image = (image - image.mean()) / image.std() if image.std() else np.zeros_like(image)
            image = self.ratio_pad(image=image)

        # image = cv2.resize(image, dsize=(self.image_height * self.image_ratio, self.image_height))
        image = cv2.resize(image, dsize=self.image_size)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).to(torch.float)

        target = torch.tensor([target], dtype=torch.float)

        return image, target, str(image_path)

    def resize(self, image):
        ratio = self.inner_size / min(image.shape[:2])
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

        return image
