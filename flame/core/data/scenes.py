import cv2
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class SceneDataset(Dataset):
    def __init__(self, datadir, csv_path, image_extent, image_size, inner_size, transforms=None):
        super(SceneDataset, self).__init__()
        self.inner_size = inner_size
        self.image_size = image_size
        self.transforms = transforms if transforms is not None else []

        df = pd.read_csv(str(csv_path))
        labels = np.array(df.drop(['names'], axis=1)).tolist()
        paths = [str(Path(datadir).joinpath(str(image_name) + image_extent)) for image_name in list(df['names'])]

        self.data = [(path, label) for path, label in zip(paths, labels)]

        print(f'- {Path(datadir).parents[0].stem}: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        sample = cv2.imread(str(image_path))
        sample = self._resize(sample, self.inner_size)

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            sample = transform(image=sample)

        sample = cv2.resize(sample, dsize=self.image_size)
        sample = np.ascontiguousarray(sample)
        sample = torch.from_numpy(sample).permute(2, 0, 1).to(torch.float)

        if (sample == sample.mean()).all():
            sample = torch.zeros_like(sample)
        else:
            sample = (sample - sample.mean()) / sample.std()

        target = torch.from_numpy(np.asarray(label)).to(torch.float)

        return sample, target, str(image_path)

    def _resize(self, image, size):
        ratio = size / min(image.shape[:2])
        image = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
        return image
