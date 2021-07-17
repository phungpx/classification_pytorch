import cv2
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


imsize_map = {"b0": 224, "b1": 240, "b2": 260, "b3": 300, "b4": 380, "b5": 456, "b6": 528, "b7": 600}


class HymenopteraDataset(Dataset):
    def __init__(self, dirname, version, classes, mean, std, transforms=None):
        super(HymenopteraDataset, self).__init__()
        self.classes = classes
        self.dirname = Path(dirname)
        self.imsize = imsize_map[version]
        self.transforms = transforms if transforms else []
        self.std = torch.tensor(std, dtype=torch.float).reshape(1, 1, 3)
        self.mean = torch.tensor(mean, dtype=torch.float).reshape(1, 1, 3)

        self.image_paths = []
        for class_name in classes:
            self.image_paths.extend(Path(dirname).joinpath(class_name).glob('**/*.jpg'))

        print(f'- {self.dirname.stem}: {len(self.image_paths)}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_infos = [str(image_path), image.shape[1::-1]]

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image = transform(image=image)

        image = cv2.resize(image, dsize=(self.imsize, self.imsize))
        for image_parent in list(image_path.parents)[::-1]:
            if image_parent.parent == self.dirname:
                class_name = image_parent.stem
                break

        target = torch.tensor(self.classes[class_name])

        sample = np.ascontiguousarray(image)
        sample = torch.from_numpy(sample)
        sample = (sample.float().div(255.) - self.mean) / self.std
        sample = sample.permute(2, 0, 1).contiguous()

        return sample, target, image_infos
