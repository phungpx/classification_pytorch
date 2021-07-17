import cv2
import torch
import random
import numpy as np
from pathlib import Path
from natsort import natsorted
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


class SpoofingCardDataset(Dataset):
    def __init__(self, dirname, classes, card_types, crop_ratios, image_sizes, image_patterns, label_patterns, transforms=None):
        super(SpoofingCardDataset, self).__init__()
        self.data_paths = []
        self.classes = classes
        self.dirname = Path(dirname)
        self.image_sizes = image_sizes
        self.crop_ratios = crop_ratios
        self.card_types = card_types
        self.transforms = transforms if transforms else []

        image_paths, label_paths = [], []
        for class_name in classes:
            for image_pattern in image_patterns:
                image_paths.extend(Path(dirname).joinpath(class_name).glob('**/{}'.format(image_pattern)))
            for label_pattern in label_patterns:
                label_paths.extend(Path(dirname).joinpath(class_name).glob('**/{}'.format(label_pattern)))

        image_paths = natsorted(image_paths, key=lambda x: str(x.stem))
        label_paths = natsorted(label_paths, key=lambda x: str(x.stem))

        self.data_pairs = [[image_path, label_path] for image_path, label_path in zip(image_paths, label_paths)]

        print(f'- {self.dirname.stem}: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def _crop_to_bbox(self, image, points, crop_ratios):
        h, w = image.shape[:2]
        bbox = np.array(points)
        bbox = [np.min(bbox[:, 0]), np.min(bbox[:, 1]), np.max(bbox[:, 0]), np.max(bbox[:, 1])]
        hb, wb = np.abs(bbox[3] - bbox[1]), np.abs(bbox[2] - bbox[0])

        samples = []
        for crop_ratio in crop_ratios:
            if crop_ratio == 'full':
                samples.append(image)
            else:
                x1 = max(bbox[0] - (wb * crop_ratio), 0)
                y1 = max(bbox[1] - (hb * crop_ratio), 0)
                x2 = min(bbox[2] + (wb * crop_ratio), w)
                y2 = min(bbox[3] + (hb * crop_ratio), h)
                samples.append(image[int(y1):int(y2), int(x1):int(x2)])

        return samples

    def _generate_training_data(self, image_path, xml_path, class_name):
        batch_sample, batch_target = [], []
        image = cv2.imread(str(image_path))
        root = ET.parse(str(xml_path)).getroot()

        for card_type in self.card_types:
            regions = root.findall('.//*[@name=\"{}\"]/../..'.format(card_type))
            for region in regions:
                points = region[0].get('points')
                points = [[int(float(coord)) for coord in point.split(',')] for point in points.split()]
                samples = self._crop_to_bbox(image, points, crop_ratios=self.crop_ratios)
                batch_sample.append(samples)
                batch_target.append(class_name)

        return batch_sample, batch_target

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]

        for mask_parent in list(label_path.parents)[::-1]:
            if mask_parent.parent == self.dirname:
                class_name = mask_parent.stem
                break

        batch_sample, batch_target = self._generate_training_data(image_path, label_path, class_name)

        if not len(batch_sample):
            print('image {} has no mask'.format(image_path))

        idx = np.random.choice(list(range(len(batch_sample))))
        samples, target = batch_sample[idx], batch_target[idx]
        samples = [cv2.resize(sample, dsize=(image_size, image_size)) for sample, image_size in zip(samples, self.image_sizes)]

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            samples = [transform(image=sample) for sample in samples]

        samples = [np.ascontiguousarray(sample) for sample in samples]
        samples = [torch.from_numpy(sample) for sample in samples]
        samples = [sample.permute(2, 0, 1).to(torch.float) for sample in samples]
        samples = [(sample - sample.mean()) / sample.std() for sample in samples]
        target = torch.tensor(self.classes[target])

        return samples, target
