import cv2
import json
import argparse
import numpy as np


def get_image_ratio_stats(train_dirs: List[str], image_pattern: str) -> Dict[str, float]:
    image_ratios = []

    for train_dir in train_dirs:
        for image_path in Path(train_dir).glob(image_pattern):
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            image_ratio = width / height
            image_ratios.append(image_ratio)

    ratio_stats = dict()
    ratio_stats['min'] = np.min(image_ratios) if len(image_ratios) else 0.
    ratio_stats['max'] = np.max(image_ratios) if len(image_ratios) else 0.
    ratio_stats['std'] = np.std(image_ratios) if len(image_ratios) else 0.
    ratio_stats['mean'] = np.mean(image_ratios) if len(image_ratios) else 0.

    image_ratio = int(float(ratio_stats['mean'] + ratio_stats['std']))

    return ratio_stats, image_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train-dirs', help='training directories.')
    parser.add_argument('--pattern', help='glob pattern if image_path is a dir.')
    args = parser.parse_args()

    ratio_stats, image_ratio = get_image_ratio_stats(args.train_dirs, args.pattern)
    print(json.dumps(ratio_stats, ensure_ascii=False, indent=4))
    print(f'__image_ratio__ = {image_ratio}')
