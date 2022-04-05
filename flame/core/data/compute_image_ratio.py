import cv2
import json
import sympy
import argparse
import numpy as np
from typing import List, Dict
from pathlib import Path


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


def get_text_ratio_stats(dir_name, field_name='Serial'):
    json_paths = list(Path(dir_name).glob('**/*.json'))
    print(f'number of file: {len(json_paths)}')

    image_ratios = []
    for json_path in json_paths:
        with json_path.open(mode='r', encoding='utf8') as f:
            data = json.load(f)

        for shape in data['shapes']:
            if shape['label'] == field_name:
                if shape['shape_type'] == 'rectangle':
                    points = shape['points']
                    width = points[1][0] - points[0][0] + 1
                    height = points[1][1] - points[0][1] + 1
                    image_ratios.append(width / height)
                elif shape['shape_type'] == 'polygon':
                    points = [sympy.Point2D(*point) for point in shape['points']]
                    width = (float(points[0].distance(points[1])) + float(points[2].distance(points[3]))) / 2
                    height = (float(points[0].distance(points[3])) + float(points[1].distance(points[2]))) / 2
                    image_ratios.append(width / height)

    ratio_stats = dict()
    ratio_stats['min'] = np.min(image_ratios) if len(image_ratios) else 0.
    ratio_stats['max'] = np.max(image_ratios) if len(image_ratios) else 0.
    ratio_stats['std'] = np.std(image_ratios) if len(image_ratios) else 0.
    ratio_stats['mean'] = np.mean(image_ratios) if len(image_ratios) else 0.

    image_ratio = int(float(ratio_stats['mean'] + ratio_stats['std']))

    return ratio_stats, image_ratio


def extract_text_line(dir_name, save_dir, field_name='V_SERIAL'):
    json_paths = list(Path(dir_name).glob('**/*.json'))
    print(f'number of file: {len(json_paths)}')

    for json_path in json_paths:
        _save_dir = Path(save_dir).joinpath(json_path.parent.stem)
        if not _save_dir.exists():
            _save_dir.mkdir(parents=True)

        with json_path.open(mode='r', encoding='utf8') as f:
            data = json.load(f)

        image_path = json_path.with_name(Path(data['imagePath']).name)
        image = cv2.imread(str(image_path))

        for shape in data['shapes']:
            if shape['label'] == field_name:
                if shape['shape_type'] == 'rectangle':
                    points = shape['points']
                    (x1, y1), (x2, y2) = points
                    text = image[y1:y2, x1:x2]
                elif shape['shape_type'] == 'polygon':
                    points = [sympy.Point2D(*point) for point in shape['points']]
                    width = (float(points[0].distance(points[1])) + float(points[2].distance(points[3]))) / 2
                    height = (float(points[0].distance(points[3])) + float(points[1].distance(points[2]))) / 2
                    image_ratios.append(width / height)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-name', help='training directories.')
    parser.add_argument('--field-name', type=str)
    parser.add_argument('--pattern', help='glob pattern if image_path is a dir.')
    args = parser.parse_args()

    # ratio_stats, image_ratio = get_image_ratio_stats(args.train_dirs, args.pattern)
    ratio_stats, image_ratio = get_text_ratio_stats(args.dir_name, args.field_name)
    print(json.dumps(ratio_stats, ensure_ascii=False, indent=4))
    print(f'__image_ratio__ = {image_ratio}')
