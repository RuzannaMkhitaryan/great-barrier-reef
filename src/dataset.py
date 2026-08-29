# -*- coding: utf-8 -*-
"""
dataset.py — Data pipeline for Great Barrier Reef starfish detection (YOLOv8)

Prepares raw Kaggle annotations and images for Ultralytics YOLOv8 by:
1. Merging raw metadata with cross-validation split definitions.
2. Downsampling empty (negative) training frames if requested.
3. Converting COCO pixel coordinates into normalized YOLO format.
4. Symlinking images and exporting target labels and data.yaml.
"""

import ast
import os

import pandas as pd


def load_data(train_csv_path: str, splits_csv_path: str) -> pd.DataFrame:
    """
    Load raw dataset labels and merge with cross-validation split assignments.
    Args:
        train_csv_path: Path to Kaggle train.csv file.
        splits_csv_path: Path to cross-validation splits metadata file.
    Returns:
        DataFrame containing merged annotations, empty-frame flags, and assigned splits.
    """
    df = pd.read_csv(train_csv_path)
    df['annotations'] = df['annotations'].apply(ast.literal_eval)
    df['is_empty'] = df['annotations'].apply(len) == 0

    splits = pd.read_csv(splits_csv_path)

    if len(splits) != len(df):
      raise ValueError(f"splits.csv contains {len(splits)} rows",f"but train.csv contains {len(df)} rows.")

    required_columns = {"image_id", "split"}
    missing = required_columns - set(splits.columns)

    if missing:
        raise ValueError(f"Missing columns in splits.csv: {missing}")
    if splits["image_id"].duplicated().any():
        raise ValueError("splits.csv contains duplicate image_id values.")

    return df.merge(splits[['image_id', 'split']], on='image_id', how='inner',validate='one_to_one')


def filter_negatives(df: pd.DataFrame, ratio: float | None=None, seed: int = 42) -> pd.DataFrame:
    """
    Downsample empty (negative) frames based on a target negative-to-positive ratio.
    Note: Should only be applied to training data to preserve true validation metrics.
    Args:
        df: Input DataFrame containing bounding box data.
        ratio: Maximum ratio of empty to positive frames. If None, retains all frames.
        seed: Random seed for reproducibility.
    Returns:
        DataFrame with filtered empty frames.
    """
    if ratio is None:
        return df
    if ratio<0:
      raise ValueError('ratio must be >=0 or None')

    positives = df[~df['is_empty']]
    negatives = df[df['is_empty']]

    n_keep = min(len(negatives), int(len(positives) * ratio))
    sampled_negatives = negatives.sample(n=n_keep, random_state=seed)

    return pd.concat([positives, sampled_negatives]).reset_index(drop=True)


def coco_to_yolo_box(box: dict, img_width: int, img_height: int):
    """
    Convert a COCO-style bounding box into normalized YOLO format.
    Input:
        x, y, width, height in pixels
    Output:
        x_center, y_center, width, height
        normalized to [0, 1]
    Boxes extending outside the image are clipped first.
    Returns None if the resulting box has no valid area.
    """
    x, y, w, h = float(box['x']), float(box['y']), float(box['width']), float(box['height'])

    # Convert corner coordinates
    x1, y1, x2, y2= x, y, x+w, y+h

    # Clip box to image boundaries
    x1 = max(0.0, min(x1, img_width))
    y1 = max(0.0, min(y1, img_height))
    x2 = max(0.0, min(x2, img_width))
    y2 = max(0.0, min(y2, img_height))

    # Recalculate dimensions after clipping
    clipped_w = x2 - x1
    clipped_h = y2 - y1

    # Ignore invalid/zero-area boxes
    if clipped_w <= 0 or clipped_h <= 0:
        return None

    # Convert to YOLO center format
    x_c= ((x1 + x2) / 2) / img_width
    y_c = ((y1 + y2) / 2) / img_height

    width_norm = clipped_w / img_width
    height_norm = clipped_h / img_height

    return x_c,y_c,width_norm,height_norm


def write_yolo_labels(
    df: pd.DataFrame,
    labels_dir: str,
    img_width: int = 1280,
    img_height: int = 720,
) -> None:
    """
    Write normalized YOLO annotation .txt files for each frame.
    Args:
        df: DataFrame with frame metadata and annotations.
        labels_dir: Directory where label text files will be saved.
        img_width: Frame width in pixels.
        img_height: Frame height in pixels.
    """
    os.makedirs(labels_dir, exist_ok=True)

    for _, row in df.iterrows():
        label_path = os.path.join(labels_dir, f"{row['image_id']}.txt")
        lines = []

        for box in row["annotations"]:
            converted = coco_to_yolo_box(box,img_width=img_width,img_height=img_height)
            if converted is None:
                continue

            x_c, y_c, w, h = converted
            lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # Empty files are intentionally created for negative samples
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))


def link_images(df: pd.DataFrame, source_root: str, images_dir: str) -> None:
    """
    Create flattened symlinks pointing to dataset images using image_id naming.
    Args:
        source_root: Directory containing original video subfolders.
        images_dir: Target output directory for symlinks.
    """
    os.makedirs(images_dir, exist_ok=True)

    for image_id in df['image_id']:
        try:
            video_id, video_frame = image_id.split("-", 1)
        except ValueError as e:
            raise ValueError(f"Invalid image_id format: {image_id}") from e

        src = os.path.join(source_root, f'video_{video_id}', f'{video_frame}.jpg')
        dst = os.path.join(images_dir, f'{image_id}.jpg')

        if os.path.exists(dst):
            continue
        if not os.path.exists(src):
          raise FileNotFoundError(f"Source image not found: {src}")

        os.symlink(os.path.abspath(src), dst)


def write_data_yaml(yaml_path: str, images_train_dir: str, images_val_dir: str) -> None:
    """
    Write standard dataset configuration YAML required by YOLOv8.
    Args:
        yaml_path: Target path for the output data.yaml file.
        images_train_dir: Path to training images directory.
        images_val_dir: Path to validation images directory.
    """
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    content = (
        f"train: {os.path.abspath(images_train_dir)}\n"
        f"val: {os.path.abspath(images_val_dir)}\n"
        "nc: 1\n"
        "names: ['starfish']\n"
    )

    with open(yaml_path, 'w') as f:
        f.write(content)



if __name__ == '__main__':
    TRAIN_CSV = 'data/train.csv'
    SPLITS_CSV = 'data/splits.csv'
    RAW_IMAGES_ROOT = 'data/train_images'
    NEGATIVE_RATIO = None

    full_df = load_data(TRAIN_CSV, SPLITS_CSV)

    train_df = full_df[full_df['split'] == 'train']
    val_df = full_df[full_df['split'] == 'val']  # never filtered

    train_df = filter_negatives(train_df, ratio=NEGATIVE_RATIO)

    print(f"Train frames after filtering: {len(train_df)}")
    print(f"Val frames (untouched):       {len(val_df)}")

    write_yolo_labels(train_df, labels_dir='data/labels/train')
    write_yolo_labels(val_df, labels_dir='data/labels/val')

    link_images(train_df, RAW_IMAGES_ROOT, images_dir='data/images/train')
    link_images(val_df, RAW_IMAGES_ROOT, images_dir='data/images/val')

    write_data_yaml('data/data.yaml', 'data/images/train', 'data/images/val')

    print("Label files, image links, and data.yaml written.")

