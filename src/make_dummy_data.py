"""
make_dummy_data.py - Generates a tiny fake dataset so train.py can be
tested end-to-end BEFORE Person A's real data pipeline is ready.

Run this once: python make_dummy_data.py
It creates:
  dummy_data/
    images/train/*.jpg   (5 random noise images)
    images/val/*.jpg     (2 random noise images)
    labels/train/*.txt   (matching YOLO-format labels, one fake box each)
    labels/val/*.txt
    data.yaml            (tells YOLO where everything is)

Delete the dummy_data/ folder once you swap in real data.
"""

import os
import random
from PIL import Image

BASE_DIR = "dummy_data"
IMG_SIZE = 960


def make_random_image(path):
    """Creates a random noise JPG image, standing in for a real frame."""
    img = Image.new(
        "RGB",
        (IMG_SIZE, IMG_SIZE),
        color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ),
    )
    img.save(path)


def make_fake_label(path):
    """
    Creates a YOLO-format label file with one fake bounding box.
    Format: class_id x_center y_center width height (all normalized 0-1)
    class_id 0 = our only class, "starfish"
    """
    with open(path, "w") as f:
        f.write("0 0.5 0.5 0.1 0.1\n")


def build_split(split_name, n_images):
    img_dir = os.path.join(BASE_DIR, "images", split_name)
    lbl_dir = os.path.join(BASE_DIR, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(n_images):
        img_path = os.path.join(img_dir, f"dummy_{i}.jpg")
        lbl_path = os.path.join(lbl_dir, f"dummy_{i}.txt")
        make_random_image(img_path)
        make_fake_label(lbl_path)


def write_data_yaml():
    yaml_path = os.path.join(BASE_DIR, "data.yaml")
    content = f"""path: {os.path.abspath(BASE_DIR)}
train: images/train
val: images/val

names:
  0: starfish
"""
    with open(yaml_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    build_split("train", n_images=5)
    build_split("val", n_images=2)
    write_data_yaml()
    print(f"Dummy dataset created at ./{BASE_DIR}/")
    print("Run train.py now to confirm the training loop works end-to-end.")