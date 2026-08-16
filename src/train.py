"""
train.py - Training loop for YOLOv8 on Great Barrier Reef COTS detection

Currently pointed at DUMMY data (see make_dummy_data.py) so the full
loop can be tested before Person A's real data pipeline is ready.

TO SWITCH TO REAL DATA LATER:
Just change DATA_YAML_PATH below to point at the real data.yaml that
Person A's pipeline produces. Nothing else in this file needs to change.
"""

import os
from model import load_model, get_train_config

# ---- CONFIG ----
# TODO: swap this to the real data.yaml once Person A's pipeline is ready
DATA_YAML_PATH = "dummy_data/data.yaml"
OUTPUT_DIR = "outputs/train_runs"
RUN_NAME = "dummy_run"  # rename to e.g. "final_model_v1" for real runs


def train():
    model = load_model()

    # Low epoch count for the dummy smoke test - raise this once
    # you're training on real data (e.g. 50-100 epochs)
    config = get_train_config(epochs=3)

    results = model.train(
        data=DATA_YAML_PATH,
        imgsz=config["imgsz"],
        epochs=config["epochs"],
        batch=config["batch"],
        lr0=config["lr0"],
        project=OUTPUT_DIR,
        name=RUN_NAME,
    )

    print("Training complete.")
    print(f"Results and checkpoints saved to: {OUTPUT_DIR}/{RUN_NAME}")
    return results


if __name__ == "__main__":
    train()