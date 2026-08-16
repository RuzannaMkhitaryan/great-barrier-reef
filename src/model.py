"""
model.py - YOLOv8 model loading and training configuration
Great Barrier Reef COTS Detection Project
"""

from ultralytics import YOLO


def load_model(pretrained_weights="yolov8n.pt"):
    """
    Load a YOLOv8 model, starting from pretrained COCO weights.

    We use pretrained weights (transfer learning) instead of training
    from scratch - this converges much faster and works better with
    a limited dataset. Ultralytics auto-downloads yolov8n.pt the first
    time you run this if it's not already cached locally.
    """
    model = YOLO(pretrained_weights)
    return model


def get_train_config(image_size=960, epochs=50, batch_size=16, learning_rate=0.01):
    """
    Returns a dict of training hyperparameters.

    image_size=960: COTS (starfish) objects are small in the frame,
    so a higher resolution than the YOLO default (640) helps detect
    small/dense targets - matches the known challenge of this dataset.

    Tune these later once you're running against real data:
    - epochs: start low (e.g. 3-5) for smoke tests, raise once things work
    - batch_size: lower this if you hit GPU memory errors
    """
    config = {
        "imgsz": image_size,
        "epochs": epochs,
        "batch": batch_size,
        "lr0": learning_rate,
    }
    return config


if __name__ == "__main__":
    # Quick sanity check - run this file directly (`python model.py`)
    # to confirm the model loads correctly before wiring it into train.py
    model = load_model()
    print("Model loaded successfully.")
    print("Task:", model.task)
    print("Model summary:")
    model.info()