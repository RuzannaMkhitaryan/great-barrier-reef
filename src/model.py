"""
YOLO model loading and training configuration for the
Great Barrier Reef Crown-of-Thorns Starfish detection project.
"""

from ultralytics import YOLO

def load_model(pretrained_weights="yolo11s.pt"):
     """
    Load a pretrained YOLO model for transfer learning.
    Args:
        pretrained_weights: Path or name of pretrained YOLO weights.
    Returns:
        Loaded Ultralytics YOLO model.
    """
    model = YOLO(pretrained_weights)
    return model

def get_train_config(
    image_size=1280,
    epochs=100,
    batch_size=16,
    learning_rate=0.01,
    patience=20,
    optimizer="SGD",
    device=None,
    save_period=-1,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
):
    
     """
    Return the training hyperparameters for the YOLO model.
    Higher input resolution is used to improve detection of small COTS
    in underwater frames. Early stopping prevents unnecessary training
    once validation performance stops improving.
    Augmentation settings are used to improve generalization and account
    for sparse targets and variations in underwater image appearance.
    Args:
        image_size: Input image size used during training.
        epochs: Maximum number of training epochs.
        batch_size: Number of images per training batch.
        learning_rate: Initial learning rate.
        patience: Epochs without validation improvement before early stopping.
        optimizer: Optimizer used during training.
        device: Training device or device list. None lets Ultralytics select automatically.
        save_period: Interval for saving intermediate checkpoints.
        mosaic: Mosaic augmentation probability.
        mixup: MixUp augmentation probability.
        copy_paste: Copy-paste augmentation probability.
        hsv_h: HSV hue augmentation.
        hsv_s: HSV saturation augmentation.
        hsv_v: HSV value augmentation.
    Returns:
        Dictionary of Ultralytics training parameters.
    """
    config = {
        "imgsz": image_size,
        "epochs": epochs,
        "batch": batch_size,
        "lr0": learning_rate,
        "patience": patience,
        "optimizer": optimizer,
        "device": device,
        "save_period": save_period,
        "mosaic": mosaic,
        "mixup": mixup,
        "copy_paste": copy_paste,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
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
