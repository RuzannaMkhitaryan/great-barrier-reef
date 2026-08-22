"""
model.py - YOLO model loading and training configuration
Great Barrier Reef COTS Detection Project

Swapping YOLO versions:
    Ultralytics uses one unified API across YOLOv8/v9/v10/YOLO11/YOLO12/YOLO26.
    To change architectures, just change `pretrained_weights` below (or pass
    a different value into load_model()), e.g.:
        "yolov8n.pt" / "yolov8s.pt" / "yolov8m.pt"
        "yolo11n.pt" / "yolo11s.pt" / "yolo11m.pt"
        "yolo26n.pt" / "yolo26s.pt" / "yolo26m.pt"
    Everything downstream (train(), get_train_config()) stays the same.
    Avoid yolo12*.pt - Ultralytics flags YOLO12's attention layers as
    causing training instability, so it's not recommended for real runs.
"""

from ultralytics import YOLO


def load_model(pretrained_weights="yolo11s.pt"):
    """
    Load a YOLO model, starting from pretrained COCO weights (transfer
    learning) - converges much faster and works better with a limited
    dataset than training from scratch.

    Default changed from yolov8n.pt (nano, ~3M params) to yolo11s.pt
    (small, ~9M params): the nano model doesn't have much spare capacity
    for a hard, small-object task like COTS detection, and YOLO11
    generally outperforms YOLOv8 at a comparable parameter count.
    Ultralytics auto-downloads the checkpoint the first time you run this
    if it isn't already cached locally.
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
    Returns a dict of training hyperparameters.

    image_size=1280: COTS (starfish) are small in-frame, so a higher
    resolution than the YOLO default (640) - and higher than our earlier
    960 - helps detect small/dense targets. Lower batch_size if you hit a
    GPU out-of-memory error at this size.

    epochs=100 / patience=20: an earlier 8-epoch smoke-test run was still
    improving every single epoch (box/cls/dfl loss all still dropping)
    when it stopped - it never converged. Train for real and let
    `patience` (epochs with no val improvement before early stopping)
    decide when to stop, instead of a fixed low epoch count.

    optimizer / learning_rate: an earlier run asked for optimizer="SGD",
    lr0=0.01 in the `model.train()` call, but the Ultralytics log showed
    it silently using optimizer='auto' -> AdamW(lr=0.002) instead. These
    are threaded through explicitly here so train() can print + verify
    what Ultralytics actually used, instead of that mismatch hiding in
    the log.

    device: pass a list like [0, 1] to use both GPUs on a Kaggle T4x2
    session. Defaults to None (Ultralytics auto-selects a single device).

    save_period: save a checkpoint (epochN.pt) every N epochs, in addition
    to the usual last.pt/best.pt. Defaults to -1 (disabled). Set this to a
    positive number for any run long enough that you'd want to resume it
    after an interruption instead of losing the whole run.

    mosaic / mixup / copy_paste / hsv_*: augmentation tuned for a small,
    sparse, underwater object - copy_paste and mixup oversample the rare
    positive examples, hsv_* accounts for underwater color cast.
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
