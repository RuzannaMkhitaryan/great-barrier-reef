## Data & EDA Findings

**Dataset:** TensorFlow — Help Protect the Great Barrier Reef, focused on crown-of-thorns starfish detection in underwater video frames.

Full exploratory analysis is available in [`notebooks/eda.ipynb`](notebooks/eda.ipynb).

### Key EDA Findings

* **23,501 total frames** across **3 videos** and **20 sequences**.
* **79.1% of frames contain no annotated starfish**, resulting in substantial class imbalance.
* The three videos have noticeably different empty-frame rates:

  * Video 0: **68.1%**
  * Video 1: **74.5%**
  * Video 2: **92.1%**
* Sequence counts also differ between videos:

  * Video 0: **8 sequences**
  * Video 1: **8 sequences**
  * Video 2: **4 sequences**
* No duplicate `image_id` values were found.
* No missing image files were detected.
* Sampled images were confirmed to have a resolution of **1280 × 720**.

These findings influenced both the train/validation splitting strategy and the decision to experiment with different amounts of negative/background training frames.

---

## Train / Validation Split

The canonical split is generated in:

```text
notebooks/train_val_split.ipynb
```

and saved as:

```text
data/splits.csv
```

### Why Sequence-Level Splitting?

Frames from the same video sequence are temporally related and may be near-duplicates. Randomly splitting individual frames could therefore place very similar images in both training and validation sets, causing data leakage.

To prevent this, complete sequences are kept entirely on one side:

```text
Sequence A → Train
Sequence B → Train
Sequence C → Validation
```

Each video is processed separately so that all three videos are represented in both training and validation data.

### Split Selection

For each video, multiple candidate sequence-level splits are generated using `GroupShuffleSplit`.

The current implementation tests **500 candidate splits** and selects the candidate that best balances:

1. closeness to the desired **20% validation size**;
2. similarity between the validation empty-frame rate and the full video's empty-frame rate.

Because complete sequences must remain together, perfect balancing is not always possible, especially for Video 2, which contains only four sequences.

### Final Split

```text
Train: 19,444 frames
Val:    4,057 frames
```

Per-video distribution:

```text
          train   val
Video 0    5430  1278
Video 1    6978  1254
Video 2    7036  1525
```

Empty-frame distribution:

```text
          train   val   full
Video 0    68.0  68.1   68.1
Video 1    72.8  84.1   74.5
Video 2    90.8  98.1   92.1
```

Integrity checks verify that:

* no sequence appears in both train and validation;
* no `image_id` appears in both sets;
* every original frame is included exactly once.

The resulting `data/splits.csv` contains:

```text
image_id, video_id, sequence, split
```

and serves as the canonical train/validation membership used by downstream preprocessing and training code.

---

## Dataset Preparation

YOLO dataset preparation is implemented in:

```text
notebooks/dataset.py
```

The script:

1. loads the original Kaggle `train.csv`;
2. parses the annotation column;
3. merges the annotations with `data/splits.csv`;
4. marks frames as positive or negative;
5. optionally downsamples negative **training** frames;
6. converts COCO-style bounding boxes to YOLO format;
7. clips bounding boxes to valid image boundaries;
8. creates YOLO `.txt` label files;
9. creates symbolic links to the original images;
10. generates `data/data.yaml`.

The validation set is **never downsampled**, ensuring that all experiments are evaluated on the same fixed validation distribution.

### Negative-Frame Sampling

The training pipeline supports different negative-to-positive ratios:

```text
ratio = 0.0
→ keep no negative frames

ratio = 1.0
→ keep at most 1 negative frame per positive frame

ratio = 2.0
→ keep at most 2 negative frames per positive frame

ratio = None
→ keep all available negative frames
```

Sampling uses a fixed random seed for reproducibility.

### Bounding-Box Conversion

Original annotations are provided in COCO-style format:

```text
x, y, width, height
```

They are converted into normalized YOLO format:

```text
class_id x_center y_center width height
```

There is only one object class:

```text
0 → starfish
```

Bounding boxes extending outside image boundaries are clipped before conversion. Invalid or zero-area boxes are ignored.

Negative frames receive empty YOLO label files.

---

## Generated YOLO Dataset

Dataset preparation creates the following structure:

```text
data/
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
├── data.yaml
└── splits.csv
```

The image directories contain symbolic links rather than duplicated image files.

`data.yaml` defines the YOLO dataset configuration:

```yaml
train: path/to/data/images/train
val: path/to/data/images/val

nc: 1
names: ['starfish']
```

---

## Preliminary YOLOv8 Experiments

Initial experiments are implemented in:

```text
notebooks/train.ipynb
```

A lightweight pretrained **YOLOv8n** model is used to evaluate the effect of negative/background frames.

The following negative-to-positive configurations were tested:

```text
0.0
1.0
2.0
all negatives
```

For every experiment, the pipeline:

```text
Clean previous generated YOLO dataset
        ↓
Load train.csv + splits.csv
        ↓
Use the same fixed train/validation split
        ↓
Downsample training negatives according to ratio
        ↓
Generate YOLO labels and image links
        ↓
Generate data.yaml
        ↓
Initialize a fresh YOLOv8n model
        ↓
Train
        ↓
Evaluate on the unchanged validation set
```

### Preliminary Results

| Negative Ratio | Training Data                    |     mAP@50 |  mAP@50–95 |
| -------------- | -------------------------------- | ---------: | ---------: |
| `0.0`          | Positive frames only             |     0.0084 |     0.0022 |
| `1.0`          | Up to 1 negative per positive    |     0.0205 |     0.0101 |
| `2.0`          | Up to 2 negatives per positive   |     0.0495 |     0.0212 |
| `all`          | All available training negatives | **0.0600** | **0.0257** |

The preliminary results show a consistent improvement as more negative/background frames are included.

Therefore, **all negatives (`ratio=None`) currently provide the best preliminary result**, while `ratio=2.0` remains a useful candidate when faster experimentation is needed.

These results were obtained using short training runs and should not yet be considered final model-selection results.

---

## Current Data Pipeline

```text
Kaggle train.csv + train_images/
            │
            ▼
        EDA
   notebooks/eda.ipynb
            │
            ▼
Sequence-Level Train / Val Split
 notebooks/train_val_split.ipynb
            │
            ▼
      data/splits.csv
            │
            ▼
     Dataset Preparation
    notebooks/dataset.py
            │
            ├── split assignment
            ├── negative sampling
            ├── COCO → YOLO conversion
            ├── YOLO labels
            ├── image symlinks
            └── data.yaml
            │
            ▼
      YOLOv8 Training
    notebooks/train.ipynb
            │
            ▼
      Validation Metrics
     mAP@50 / mAP@50–95
```
## Baseline Model

After selecting a negative-to-positive ratio of `2.0`, a baseline YOLOv8n model was trained using the fixed sequence-level train/validation split.

### Configuration

- Model: YOLOv8n
- Pretrained weights: Yes (`yolov8n.pt`)
- Negative ratio: `2.0`
- Training frames: 12,846
- Validation frames: 4,057
- Image size: 320
- Batch size: 32
- Epochs: 20
- Random seed: 42
- Optimizer: AdamW (automatically selected by Ultralytics)
- Learning rate: 0.002

### Baseline Results

| Metric | Score |
|---|---:|
| mAP@50 | 0.0508 |
| mAP@50-95 | 0.0252 |

The model checkpoint corresponding to the best validation performance was saved as `best.pt`.

### Baseline Checkpoint

The baseline model's best checkpoint (`best.pt`) is available in Google Drive:

[Baseline checkpoint folder](https://drive.google.com/drive/folders/1xf0_AAXmX-Zf0HHk1QR14CI8TpzXakV4?usp=sharing)

This checkpoint can be used by other team members for inference, comparison with later experiments, or as a starting point for further model development.

The baseline model is intended as a reference configuration. Later experiments can compare changes in image resolution, model size, optimizer settings, learning rate, augmentation, and other hyperparameters against this baseline.
## Next Steps

* Re-evaluate the strongest negative-frame configurations (`ratio=None` and `ratio=2.0`) with longer training.
* Increase training resolution from the preliminary `320` setting, starting with `640`.
* Increase the number of training epochs and use early stopping where appropriate.
* Use fixed training seeds for more reproducible model comparisons.
* Track additional metrics such as precision and recall.
* Perform visual error analysis on false positives, false negatives, and difficult small-starﬁsh examples.
* Compare model and training configurations before selecting the final model.
* Build the final inference pipeline for unseen video frames.
