# Data Pipeline

This document describes the complete data and model pipeline used for Crown-of-Thorns Starfish (COTS) detection in the Great Barrier Reef project.

The project is based on the Kaggle **TensorFlow — Help Protect the Great Barrier Reef** competition:

https://www.kaggle.com/competitions/tensorflow-great-barrier-reef

## 1. Exploratory Data Analysis

Exploratory data analysis is performed in:

```text
notebooks/eda.ipynb
```

Main dataset findings:

- **23,501** image frames
- **3 videos**
- **20 sequences**
- Image resolution: **1280 × 720**
- About **79.1%** of frames contain no annotated starfish
- No duplicate `image_id` values were found
- No missing image files were detected

The high proportion of empty frames influenced both the train/validation splitting strategy and the decision to experiment with different amounts of negative/background training data.

## 2. Train / Validation Split

The canonical train/validation split is generated in:

```text
notebooks/train_val_split.ipynb
```

and stored in:

```text
data/splits.csv
```

The data is split at the **sequence level** instead of randomly by frame.

Frames belonging to the same underwater sequence are visually and temporally related. Random frame-level splitting could therefore place very similar images in both training and validation sets and cause data leakage.

`GroupShuffleSplit` is used to generate sequence-level splits while keeping complete sequences together.

Final split:

```text
Train: 19,444 frames
Validation: 4,057 frames
```

The same validation set is preserved across experiments to ensure fair model comparison.

## 3. Dataset Preparation

Dataset preparation is implemented in:

```text
src/dataset.py
```

The preprocessing pipeline:

1. Loads the original Kaggle `train.csv`
2. Parses bounding-box annotations
3. Merges annotations with `data/splits.csv`
4. Identifies positive and negative frames
5. Optionally downsamples negative training frames
6. Clips bounding boxes to valid image boundaries
7. Converts annotations from COCO to YOLO format
8. Creates YOLO label files
9. Links images into train and validation directories
10. Generates `data.yaml`

The original bounding boxes use COCO format:

```text
x, y, width, height
```

They are converted to normalized YOLO format:

```text
class_id x_center y_center width height
```

The project contains one object class:

```text
0 → starfish
```

Negative frames receive empty YOLO label files.

## 4. Negative-Frame Sampling

Because most frames contain no annotated starfish, different negative-to-positive ratios were tested.

The supported configurations are:

```text
0.0  → no negative frames
1.0  → up to 1 negative per positive frame
2.0  → up to 2 negatives per positive frame
None → all available negative frames
```

Only the training set is downsampled. The validation set remains unchanged.

Experiments are documented in:

```text
notebooks/train-yolov8n.ipynb
```

### Preliminary Results

| Negative ratio | Description | mAP@50 | mAP@50–95 |
|---|---|---:|---:|
| `0.0` | Positive frames only | 0.0084 | 0.0022 |
| `1.0` | Up to 1 negative per positive | 0.0205 | 0.0101 |
| `2.0` | Up to 2 negatives per positive | 0.0495 | 0.0212 |
| `all` | All available negatives | **0.0600** | **0.0257** |

The experiments showed that including background examples significantly improved detection performance.

## 5. Baseline Model

A **YOLOv8n** model was trained as a baseline reference.

The baseline experiment is available in:

```text
notebooks/baseline_model.ipynb
```

Main configuration:

- Model: **YOLOv8n**
- Image size: **320**
- Batch size: **32**
- Epochs: **20**
- Negative ratio: **2.0**

Baseline results:

| Metric | Score |
|---|---:|
| mAP@50 | 0.0508 |
| mAP@50–95 | 0.0252 |

The baseline checkpoint is stored externally because trained `.pt` model files are not tracked in Git:

[Baseline checkpoint folder](https://drive.google.com/drive/folders/1xf0_AAXmX-Zf0HHk1QR14CI8TpzXakV4?usp=sharing)

The checkpoint can be used for inference and comparison with the final model.

## 6. Final Model Training

After the preliminary YOLOv8 experiments, the final model was trained using **YOLO11s**.

Final training is documented in:

```text
notebooks/train-final-model-yolo11s.ipynb
```

Reusable model configuration is defined in:

```text
src/model.py
```

The final training setup uses:

- Pretrained **YOLO11s** weights
- Transfer learning
- Input image size of **1280**
- SGD optimizer
- Data augmentation
- Early stopping
- Model checkpointing

The higher input resolution helps preserve information about small COTS in underwater frames.

The best model checkpoint was obtained at **epoch 34**.

## 7. Evaluation

Evaluation utilities are implemented in:

```text
src/metrics.py
```

The main evaluation metric for the project is the **F2 score**.

F2 gives more importance to recall than precision:

```text
F2 = 5 × (Precision × Recall) / (4 × Precision + Recall)
```

This is useful for COTS detection because missing an existing starfish is particularly important.

### Final Validation Results

| Metric | Score |
|---|---:|
| **F2** | **0.603** |
| Precision | 0.877 |
| Recall | 0.559 |
| mAP@50 | 0.619 |
| mAP@50–95 | 0.359 |

The final YOLO11s model achieved high precision while maintaining moderate recall, resulting in an **F2 score of 0.603**.

## 8. Prediction

Inference on new images is implemented in:

```text
src/predict.py
```

The prediction pipeline:

```text
Input image
     ↓
Load trained YOLO model
     ↓
Run inference
     ↓
Apply confidence threshold
     ↓
Extract detections
     ↓
Bounding boxes + confidence scores
     ↓
Save visualized prediction
```

Prediction information can also be written to:

```text
prediction_results.txt
```

This file is generated during inference and is excluded from version control through `.gitignore`.

Trained model weights such as `.pt` files are also kept outside the Git repository.

## Pipeline Overview

```text
Raw Kaggle Dataset
        ↓
Exploratory Data Analysis
        ↓
Sequence-Level Train / Validation Split
        ↓
Dataset Preparation
        ↓
COCO → YOLO Annotation Conversion
        ↓
Negative-Frame Sampling Experiments
        ↓
YOLOv8n Baseline
        ↓
YOLO11s Final Model
        ↓
Evaluation
(F2, Precision, Recall, mAP)
        ↓
Inference on New Images
```
