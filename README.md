# Great Barrier Reef — Crown-of-Thorns Starfish Detection

A team-based computer vision project for detecting **crown-of-thorns starfish (COTS)** in underwater images using YOLO object detection.

The project is based on the Kaggle **TensorFlow — Help Protect the Great Barrier Reef** competition dataset:

https://www.kaggle.com/competitions/tensorflow-great-barrier-reef

The workflow covers exploratory data analysis, dataset preparation, model experimentation, training, evaluation, and inference.

## Dataset & EDA

Exploratory data analysis is available in:

```text
notebooks/eda.ipynb
```

Main findings:

* **23,501** total image frames
* **3 videos** and **20 sequences**
* About **79.1%** of frames contain no annotated starfish
* Images have a resolution of **1280 × 720**
* No duplicate `image_id` values or missing image files were found

The large number of empty frames motivated experiments with different amounts of negative/background training data.

## Train / Validation Split

The train/validation split is created in:

```text
notebooks/train_val_split.ipynb
```

and stored in:

```text
data/splits.csv
```

Frames are split at the **sequence level** rather than randomly. Frames from the same underwater sequence are visually and temporally related, so random frame-level splitting could cause very similar images to appear in both training and validation sets.

`GroupShuffleSplit` is used to keep complete sequences together.

Final split:

```text
Train: 19,444 frames
Validation: 4,057 frames
```

The validation set remains unchanged across experiments to allow fair model comparison.

## Dataset Preparation

Dataset preparation is implemented in:

```text
src/dataset.py
```

The pipeline:

1. Loads the original Kaggle annotations
2. Merges them with the fixed train/validation split
3. Identifies positive and negative frames
4. Optionally downsamples negative training frames
5. Converts COCO bounding boxes to normalized YOLO format
6. Creates YOLO label files
7. Links images into train/validation directories
8. Generates `data.yaml`

There is one detection class:

```text
0 → starfish
```

## Negative-Frame Experiments

Because most frames contain no starfish, different negative-to-positive training ratios were tested with **YOLOv8n** in:

```text
notebooks/train-yolov8n.ipynb
```

| Negative ratio | Description                    |     mAP@50 |  mAP@50–95 |
| -------------- | ------------------------------ | ---------: | ---------: |
| `0.0`          | Positive frames only           |     0.0084 |     0.0022 |
| `1.0`          | Up to 1 negative per positive  |     0.0205 |     0.0101 |
| `2.0`          | Up to 2 negatives per positive |     0.0495 |     0.0212 |
| `all`          | All available negatives        | **0.0600** | **0.0257** |

These preliminary experiments showed that including background examples significantly improved detection performance.

## Baseline Model

A YOLOv8n baseline was trained using a negative ratio of `2.0`.

Main configuration:

* Model: **YOLOv8n**
* Image size: **320**
* Batch size: **32**
* Epochs: **20**
* Training frames: **12,846**
* Validation frames: **4,057**

Baseline results:

| Metric    |  Score |
| --------- | -----: |
| mAP@50    | 0.0508 |
| mAP@50–95 | 0.0252 |

The full baseline experiment is available in:

```text
notebooks/baseline_model.ipynb
```
## Baseline Checkpoint

The baseline YOLOv8n model checkpoint (`best.pt`) is stored externally because model weight files are not tracked in this repository.

[Baseline checkpoint folder](https://drive.google.com/drive/folders/1xf0_AAXmX-Zf0HHk1QR14CI8TpzXakV4?usp=sharing)

The checkpoint can be used for inference and comparison with the final YOLO11s model.

## Final Model

After the initial YOLOv8 experiments, the project moved to **YOLO11s**, which provides greater model capacity for detecting small COTS in underwater scenes.

Final-model training is documented in:

```text
notebooks/train-final-model-yolo11s.ipynb
```

The reusable model configuration is defined in:

```text
src/model.py
```

The current configuration uses transfer learning with pretrained YOLO11s weights, higher-resolution input, early stopping, and image augmentation.

### Final YOLO11s Checkpoint

The final trained YOLO11s checkpoint is available here:

[Download final model checkpoint](https://drive.google.com/file/d/1Jnt6G-A5glUnfwgBp0nbs4rui6Sf9QK6/view?usp=sharing)

After downloading the checkpoint, place it in:

```text
outputs/final.pt
```
## Evaluation

The **main evaluation metric for this project is F2 score**.

F2 gives more weight to **recall** than precision, which is useful for this task because missing an actual starfish detection is especially important.

Additional metrics include:

* Precision
* Recall
* mAP@50
* mAP@50–95

F2 calculation and metric processing are implemented in:

```text
src/metrics.py
```

### Final Results

| Metric | Score |
|---|---:|
| **F2** | **0.603** |
| Precision | 0.877 |
| Recall | 0.559 |
| mAP@50 | 0.619 |
| mAP@50–95 | 0.359 |

## Prediction

Inference on new images is implemented in:

```text
src/predict.py
```

The script loads the trained model, detects COTS, saves an image with predicted bounding boxes, and records detection coordinates and confidence scores.

Generated prediction logs and model checkpoints are not tracked in Git.

## Project Structure

```text
great-barrier-reef/
│
├── data/
│   └── splits.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── train_val_split.ipynb
│   ├── train-yolov8n.ipynb
│   ├── baseline_model.ipynb
│   └── train-final-model-yolo11s.ipynb
│
├── src/
│   ├── dataset.py
│   ├── metrics.py
│   ├── model.py
│   └── predict.py
│
├── .gitignore
├── README.md
├── data_pipeline.md
└── requirements.txt
```

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Technologies

* Python
* PyTorch
* Ultralytics YOLO
* Pandas / NumPy
* OpenCV
* scikit-learn
* Matplotlib
* Jupyter / Google Colab
