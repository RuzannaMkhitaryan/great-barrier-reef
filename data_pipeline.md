## Data Pipeline and Next Steps

### Current Pipeline

```text id="d8m5h2"
Raw Kaggle Dataset
        ↓
EDA
        ↓
Sequence-level Train / Validation Split
        ↓
splits.csv
        ↓
Dataset Preparation
        ↓
Negative-frame Sampling
        ↓
COCO → YOLO Annotation Conversion
        ↓
YOLO Dataset Generation
        ↓
Preliminary YOLOv8 Training
        ↓
Negative-ratio Comparison
```

### 1. Exploratory Data Analysis — Completed

File:

```text id="kflw32"
eda.ipynb
```

Main findings:

* 23,501 frames
* 3 videos
* 20 sequences
* high class imbalance
* approximately 79% of frames contain no annotated starfish
* different videos have different empty-frame distributions
* image dimensions were checked
* missing images and duplicate image IDs were checked

These findings motivated the negative-frame sampling experiments.

### 2. Train / Validation Split — Completed

File:

```text id="l62w1u"
train_val_split.ipynb
```

The split is performed at the sequence level rather than individual-frame level.

This prevents adjacent or near-duplicate frames from the same sequence from appearing in both training and validation sets.

Multiple candidate sequence-level splits are generated for each video, and the most representative split is selected based on:

* validation-set size
* empty-frame distribution

Output:

```text id="gq7rfo"
data/splits.csv
```

Current split:

```text id="57x7tp"
Train: 19,444 frames
Val:    4,057 frames
```

### 3. Dataset Preparation — Completed

File:

```text id="pu9fq4"
dataset.py
```

Responsibilities:

```text id="79k0m4"
train.csv + splits.csv
        ↓
parse annotations
        ↓
assign train / validation
        ↓
optionally sample negative training frames
        ↓
clip invalid/out-of-frame bounding boxes
        ↓
COCO → YOLO conversion
        ↓
create YOLO .txt labels
        ↓
create image symlinks
        ↓
generate data.yaml
```

Validation data is never filtered.

### 4. Negative-frame Experiments — Preliminary Stage Completed

File:

```text id="k9flcm"
train.ipynb
```

Tested negative-to-positive ratios:

```text id="4ouvkq"
0.0
1.0
2.0
None / all negatives
```

Each experiment:

```text id="9cydzq"
cleans generated YOLO data
        ↓
loads fixed train/val split
        ↓
changes training negatives only
        ↓
creates YOLO dataset
        ↓
initializes fresh YOLOv8n
        ↓
trains
        ↓
evaluates on fixed validation set
```

Preliminary results indicate that keeping more negative/background frames improves validation performance.

The all-negative configuration produced the best result in the current short experiment.

---

# Next Steps

### 5. Finalize the Negative-frame Strategy

Do not treat the short 5-epoch experiment as the final model-selection result.

Recommended candidates for longer experiments:

```text id="x82y7m"
ratio = None
ratio = 2.0
```

Compare them again with longer training before choosing the final configuration.

### 6. Increase Training Resolution

The preliminary experiments use:

```text id="r6mt51"
imgsz = 320
```

Starfish can be relatively small objects, so future training should test higher resolutions such as:

```text id="rly3y2"
640
1280
```

Start with `640` because it provides a better balance between detail, GPU memory usage, and training speed.

### 7. Train for More Epochs

Current ratio experiments use only a few epochs for fast comparison.

For proper model training, increase the number of epochs, for example:

```text id="v6eohc"
30+
```

and use early stopping where appropriate.

### 8. Improve Experiment Reproducibility

Use a fixed training seed, for example:

```python id="q2l8c2"
seed=42
```

in YOLO training so that ratio and model comparisons are more controlled.

Keep the existing fixed:

```text id="1hjptc"
train / validation split
```

for all model comparisons.

### 9. Track More Validation Metrics

Continue recording:

```text id="i4pej3"
mAP50
mAP50-95
```

Also consider tracking:

```text id="vl0dru"
precision
recall
training loss
validation loss
```

This will make it easier to understand why one model performs better than another.

### 10. Inspect Prediction Errors

After obtaining a stronger model, visually inspect validation predictions.

Focus on:

```text id="loefle"
false positives
false negatives
missed small starfish
multiple-starﬁsh frames
difficult reef backgrounds
```

This can guide later preprocessing or model changes.

### 11. Model / Hyperparameter Experiments

After establishing the baseline YOLOv8n pipeline, possible experiments include:

```text id="4x17lr"
YOLO model size
image resolution
batch size
learning rate
augmentation settings
negative-frame ratio
number of epochs
```

Change one major factor at a time where possible so results remain interpretable.

### 12. Select the Final Model

Choose the model based primarily on validation performance rather than training performance.

Save:

```text id="9ckm4q"
best model weights
training configuration
negative ratio
image size
metrics
```

so the experiment can be reproduced.

### 13. Inference Pipeline

After model selection, implement inference on unseen frames.

Expected flow:

```text id="0qcnge"
Input image
    ↓
Load trained YOLO model
    ↓
Run detection
    ↓
Bounding boxes + confidence scores
    ↓
Apply confidence threshold
    ↓
Final starfish detections
```

### 14. Competition-style Evaluation / Submission

If the project intends to reproduce the original Kaggle task, the final stage can include:

```text id="dtj78t"
test-video inference
        ↓
prediction formatting
        ↓
competition-compatible output
```

This part has not yet been implemented in the current project files.

---

## Full Project Flow

```text id="vmxagq"
RAW DATA
   │
   ▼
EDA
   │
   ▼
SEQUENCE-LEVEL SPLIT
   │
   ▼
splits.csv
   │
   ▼
DATASET PREPARATION
   │
   ├── negative sampling
   ├── COCO → YOLO
   ├── labels
   ├── image links
   └── data.yaml
   │
   ▼
PRELIMINARY YOLO EXPERIMENTS
   │
   ▼
SELECT PROMISING NEGATIVE RATIOS
   │
   ▼
LONGER TRAINING
   │
   ├── higher resolution
   ├── more epochs
   └── reproducible settings
   │
   ▼
MODEL COMPARISON
   │
   ▼
ERROR ANALYSIS
   │
   ▼
FINAL MODEL
   │
   ▼
INFERENCE
   │
   ▼
FINAL EVALUATION / SUBMISSION
```

## Current Project Status

```text id="brzx8u"
EDA                         ✅
Train / validation split    ✅
YOLO dataset preparation    ✅
Negative-ratio experiment   ✅ preliminary
Final training              ⏳
Model selection             ⏳
Error analysis              ⏳
Inference pipeline          ⏳
Final evaluation            ⏳
```
