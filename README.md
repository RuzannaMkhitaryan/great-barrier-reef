## Data & EDA Findings

Dataset: [TensorFlow - Help Protect the Great Barrier Reef](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef)
(crown-of-thorns starfish detection in underwater video frames).

**Key findings from EDA** (full analysis: `notebooks/eda.ipynb`):
- 23,501 total frames across 3 videos; **79.1% of frames have no starfish
  annotations** (significant class imbalance).
- The 3 videos differ substantially in empty-frame rate (68.1% / 74.5% /
  92.1%) and sequence count (8 / 8 / 4), so a naive video-level or random
  train/val split would be unfair or leak near-duplicate frames.

**Train/val split** (`notebooks/train_val_split.ipynb` → `data/splits.csv`):
- Split at the **sequence level, separately per video**, then combined —
  ensures every video is represented in both train and val, keeps each
  video's empty/non-empty ratio consistent across the split, and prevents
  leakage (no sequence's frames appear in both splits).
- Output: `data/splits.csv` (`image_id, video_id, sequence, split`) — the
  canonical split membership all training code should read from.

**Usage:**
```python
train_csv = pd.read_csv('train.csv')
train_csv['annotations'] = train_csv['annotations'].apply(ast.literal_eval)
splits = pd.read_csv('data/splits.csv')
full = train_csv.merge(splits[['image_id', 'split']], on='image_id')
train_data = full[full['split'] == 'train']
val_data = full[full['split'] == 'val']
```
