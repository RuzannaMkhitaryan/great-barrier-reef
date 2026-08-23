"""
metrics.py - derived metrics not computed natively by Ultralytics
Great Barrier Reef COTS Detection Project
"""

import pandas as pd


def compute_f2(precision, recall):
    """F2 score - weights recall 4x more than precision, appropriate here
    since missed detections (recall) matter more than false positives for
    the competition/project goal."""
    if precision + recall == 0:
        return 0.0
    return 5 * (precision * recall) / (4 * precision + recall)


def f2_from_results_csv(results_csv_path, epoch=None):
    """
    Compute F2 from a training run's results.csv.
    epoch=None uses the last (final/best-saved) row; pass an int to check
    a specific epoch instead.
    """
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()
    row = df.iloc[epoch] if epoch is not None else df.iloc[-1]
    precision = row["metrics/precision(B)"]
    recall = row["metrics/recall(B)"]
    f2 = compute_f2(precision, recall)
    print(f"P={precision:.3f}  R={recall:.3f}  F2={f2:.3f}")
    return f2