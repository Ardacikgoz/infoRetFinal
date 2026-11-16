from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMNS


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    y_pred = (y_prob >= threshold).astype(np.int8)
    rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for index, label in enumerate(LABEL_COLUMNS):
        truth = y_true[:, index].astype(bool)
        pred = y_pred[:, index].astype(bool)
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(truth.sum()),
            }
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    per_label = pd.DataFrame(rows)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    summary = {
        "macro_f1": float(per_label["f1"].mean()),
        "micro_f1": float(micro_f1),
        "threshold": threshold,
    }
    return per_label, summary
