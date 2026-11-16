from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FOLDS_PATH, ID_COLUMN, LABEL_COLUMNS, N_FOLDS


def make_multilabel_folds(
    df: pd.DataFrame,
    targets: np.ndarray,
    n_folds: int = N_FOLDS,
    random_state: int = 42,
) -> np.ndarray:
    """Create reproducible fold ids with a lightweight multi-label balancer."""
    if len(df) != len(targets):
        raise ValueError("df and targets must have the same number of rows")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(df) < n_folds:
        raise ValueError("Number of rows must be at least n_folds")

    rng = np.random.default_rng(random_state)
    folds = np.full(len(df), -1, dtype=np.int8)
    fold_sizes = np.zeros(n_folds, dtype=np.int32)
    fold_label_counts = np.zeros((n_folds, targets.shape[1]), dtype=np.int32)
    label_totals = targets.sum(axis=0)

    # Sort positive rows first, prioritizing examples containing rarer labels.
    rare_label_weight = np.where(label_totals > 0, 1.0 / label_totals, 0.0)
    row_weights = targets @ rare_label_weight
    positive_counts = targets.sum(axis=1)
    tie_breakers = rng.random(len(df))
    order = np.lexsort((tie_breakers, -row_weights, -positive_counts))

    for row_index in order:
        labels = np.flatnonzero(targets[row_index])
        if len(labels) == 0:
            chosen_fold = int(np.argmin(fold_sizes))
        else:
            label_pressure = fold_label_counts[:, labels].sum(axis=1)
            candidates = np.flatnonzero(label_pressure == label_pressure.min())
            candidate_sizes = fold_sizes[candidates]
            chosen_fold = int(candidates[np.argmin(candidate_sizes)])

        folds[row_index] = chosen_fold
        fold_sizes[chosen_fold] += 1
        fold_label_counts[chosen_fold] += targets[row_index]

    return folds


def build_fold_assignments(
    df: pd.DataFrame,
    targets: np.ndarray,
    n_folds: int = N_FOLDS,
    random_state: int = 42,
) -> pd.DataFrame:
    folds = make_multilabel_folds(df, targets, n_folds, random_state)
    assignments = df[[ID_COLUMN]].copy()
    assignments["fold"] = folds
    return assignments


def save_fold_assignments(assignments: pd.DataFrame, path=FOLDS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(path, index=False)


def summarize_folds(assignments: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    merged = assignments.merge(df[[ID_COLUMN, *LABEL_COLUMNS]], on=ID_COLUMN, how="left")
    summary = merged.groupby("fold")[LABEL_COLUMNS].sum()
    summary.insert(0, "rows", merged.groupby("fold").size())
    return summary
