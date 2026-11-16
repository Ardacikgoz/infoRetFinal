from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FOLDS_PATH, ID_COLUMN, LABEL_COLUMNS, REQUIRED_COLUMNS, TEXT_COLUMN, TRAIN_PATH


def load_train_dataframe(path=TRAIN_PATH) -> pd.DataFrame:
    """Load train.csv and validate the columns used by this project."""
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[LABEL_COLUMNS] = df[LABEL_COLUMNS].astype(np.int8)
    return df


def get_targets(df: pd.DataFrame) -> np.ndarray:
    """Return the multi-label target matrix with shape (rows, 6)."""
    missing = [column for column in LABEL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    return df[LABEL_COLUMNS].to_numpy(dtype=np.int8)


def subsample_by_label_signature(
    df: pd.DataFrame,
    sample_size: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Optionally sample rows while approximately preserving label combinations.

    The method groups rows by their exact 6-label binary signature, allocates the
    requested sample size proportionally across those groups, and uses largest
    remainders to keep the final size exact when possible.
    """
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive when provided")

    rng = np.random.default_rng(random_state)
    signatures = df[LABEL_COLUMNS].astype(str).agg("".join, axis=1)
    group_indices = {
        signature: group.index.to_numpy()
        for signature, group in df.groupby(signatures, sort=True)
    }

    total_rows = len(df)
    raw_allocations = {
        signature: sample_size * len(indices) / total_rows
        for signature, indices in group_indices.items()
    }
    allocations = {
        signature: min(len(group_indices[signature]), int(np.floor(value)))
        for signature, value in raw_allocations.items()
    }

    remaining = sample_size - sum(allocations.values())
    remainders = sorted(
        raw_allocations,
        key=lambda signature: (
            raw_allocations[signature] - np.floor(raw_allocations[signature]),
            len(group_indices[signature]),
        ),
        reverse=True,
    )
    for signature in remainders:
        if remaining == 0:
            break
        if allocations[signature] < len(group_indices[signature]):
            allocations[signature] += 1
            remaining -= 1

    selected_indices = []
    for signature, indices in group_indices.items():
        count = allocations[signature]
        if count:
            selected_indices.extend(rng.choice(indices, size=count, replace=False))

    rng.shuffle(selected_indices)
    return df.loc[selected_indices].reset_index(drop=True)


def load_train_data(
    sample_size: int | None = None,
    random_state: int = 42,
    path=TRAIN_PATH,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load train.csv, optionally subsample, and return dataframe plus targets."""
    df = load_train_dataframe(path)
    df = subsample_by_label_signature(df, sample_size, random_state)
    targets = get_targets(df)
    return df, targets


def load_active_split(train_path=TRAIN_PATH, folds_path=FOLDS_PATH) -> pd.DataFrame:
    """Load train.csv rows that are present in the saved official fold file."""
    df = load_train_dataframe(train_path)
    folds = pd.read_csv(folds_path)
    required_fold_columns = {ID_COLUMN, "fold"}
    missing = required_fold_columns - set(folds.columns)
    if missing:
        raise ValueError(f"Missing fold columns: {sorted(missing)}")

    active = folds.merge(df, on=ID_COLUMN, how="left", validate="one_to_one")
    if active[TEXT_COLUMN].isna().any():
        raise ValueError("Some fold ids were not found in train.csv")

    active["fold"] = active["fold"].astype(np.int8)
    active[LABEL_COLUMNS] = active[LABEL_COLUMNS].astype(np.int8)
    return active
