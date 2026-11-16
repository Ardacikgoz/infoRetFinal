from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABEL_COLUMNS, RANDOM_SEED, TEXT_COLUMN  # noqa: E402
from src.data import load_train_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect train.csv for this project.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, targets = load_train_data(args.sample_size, args.seed)

    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    print(f"empty_comments: {(df[TEXT_COLUMN].str.len() == 0).sum()}")
    print("label_counts:")
    for label, count in zip(LABEL_COLUMNS, targets.sum(axis=0), strict=True):
        print(f"  {label}: {int(count)}")
    print(f"rows_with_any_label: {int((targets.sum(axis=1) > 0).sum())}")


if __name__ == "__main__":
    main()
