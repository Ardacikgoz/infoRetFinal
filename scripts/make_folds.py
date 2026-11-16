from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FOLDS_PATH, N_FOLDS, RANDOM_SEED  # noqa: E402
from src.data import load_train_data  # noqa: E402
from src.folds import build_fold_assignments, save_fold_assignments, summarize_folds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible train.csv fold ids.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=FOLDS_PATH,
        help="Where to save the generated fold assignments CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, targets = load_train_data(args.sample_size, args.seed)
    assignments = build_fold_assignments(df, targets, args.n_folds, args.seed)
    save_fold_assignments(assignments, args.output_path)

    print(f"saved: {args.output_path}")
    print(f"rows: {len(assignments)}")
    print("fold_summary:")
    print(summarize_folds(assignments, df).to_string())


if __name__ == "__main__":
    main()
