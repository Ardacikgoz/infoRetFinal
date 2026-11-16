from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DISTILBERT_BATCH_SIZE,
    DISTILBERT_EPOCHS,
    DISTILBERT_LEARNING_RATE,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_MODEL_NAME,
    DISTILBERT_OUTPUT_DIR,
    FOLDS_PATH,
    DISTILBERT_TOKENIZER_NAME,
    LABEL_COLUMNS,
    N_FOLDS,
    RANDOM_SEED,
    VALIDATION_FRACTION,
)
from src.data import load_active_split  # noqa: E402
from src.distilbert import TrainConfig, train_one_fold, tune_global_threshold  # noqa: E402
from src.evaluation import multilabel_metrics  # noqa: E402


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"{self.name} started", flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.perf_counter() - self.start
        print(f"{self.name} finished in {elapsed:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPU-friendly DistilBERT folds.")
    parser.add_argument("--fold", type=int, default=None, help="Train one fold only.")
    parser.add_argument("--epochs", type=int, default=DISTILBERT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DISTILBERT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DISTILBERT_MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=DISTILBERT_LEARNING_RATE)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed threshold used only with --no-threshold-search.",
    )
    parser.add_argument(
        "--no-threshold-search",
        action="store_true",
        help="Use --threshold directly instead of tuning on validation probabilities.",
    )
    parser.add_argument("--threshold-search-start", type=float, default=0.05)
    parser.add_argument("--threshold-search-end", type=float, default=0.95)
    parser.add_argument("--threshold-search-step", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--model-name", default=DISTILBERT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DISTILBERT_TOKENIZER_NAME)
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=FOLDS_PATH,
        help="Fold assignments CSV to use for the active split.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. If omitted, uses default DistilBERT output directory.",
    )
    return parser.parse_args()


def probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"{label}_prob" for label in LABEL_COLUMNS]].to_numpy(dtype=np.float32)


def add_prediction_columns(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = frame.copy()
    for label in LABEL_COLUMNS:
        output[f"{label}_pred"] = (output[f"{label}_prob"] >= threshold).astype("int8")
    return output


def _sanitize_float_for_tag(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def main() -> None:
    args = parse_args()
    with Timer("loading data"):
        active = load_active_split(folds_path=args.folds_path)

    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    invalid = sorted(set(folds) - set(active["fold"].unique()))
    if invalid:
        raise ValueError(f"Requested folds not present in outputs/folds.csv: {invalid}")

    config = TrainConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        threshold=args.threshold,
        validation_fraction=args.validation_fraction,
        local_files_only=True,
    )

    output_dir = Path(args.output_dir) if args.output_dir else DISTILBERT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    run_tag = (
        f"e{config.epochs}"
        f"_bs{config.batch_size}"
        f"_lr{_sanitize_float_for_tag(config.learning_rate)}"
        f"_ml{config.max_length}"
    )

    print(f"active_rows: {len(active)}")
    print("label_counts:")
    for label in LABEL_COLUMNS:
        print(f"  {label}: {int(active[label].sum())}")

    for fold in folds:
        print(f"training_fold: {fold}")
        (
            validation_probabilities,
            test_probabilities,
            split_summary,
        ) = train_one_fold(active, fold, config)
        validation_path = output_dir / f"fold_{fold}_{run_tag}_validation_probabilities.csv"
        test_probabilities_path = output_dir / f"fold_{fold}_{run_tag}_test_probabilities.csv"
        predictions_path = output_dir / f"fold_{fold}_{run_tag}_predictions.csv"
        threshold_path = output_dir / f"fold_{fold}_{run_tag}_threshold_search.csv"
        metrics_path = output_dir / f"fold_{fold}_{run_tag}_metrics.csv"

        print(f"fold {fold} saving validation/test probabilities", flush=True)
        validation_probabilities.to_csv(validation_path, index=False)
        test_probabilities.to_csv(test_probabilities_path, index=False)

        print(f"fold {fold} threshold search start", flush=True)
        saved_validation = pd.read_csv(validation_path)
        if args.no_threshold_search:
            print("threshold search skipped; using fixed threshold", flush=True)
            selected_threshold = args.threshold
            _, validation_summary = multilabel_metrics(
                saved_validation[LABEL_COLUMNS].to_numpy(dtype=np.int8),
                probability_matrix(saved_validation),
                selected_threshold,
            )
            threshold_results = pd.DataFrame([validation_summary])
        else:
            with Timer(f"fold {fold} threshold search from saved validation probabilities"):
                threshold_grid = np.round(
                    np.arange(
                        args.threshold_search_start,
                        args.threshold_search_end + args.threshold_search_step / 2,
                        args.threshold_search_step,
                    ),
                    4,
                )
                selected_threshold, threshold_results = tune_global_threshold(
                    saved_validation[LABEL_COLUMNS].to_numpy(dtype=np.int8),
                    probability_matrix(saved_validation),
                    threshold_grid,
                )
            print(
                f"fold {fold} threshold search end: selected_threshold={selected_threshold:.4f}",
                flush=True,
            )

        print(f"fold {fold} test evaluation start", flush=True)
        with Timer(f"fold {fold} test evaluation from saved probabilities"):
            saved_test = pd.read_csv(test_probabilities_path)
            per_label, summary = multilabel_metrics(
                saved_test[LABEL_COLUMNS].to_numpy(dtype=np.int8),
                probability_matrix(saved_test),
                selected_threshold,
            )
            summary.update(split_summary)
            test_predictions = add_prediction_columns(saved_test, selected_threshold)
        print(f"fold {fold} test evaluation end", flush=True)

        test_predictions.to_csv(predictions_path, index=False)
        threshold_results.assign(fold=fold).to_csv(threshold_path, index=False)
        per_label.assign(
            fold=fold,
            macro_f1=summary["macro_f1"],
            micro_f1=summary["micro_f1"],
            threshold=summary["threshold"],
            train_rows=summary["train_rows"],
            validation_rows=summary["validation_rows"],
            test_rows=summary["test_rows"],
        ).to_csv(metrics_path, index=False)
        summary_rows.append(summary)
        print(
            f"fold {fold}: selected_threshold={summary['threshold']:.4f} "
            f"macro_f1={summary['macro_f1']:.4f} micro_f1={summary['micro_f1']:.4f}"
        )
        print(f"saved_validation_probabilities: {validation_path}")
        print(f"saved_test_probabilities: {test_probabilities_path}")
        print(f"saved_test_predictions: {predictions_path}")
        print(f"saved_threshold_search: {threshold_path}")
        print(f"saved_metrics: {metrics_path}")

    summary_path = output_dir / f"fold_summary_{run_tag}.csv"
    new_summary = pd.DataFrame(summary_rows)
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        combined_summary = pd.concat([existing_summary, new_summary], ignore_index=True)
        combined_summary = (
            combined_summary.sort_values("fold")
            .drop_duplicates(subset=["fold"], keep="last")
            .reset_index(drop=True)
        )
    else:
        combined_summary = new_summary

    combined_summary.to_csv(summary_path, index=False)
    print(f"saved_summary: {summary_path}")


if __name__ == "__main__":
    main()
