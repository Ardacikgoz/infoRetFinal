from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FOLDS_PATH, ID_COLUMN, LABEL_COLUMNS, LLM_OUTPUT_DIR, OLLAMA_MODEL_NAME, TEXT_COLUMN  # noqa: E402
from src.data import load_active_split  # noqa: E402
from src.evaluation import multilabel_metrics  # noqa: E402
from src.llm_local import (  # noqa: E402
    LLMConfig,
    default_zero_prediction,
    infer_one,
    make_few_shot_examples,
    parse_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local Ollama LLM on one held-out fold.")
    parser.add_argument("--fold", type=int, default=0, help="Held-out fold to evaluate.")
    parser.add_argument(
        "--mode",
        choices=["zero-shot", "few-shot"],
        default="zero-shot",
        help="Prompting mode.",
    )
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=3,
        help="Number of in-context examples for few-shot mode (must be >= 3).",
    )
    parser.add_argument(
        "--few-shot-source-fold",
        type=int,
        default=None,
        help=(
            "Optional fixed fold id to draw few-shot examples from. "
            "Helps keep few-shot prompts more consistent across evaluation folds."
        ),
    )
    parser.add_argument("--model-name", default=OLLAMA_MODEL_NAME)
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=FOLDS_PATH,
        help="Fold assignments CSV to use for the active split.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. If omitted, uses default LLM output directory.",
    )
    parser.add_argument("--use-http", action="store_true")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for number of test rows to evaluate (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--sample-chunk-size",
        type=int,
        default=None,
        help=(
            "Optional checkpoint chunk size (e.g. 500). "
            "When set, saves per-chunk intermediate outputs under output_dir/sample_runs."
        ),
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=60,
        help="Maximum generated tokens per sample for Ollama HTTP calls.",
    )
    parser.add_argument(
        "--repair-parse-errors",
        action="store_true",
        help="After main pass, retry parse_error rows with stricter generation settings.",
    )
    parser.add_argument(
        "--repair-num-predict",
        type=int,
        default=40,
        help="num_predict used in parse-error repair retries.",
    )
    parser.add_argument(
        "--repair-timeout-seconds",
        type=int,
        default=180,
        help="Timeout used in parse-error repair retries.",
    )
    parser.add_argument(
        "--repair-max-retries",
        type=int,
        default=1,
        help="Max retry attempts per parse_error row in repair pass.",
    )
    parser.add_argument(
        "--max-repair-parse-errors",
        type=int,
        default=100,
        help="Skip repair pass if parse_error count is above this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "few-shot" and args.few_shot_k < 3:
        raise ValueError("--few-shot-k must be at least 3")
    if args.repair_max_retries <= 0:
        raise ValueError("--repair-max-retries must be positive")
    if args.max_repair_parse_errors < 0:
        raise ValueError("--max-repair-parse-errors cannot be negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if args.sample_chunk_size is not None and args.sample_chunk_size <= 0:
        raise ValueError("--sample-chunk-size must be positive when provided")

    active = load_active_split(folds_path=args.folds_path)
    available_folds = set(active["fold"].unique())
    if args.fold not in available_folds:
        raise ValueError(f"Fold {args.fold} not in outputs/folds.csv; available={sorted(available_folds)}")

    train_frame = active[active["fold"] != args.fold].reset_index(drop=True)
    test_frame = active[active["fold"] == args.fold].reset_index(drop=True)
    if args.max_samples is not None:
        test_frame = test_frame.head(args.max_samples).reset_index(drop=True)

    few_shot_examples = None
    if args.mode == "few-shot":
        if args.few_shot_source_fold is not None:
            source_frame = active[active["fold"] == args.few_shot_source_fold].reset_index(drop=True)
            if len(source_frame) < args.few_shot_k:
                raise ValueError(
                    f"--few-shot-source-fold {args.few_shot_source_fold} does not have "
                    f"enough rows for few-shot-k={args.few_shot_k}"
                )
            few_shot_examples = make_few_shot_examples(source_frame, args.few_shot_k)
            print(
                f"few-shot source: fixed fold {args.few_shot_source_fold} "
                f"rows={len(source_frame)} selected={len(few_shot_examples)}",
                flush=True,
            )
        else:
            few_shot_examples = make_few_shot_examples(train_frame, args.few_shot_k)
            print(
                f"few-shot source: train split rows={len(train_frame)} "
                f"selected={len(few_shot_examples)}",
                flush=True,
            )

    config = LLMConfig(
        model_name=args.model_name,
        mode=args.mode,
        few_shot_k=args.few_shot_k,
        use_http=args.use_http,
        ollama_url=args.ollama_url,
        timeout_seconds=args.timeout_seconds,
        num_predict=args.num_predict,
    )

    output_dir = Path(args.output_dir) if args.output_dir else LLM_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_runs_dir = output_dir / "sample_runs"
    if args.sample_chunk_size is not None:
        sample_runs_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.mode}_np{args.num_predict}_to{args.timeout_seconds}"

    raw_rows = []
    parsed_rows = []
    inference_error_count = 0
    parse_fallback_count = 0
    final_parse_error_count = 0
    parse_error_indices: list[int] = []
    total_rows = len(test_frame)
    started_at = time.perf_counter()
    print(
        f"starting llm evaluation: fold={args.fold} mode={args.mode} "
        f"test_rows={total_rows} model={args.model_name} "
        f"max_samples={args.max_samples}",
        flush=True,
    )
    for index, row in test_frame.iterrows():
        text_value = row[TEXT_COLUMN]
        text_length = len(text_value)
        error_message = ""
        raw_output = ""
        try:
            raw_output = infer_one(
                text=text_value,
                config=config,
                few_shot_examples=few_shot_examples,
            )
        except Exception as exc:
            error_message = f"inference_error: {exc}"
        raw_rows.append(
            {
                "id": row[ID_COLUMN],
                "fold": int(args.fold),
                "mode": args.mode,
                "model_name": args.model_name,
                "raw_output": raw_output,
                "error_type": "",
                "error": error_message,
            }
        )

        if error_message:
            inference_error_count += 1
            raw_rows[-1]["error_type"] = "inference_error"
            parsed = default_zero_prediction()
        else:
            try:
                parsed, used_fallback = parse_prediction(raw_output)
                if used_fallback:
                    parse_fallback_count += 1
                    parse_error_indices.append(index)
                    raw_rows[-1]["error_type"] = "parse_error"
                    raw_rows[-1]["error"] = "parse_error: fallback_parser_used"
            except Exception as exc:
                parsed = default_zero_prediction()
                parse_fallback_count += 1
                parse_error_indices.append(index)
                raw_rows[-1]["error_type"] = "parse_error"
                raw_rows[-1]["error"] = f"parse_error: {exc}"
        parsed_rows.append(
            {
                "id": row[ID_COLUMN],
                "fold": int(args.fold),
                **{label: int(row[label]) for label in LABEL_COLUMNS},
                **{f"{label}_pred": int(parsed[label]) for label in LABEL_COLUMNS},
            }
        )
        completed = index + 1
        elapsed = time.perf_counter() - started_at
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = total_rows - completed
        eta_seconds = remaining / rate if rate > 0 else 0.0
        status = raw_rows[-1]["error_type"] if raw_rows[-1]["error_type"] else "ok"
        pred_summary = " ".join(f"{label}={int(parsed[label])}" for label in LABEL_COLUMNS)
        print(
            f"progress: {completed}/{total_rows} "
            f"({(completed / total_rows) * 100:.1f}%) "
            f"text_len={text_length} "
            f"status={status} "
            f"{pred_summary} "
            f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s",
            flush=True,
        )

        if args.sample_chunk_size is not None and (
            completed % args.sample_chunk_size == 0 or completed == total_rows
        ):
            chunk_start = completed - (completed % args.sample_chunk_size)
            if chunk_start == completed:
                chunk_start = completed - args.sample_chunk_size
            chunk_start = max(chunk_start, 0)
            chunk_end = completed
            chunk_raw_df = pd.DataFrame(raw_rows[chunk_start:chunk_end])
            chunk_parsed_df = pd.DataFrame(parsed_rows[chunk_start:chunk_end])
            chunk_y_true = chunk_parsed_df[LABEL_COLUMNS].to_numpy(dtype=np.int8)
            chunk_y_pred = chunk_parsed_df[[f"{label}_pred" for label in LABEL_COLUMNS]].to_numpy(dtype=np.int8)
            chunk_per_label, chunk_summary = multilabel_metrics(
                chunk_y_true,
                chunk_y_pred.astype(np.float32),
                threshold=0.5,
            )
            chunk_summary.update(
                {
                    "fold": int(args.fold),
                    "mode": args.mode,
                    "model_name": args.model_name,
                    "chunk_start_index": int(chunk_start),
                    "chunk_end_index_exclusive": int(chunk_end),
                    "chunk_size": int(chunk_end - chunk_start),
                }
            )
            chunk_base = (
                f"fold_{args.fold}_{run_tag}_chunk_{chunk_start}_{chunk_end}"
            )
            chunk_raw_df.to_csv(sample_runs_dir / f"{chunk_base}_raw_outputs.csv", index=False)
            chunk_parsed_df.to_csv(sample_runs_dir / f"{chunk_base}_predictions.csv", index=False)
            chunk_per_label.assign(
                fold=args.fold,
                mode=args.mode,
                model_name=args.model_name,
                macro_f1=chunk_summary["macro_f1"],
                micro_f1=chunk_summary["micro_f1"],
                chunk_start_index=chunk_start,
                chunk_end_index_exclusive=chunk_end,
            ).to_csv(sample_runs_dir / f"{chunk_base}_metrics.csv", index=False)
            (sample_runs_dir / f"{chunk_base}_summary.json").write_text(
                json.dumps(chunk_summary, indent=2),
                encoding="utf-8",
            )
            print(
                f"saved_chunk: {chunk_base} "
                f"macro_f1={chunk_summary['macro_f1']:.4f} "
                f"micro_f1={chunk_summary['micro_f1']:.4f}",
                flush=True,
            )

    if args.repair_parse_errors and parse_error_indices:
        if len(parse_error_indices) > args.max_repair_parse_errors:
            final_parse_error_count = len(parse_error_indices)
            for row_index in parse_error_indices:
                raw_rows[row_index]["error_type"] = "final_parse_error"
                if not raw_rows[row_index]["error"]:
                    raw_rows[row_index]["error"] = "final_parse_error: skipped_due_to_threshold"
            print(
                "repair pass skipped: "
                f"parse_error_rows={len(parse_error_indices)} exceeds "
                f"max_repair_parse_errors={args.max_repair_parse_errors}",
                flush=True,
            )
        else:
            print(
                f"repair pass started: parse_error_rows={len(parse_error_indices)} "
                f"retries={args.repair_max_retries}",
                flush=True,
            )
            repair_config = LLMConfig(
                model_name=args.model_name,
                mode=args.mode,
                few_shot_k=args.few_shot_k,
                use_http=True,
                ollama_url=args.ollama_url,
                timeout_seconds=args.repair_timeout_seconds,
                num_predict=args.repair_num_predict,
            )
            repaired = 0
            unresolved = 0
            repair_attempts_total = 0
            for row_index in parse_error_indices:
                row = test_frame.iloc[row_index]
                success = False
                for attempt in range(1, args.repair_max_retries + 1):
                    repair_attempts_total += 1
                    try:
                        repair_raw = infer_one(
                            text=row[TEXT_COLUMN],
                            config=repair_config,
                            few_shot_examples=few_shot_examples,
                        )
                        repair_parsed, repair_used_fallback = parse_prediction(repair_raw)
                        if repair_used_fallback:
                            continue

                        raw_rows[row_index]["raw_output"] = repair_raw
                        raw_rows[row_index]["error_type"] = ""
                        raw_rows[row_index]["error"] = ""
                        for label in LABEL_COLUMNS:
                            parsed_rows[row_index][f"{label}_pred"] = int(repair_parsed[label])
                        success = True
                        repaired += 1
                        break
                    except Exception:
                        continue

                if not success:
                    unresolved += 1
                    raw_rows[row_index]["error_type"] = "final_parse_error"
                    if not raw_rows[row_index]["error"]:
                        raw_rows[row_index]["error"] = "final_parse_error: unresolved_after_repair"

            final_parse_error_count = unresolved
            print(
                f"repair pass finished: repaired={repaired} unresolved={unresolved} "
                f"repair_attempts_total={repair_attempts_total}",
                flush=True,
            )
    elif parse_error_indices:
        final_parse_error_count = len(parse_error_indices)
        for row_index in parse_error_indices:
            raw_rows[row_index]["error_type"] = "final_parse_error"
            if not raw_rows[row_index]["error"]:
                raw_rows[row_index]["error"] = "final_parse_error: repair_disabled"

    raw_df = pd.DataFrame(raw_rows)
    parsed_df = pd.DataFrame(parsed_rows)
    for label in LABEL_COLUMNS:
        parsed_df[f"{label}_prob"] = parsed_df[f"{label}_pred"].astype(np.float32)

    y_true = parsed_df[LABEL_COLUMNS].to_numpy(dtype=np.int8)
    y_pred = parsed_df[[f"{label}_pred" for label in LABEL_COLUMNS]].to_numpy(dtype=np.int8)
    per_label, summary = multilabel_metrics(y_true, y_pred.astype(np.float32), threshold=0.5)
    summary.update(
        {
            "fold": int(args.fold),
            "mode": args.mode,
            "model_name": args.model_name,
            "test_rows": int(len(test_frame)),
            "inference_errors": int(inference_error_count),
            "parse_fallbacks": int(parse_fallback_count),
            "final_parse_errors": int(final_parse_error_count),
            "total_errors": int(inference_error_count + final_parse_error_count),
            "error_rate": float((inference_error_count + final_parse_error_count) / len(test_frame)),
            "repair_enabled": bool(args.repair_parse_errors),
            "repair_num_predict": int(args.repair_num_predict),
            "repair_timeout_seconds": int(args.repair_timeout_seconds),
            "repair_max_retries": int(args.repair_max_retries),
            "max_repair_parse_errors": int(args.max_repair_parse_errors),
            "few_shot_source_fold": args.few_shot_source_fold,
        }
    )

    raw_path = output_dir / f"fold_{args.fold}_{run_tag}_raw_outputs.csv"
    parsed_path = output_dir / f"fold_{args.fold}_{run_tag}_parsed_predictions.csv"
    metrics_path = output_dir / f"fold_{args.fold}_{run_tag}_metrics.csv"
    summary_path = output_dir / f"fold_{args.fold}_{run_tag}_summary.json"
    predictions_path = output_dir / f"fold_{args.fold}_{run_tag}_predictions.csv"
    test_probabilities_path = output_dir / f"fold_{args.fold}_{run_tag}_test_probabilities.csv"
    legacy_raw_path = output_dir / f"fold_{args.fold}_{args.mode}_raw_outputs.csv"
    legacy_parsed_path = output_dir / f"fold_{args.fold}_{args.mode}_parsed_predictions.csv"
    legacy_metrics_path = output_dir / f"fold_{args.fold}_{args.mode}_metrics.csv"
    legacy_summary_path = output_dir / f"fold_{args.fold}_{args.mode}_summary.json"

    raw_df.to_csv(raw_path, index=False)
    parsed_df.to_csv(parsed_path, index=False)
    raw_df.to_csv(legacy_raw_path, index=False)
    parsed_df.to_csv(legacy_parsed_path, index=False)
    parsed_df.to_csv(predictions_path, index=False)
    parsed_df.to_csv(test_probabilities_path, index=False)
    per_label.assign(
        fold=args.fold,
        mode=args.mode,
        model_name=args.model_name,
        macro_f1=summary["macro_f1"],
        micro_f1=summary["micro_f1"],
    ).to_csv(metrics_path, index=False)
    per_label.assign(
        fold=args.fold,
        mode=args.mode,
        model_name=args.model_name,
        macro_f1=summary["macro_f1"],
        micro_f1=summary["micro_f1"],
    ).to_csv(legacy_metrics_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    legacy_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved_raw_outputs: {raw_path}")
    print(f"saved_parsed_predictions: {parsed_path}")
    print(f"saved_metrics: {metrics_path}")
    print(f"saved_summary: {summary_path}")
    print(f"saved_predictions: {predictions_path}")
    print(f"saved_test_probabilities: {test_probabilities_path}")
    print(
        "errors: "
        f"inference={inference_error_count} "
        f"fallback={parse_fallback_count} "
        f"final_parse={final_parse_error_count} "
        f"total={inference_error_count + final_parse_error_count}",
        flush=True,
    )
    print(
        f"fold={args.fold} mode={args.mode} macro_f1={summary['macro_f1']:.4f} micro_f1={summary['micro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
