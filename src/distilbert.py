from __future__ import annotations

import random
import os
import tempfile
import time
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from src.config import (
    DISTILBERT_BATCH_SIZE,
    DISTILBERT_LEARNING_RATE,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_MODEL_NAME,
    DISTILBERT_TOKENIZER_NAME,
    LABEL_COLUMNS,
    RANDOM_SEED,
    TEXT_COLUMN,
)
from src.evaluation import multilabel_metrics
from src.folds import make_multilabel_folds


@dataclass
class TrainConfig:
    model_name: str = DISTILBERT_MODEL_NAME
    tokenizer_name: str = DISTILBERT_TOKENIZER_NAME
    max_length: int = DISTILBERT_MAX_LENGTH
    batch_size: int = DISTILBERT_BATCH_SIZE
    epochs: int = 1
    learning_rate: float = DISTILBERT_LEARNING_RATE
    seed: int = RANDOM_SEED
    threshold: float = 0.5
    validation_fraction: float = 0.1
    local_files_only: bool = True


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.texts = frame[TEXT_COLUMN].tolist()
        self.labels = frame[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"text": self.texts[index], "labels": self.labels[index]}


class StepProgress:
    def __init__(self, name: str, total: int, interval: int = 25):
        self.name = name
        self.total = max(total, 1)
        self.interval = max(interval, 1)
        self.start = time.perf_counter()

    def update(self, step: int, loss: float | None = None) -> None:
        current = step + 1
        if current != self.total and current % self.interval:
            return
        elapsed = time.perf_counter() - self.start
        width = 24
        filled = int(width * current / self.total)
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"{self.name}: [{bar}] {current}/{self.total} batches "
            f"elapsed={elapsed:.1f}s"
        )
        if loss is not None:
            message += f" loss={loss:.4f}"
        print(message, flush=True)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor(np.stack([item["labels"] for item in batch]), dtype=torch.float32)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded.pop("token_type_ids", None)
        encoded["labels"] = labels
        return encoded

    return collate


def split_inner_train_validation(
    train_frame: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    targets = train_frame[LABEL_COLUMNS].to_numpy(dtype=np.int8)
    n_inner_folds = max(2, round(1 / validation_fraction))
    inner_folds = make_multilabel_folds(train_frame, targets, n_inner_folds, seed)
    validation_frame = train_frame.loc[inner_folds == 0].reset_index(drop=True)
    inner_train_frame = train_frame.loc[inner_folds != 0].reset_index(drop=True)
    return inner_train_frame, validation_frame


def load_model_and_tokenizer(config: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        local_files_only=config.local_files_only,
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
        local_files_only=config.local_files_only,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


def build_prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float | None = None,
) -> pd.DataFrame:
    prediction_frame = frame[["id", "fold", *LABEL_COLUMNS]].copy()
    for index, label in enumerate(LABEL_COLUMNS):
        prediction_frame[f"{label}_prob"] = probabilities[:, index]
        if threshold is not None:
            prediction_frame[f"{label}_pred"] = (
                probabilities[:, index] >= threshold
            ).astype(np.int8)
    return prediction_frame


def tune_global_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows = []
    best_threshold = float(thresholds[0])
    best_macro_f1 = -1.0
    best_micro_f1 = -1.0

    for threshold in thresholds:
        _, summary = multilabel_metrics(y_true, y_prob, float(threshold))
        rows.append(summary)
        is_better = (
            summary["macro_f1"] > best_macro_f1
            or (
                summary["macro_f1"] == best_macro_f1
                and summary["micro_f1"] > best_micro_f1
            )
        )
        if is_better:
            best_threshold = float(threshold)
            best_macro_f1 = summary["macro_f1"]
            best_micro_f1 = summary["micro_f1"]

    return best_threshold, pd.DataFrame(rows)


def train_one_fold(
    active: pd.DataFrame,
    fold: int,
    config: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    set_seed(config.seed + fold)
    device = torch.device("cpu")

    with Timer(f"fold {fold} data split"):
        outer_train = active[active["fold"] != fold].reset_index(drop=True)
        test_frame = active[active["fold"] == fold].reset_index(drop=True)
        train_frame, validation_frame = split_inner_train_validation(
            outer_train,
            config.validation_fraction,
            config.seed + fold,
        )
        print(
            f"fold {fold} rows: train={len(train_frame)} "
            f"validation={len(validation_frame)} test={len(test_frame)}",
            flush=True,
        )

    with Timer(f"fold {fold} model/tokenizer loading"):
        model, tokenizer = load_model_and_tokenizer(config)
    model.to(device)
    model.train()

    with Timer(f"fold {fold} tokenization setup"):
        collate = make_collate_fn(tokenizer, config.max_length)
        train_loader = DataLoader(
            TextDataset(train_frame),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"fold {fold} training start", flush=True)
    with Timer(f"fold {fold} training"):
        for epoch in range(config.epochs):
            print(f"fold {fold} epoch {epoch + 1}/{config.epochs} start", flush=True)
            progress = StepProgress(
                f"fold {fold} epoch {epoch + 1}",
                len(train_loader),
            )
            for batch_index, batch in enumerate(train_loader):
                labels = batch.pop("labels").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                logits = model(**inputs).logits
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                progress.update(batch_index, float(loss.detach().cpu()))
            print(f"fold {fold} epoch {epoch + 1}/{config.epochs} end", flush=True)
    print(f"fold {fold} training end", flush=True)

    print(f"fold {fold} evaluation start", flush=True)
    with Timer(f"fold {fold} validation inference"):
        validation_probabilities = predict(
            model,
            tokenizer,
            validation_frame,
            config,
            device,
            description=f"fold {fold} validation inference",
        )
    with Timer(f"fold {fold} test inference"):
        test_probabilities = predict(
            model,
            tokenizer,
            test_frame,
            config,
            device,
            description=f"fold {fold} test inference",
        )

    validation_predictions = build_prediction_frame(
        validation_frame,
        validation_probabilities,
    )
    test_predictions = build_prediction_frame(
        test_frame,
        test_probabilities,
    )

    summary = {
        "fold": fold,
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows": len(test_frame),
    }
    print(f"fold {fold} evaluation end", flush=True)

    return validation_predictions, test_predictions, summary


@torch.no_grad()
def predict(
    model,
    tokenizer,
    frame: pd.DataFrame,
    config: TrainConfig,
    device: torch.device,
    description: str = "inference",
) -> np.ndarray:
    model.eval()
    collate = make_collate_fn(tokenizer, config.max_length)
    loader = DataLoader(
        TextDataset(frame),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    probabilities = []
    progress = StepProgress(description, len(loader), interval=25)
    for batch_index, batch in enumerate(loader):
        batch.pop("labels")
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(**inputs).logits
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        progress.update(batch_index)
    return np.concatenate(probabilities, axis=0)
