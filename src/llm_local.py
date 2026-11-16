from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass

import pandas as pd

from src.config import LABEL_COLUMNS, TEXT_COLUMN


SYSTEM_PROMPT = (
    "You are a strict multi-label toxicity classifier.\n"
    "Return ONLY one JSON object. Do not add explanations, markdown, or extra text.\n"
    "The JSON object must contain exactly these keys and no others:\n"
    '["toxic","severe_toxic","obscene","threat","insult","identity_hate"]\n'
    "Each value must be exactly 0 or 1."
)


@dataclass
class LLMConfig:
    model_name: str
    mode: str = "zero-shot"
    few_shot_k: int = 3
    temperature: float = 0.0
    use_http: bool = False
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    timeout_seconds: int = 120
    num_predict: int = 80


def make_few_shot_examples(train_frame: pd.DataFrame, k: int) -> pd.DataFrame:
    if k < 3:
        raise ValueError("few_shot_k must be at least 3")
    if len(train_frame) < k:
        raise ValueError("Not enough training rows for requested few-shot examples")

    signatures = train_frame[LABEL_COLUMNS].astype(str).agg("".join, axis=1)
    unique = []
    used = set()
    for index, signature in signatures.items():
        if signature in used:
            continue
        unique.append(index)
        used.add(signature)
        if len(unique) == k:
            break

    if len(unique) < k:
        remaining = [idx for idx in train_frame.index if idx not in unique]
        unique.extend(remaining[: k - len(unique)])

    return train_frame.loc[unique].reset_index(drop=True)


def build_user_prompt(text: str, few_shot_examples: pd.DataFrame | None = None) -> str:
    lines = []
    if few_shot_examples is not None and len(few_shot_examples) > 0:
        lines.append("Examples:")
        for _, row in few_shot_examples.iterrows():
            label_payload = {label: int(row[label]) for label in LABEL_COLUMNS}
            lines.append(f"Text: {row[TEXT_COLUMN]}")
            lines.append(f"Labels: {json.dumps(label_payload, separators=(',', ':'))}")
        lines.append("")

    lines.append("Classify the following text.")
    lines.append("Text:")
    lines.append(text)
    lines.append("")
    lines.append(
        'Output exactly one JSON object in this schema: '
        '{"toxic":0,"severe_toxic":0,"obscene":0,"threat":0,"insult":0,"identity_hate":0}'
    )
    lines.append("Use only 0 or 1. No extra keys. No extra text.")
    return "\n".join(lines)


def call_ollama_cli(prompt: str, config: LLMConfig) -> str:
    command = [
        "ollama",
        "run",
        config.model_name,
        prompt,
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=config.timeout_seconds,
    )
    return result.stdout.strip()


def call_ollama_http(prompt: str, config: LLMConfig) -> str:
    payload = {
        "model": config.model_name,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.num_predict,
        },
    }
    request = urllib.request.Request(
        config.ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to call Ollama HTTP endpoint: {exc}") from exc

    parsed = json.loads(body)
    return str(parsed.get("response", "")).strip()


def infer_one(text: str, config: LLMConfig, few_shot_examples: pd.DataFrame | None = None) -> str:
    prompt = build_user_prompt(text, few_shot_examples)
    if config.use_http:
        return call_ollama_http(prompt, config)

    # CLI path does not pass a separate system field, so include strict rules inline.
    cli_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    try:
        return call_ollama_cli(cli_prompt, config)
    except Exception:
        return call_ollama_http(prompt, config)


def _extract_json_object(raw_output: str) -> dict:
    raw_output = raw_output.strip()
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in model output")
    fragment = raw_output[start : end + 1]
    return json.loads(fragment)


def parse_prediction(raw_output: str) -> tuple[dict[str, int], bool]:
    """Parse model output into 0/1 labels.

    Returns:
        (predictions, used_fallback_parser)
    """
    try:
        payload = _extract_json_object(raw_output)
        parsed = {}
        for label in LABEL_COLUMNS:
            value = payload.get(label, 0)
            if isinstance(value, bool):
                parsed[label] = int(value)
            elif isinstance(value, (int, float)):
                parsed[label] = int(value >= 0.5)
            elif isinstance(value, str):
                lowered = value.strip().lower()
                parsed[label] = 1 if lowered in {"1", "true", "yes"} else 0
            else:
                parsed[label] = 0
        return parsed, False
    except Exception:
        # Fallback parser: extract per-label values from loosely formatted text.
        lowered = raw_output.lower()
        parsed = {}
        for label in LABEL_COLUMNS:
            pattern = rf'"?{re.escape(label.lower())}"?\s*[:=]\s*("?)(0|1|true|false|yes|no)\1'
            match = re.search(pattern, lowered)
            if not match:
                parsed[label] = 0
                continue
            token = match.group(2)
            parsed[label] = 1 if token in {"1", "true", "yes"} else 0
        return parsed, True


def default_zero_prediction() -> dict[str, int]:
    return {label: 0 for label in LABEL_COLUMNS}
