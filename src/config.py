from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FOLDS_PATH = OUTPUT_DIR / "folds.csv"
DISTILBERT_OUTPUT_DIR = OUTPUT_DIR / "distilbert"
LLM_OUTPUT_DIR = OUTPUT_DIR / "llm_phi3_mini"

ID_COLUMN = "id"
TEXT_COLUMN = "comment_text"
LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
REQUIRED_COLUMNS = [ID_COLUMN, TEXT_COLUMN, *LABEL_COLUMNS]

N_FOLDS = 5
RANDOM_SEED = 42

# Keep None by default so scripts are explicit about reduced runs.
DEFAULT_SAMPLE_SIZE = None

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
DISTILBERT_TOKENIZER_NAME = "bert-base-uncased"
DISTILBERT_MAX_LENGTH = 128
DISTILBERT_BATCH_SIZE = 8
DISTILBERT_EPOCHS = 1
DISTILBERT_LEARNING_RATE = 2e-5
VALIDATION_FRACTION = 0.1

OLLAMA_MODEL_NAME = "phi3:mini"
