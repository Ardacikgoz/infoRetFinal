# Toxic Comment Classification: Fine-tuned DistilBERT vs. Local LLM Prompting

A controlled comparison of two approaches to **multi-label toxicity classification** on the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset:

1. **Supervised fine-tuning** of DistilBERT
2. **Prompting a small LLM running locally** (via Ollama), zero-shot and few-shot

Labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. A comment can have any combination of these labels.

📄 **Report:** [report/report.pdf](report/report.pdf)

## Experimental protocol

- A **32,000-comment subsample** of `train.csv`, with label rates nearly identical to the full dataset
- **5-fold cross-validation**, using the **same folds for every model**, so all models are scored on exactly the same held-out comments
- In each fold: 6,400 test comments, and a training pool of 25,600 comments from which the validation split is taken (never from the test fold)
- DistilBERT: per-fold training; a global decision threshold tuned on validation data (0.05–0.95) by **macro F1**, since rare labels such as `threat` and `identity_hate` would otherwise be ignored
- LLM: same test folds; few-shot examples come from a separate, label-balanced pool that does not overlap with the 32k comments
- Runs fully **locally on CPU**, with no external APIs

## Results (5-fold CV, mean ± std)

| Model / setting | Macro F1 | Micro F1 |
|---|---|---|
| DistilBERT, 1 epoch | 0.499 ± 0.018 | 0.692 ± 0.035 |
| **DistilBERT, 5 epochs** | **0.616 ± 0.055** | **0.752 ± 0.008** |
| DistilBERT, 10 epochs | 0.604 ± 0.042 | 0.748 ± 0.012 |
| Qwen2.5-3B, zero-shot | 0.333 ± 0.015 | 0.429 ± 0.005 |
| Qwen2.5-3B, few-shot (k=3) | 0.421 ± 0.025 | 0.489 ± 0.011 |
| Qwen2.5-3B, few-shot (k=5) | 0.412 ± 0.013 | 0.491 ± 0.011 |

**Findings:**
- Fine-tuned DistilBERT clearly beats local LLM prompting. 5 epochs worked best; training for 10 epochs gave no further gain.
- Few-shot prompting improves the LLM a lot over zero-shot (macro F1 0.33 → 0.42).
- Going from k=3 to k=5 examples does not help further, which suggests that the quality and label coverage of the examples matter more than their number.

## Repository structure

```
src/
  config.py        # paths, labels, hyper-parameters
  data.py          # loading & validation of train.csv
  folds.py         # 5-fold split (shared by all models)
  distilbert.py    # multi-label DistilBERT training / inference
  llm_local.py     # Ollama prompting, output parsing & repair
  evaluation.py    # per-label P/R/F1, macro & micro F1, threshold search
scripts/
  inspect_data.py, make_folds.py, train_distilbert.py, eval_local_llm.py
  download_model.py                 # caches distilbert-base-uncased
  run_*.sh                          # batch runs for several folds
results/           # per-fold metrics and summaries (predictions not included)
report/report.pdf
```

## Running

```bash
pip install -r requirements.txt
# download train.csv from Kaggle into the project root
python scripts/make_folds.py --sample-size 32000
python scripts/train_distilbert.py --fold 0 --epochs 5
# LLM baseline (requires Ollama running locally)
ollama pull qwen2.5:3b
python scripts/eval_local_llm.py --fold 0 --mode few-shot --model-name qwen2.5:3b --use-http
```

---
Final project for the Information Retrieval course at Hacettepe University, Department of Artificial Intelligence Engineering.
