"""
Optional transformer-based complaint classifier (DistilBERT/RoBERTa).

WHY THIS IS OPTIONAL IN THIS ENVIRONMENT
-----------------------------------------
Fine-tuning a transformer requires `torch` + `transformers` and downloading
pretrained weights from the Hugging Face Hub. Those downloads are not always
reachable from every deployment/sandbox network. To keep the project 100%
runnable everywhere, the TF-IDF + Logistic Regression baseline
(src/classification/baseline.py) is the classifier actually used by the
running pipeline by default (config.yaml: classification.model_type).

This module implements the SAME architecture/interface a transformer
classifier would use, so that on a machine with GPU/internet access to the
Hub, a user can flip `classification.model_type: transformer` in
config.yaml and get a real fine-tuned DistilBERT classifier with no other
code changes anywhere else in the system (see TransformerClassifier below
and classification/inference.py which picks whichever backend is
configured/available).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.config import settings, get_logger

logger = get_logger(__name__)

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    )
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


DEFAULT_MODEL_NAME = "distilbert-base-uncased"  # small & practical for local fine-tuning


if TRANSFORMERS_AVAILABLE:
    class _ComplaintTextDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item


class TransformerClassifier:
    """Fine-tunes/loads a small transformer for complaint category classification.

    Raises a clear RuntimeError (not a silent failure) if torch/transformers
    are not installed, so callers (train_models.py) can fall back to the
    baseline classifier and log why.
    """

    def __init__(self, target_name: str = "category", model_name: str = DEFAULT_MODEL_NAME):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "torch/transformers are not installed in this environment, so the "
                "transformer classifier cannot run. Falling back to the TF-IDF + "
                "Logistic Regression baseline classifier is expected and supported - "
                "see src/classification/baseline.py. Install `torch` and `transformers` "
                "and ensure Hugging Face Hub access to enable this module."
            )
        self.target_name = target_name
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str], epochs: int = 2, batch_size: int = 16) -> Dict:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        self.label_classes_ = sorted(set(labels))
        label2id = {l: i for i, l in enumerate(self.label_classes_)}
        y = [label2id[l] for l in labels]

        X_train, X_val, y_train, y_val = train_test_split(texts, y, test_size=0.15, random_state=42, stratify=y)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.label_classes_)
        )

        train_enc = self.tokenizer(X_train, truncation=True, padding=True, max_length=128)
        val_enc = self.tokenizer(X_val, truncation=True, padding=True, max_length=128)
        train_ds = _ComplaintTextDataset(train_enc, y_train)
        val_ds = _ComplaintTextDataset(val_enc, y_val)

        args = TrainingArguments(
            output_dir="./_transformer_tmp",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            report_to=[],
        )

        def compute_metrics(eval_pred):
            logits, labels_ = eval_pred
            preds = logits.argmax(-1)
            acc = accuracy_score(labels_, preds)
            _, _, f1, _ = precision_recall_fscore_support(labels_, preds, average="macro", zero_division=0)
            return {"accuracy": acc, "macro_f1": f1}

        trainer = Trainer(model=self.model, args=args, train_dataset=train_ds,
                           eval_dataset=val_ds, compute_metrics=compute_metrics)
        trainer.train()
        metrics = trainer.evaluate()
        logger.info("[%s] transformer trained. metrics=%s", self.target_name, metrics)
        return metrics

    def predict(self, texts: List[str], top_k: int = 3) -> List[Dict]:
        import torch.nn.functional as F
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1).numpy()
        results = []
        for row in probs:
            order = row.argsort()[::-1]
            top = order[:top_k]
            results.append({
                "label": self.label_classes_[top[0]],
                "confidence": float(row[top[0]]),
                "alternatives": [
                    {"label": self.label_classes_[i], "confidence": float(row[i])} for i in top[1:]
                ],
            })
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory / self.target_name)
        self.tokenizer.save_pretrained(directory / self.target_name)
        with open(directory / f"{self.target_name}_labels.json", "w") as f:
            json.dump(self.label_classes_, f)

    @classmethod
    def load(cls, directory: Path, target_name: str = "category") -> "TransformerClassifier":
        obj = cls(target_name=target_name)
        obj.tokenizer = AutoTokenizer.from_pretrained(directory / target_name)
        obj.model = AutoModelForSequenceClassification.from_pretrained(directory / target_name)
        with open(directory / f"{target_name}_labels.json") as f:
            obj.label_classes_ = json.load(f)
        return obj
