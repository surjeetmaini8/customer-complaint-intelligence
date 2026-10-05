"""
Trains the classification models (category + per-category subcategory)
using the baseline TF-IDF + Logistic Regression pipeline, and optionally
the XGBoost severity model.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --transformer   # attempt transformer classifier too (if available)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import settings, get_logger
from src.classification.baseline import BaselineClassifier
from src.classification.taxonomy import categories

logger = get_logger(__name__)


def train_category_classifier(df: pd.DataFrame) -> dict:
    logger.info("Training category classifier on %d records", len(df))
    clf = BaselineClassifier(target_name="category")
    metrics = clf.fit(df["complaint_text"].tolist(), df["category"].tolist())
    clf.save(settings.path("paths.classification_dir"))
    return metrics


def train_subcategory_classifiers(df: pd.DataFrame) -> dict:
    all_metrics = {}
    out_dir = settings.path("paths.classification_dir")
    for cat in categories():
        subset = df[df["category"] == cat]
        n_unique = subset["subcategory"].nunique()
        if len(subset) < 20 or n_unique < 2:
            logger.info("Skipping subcategory classifier for '%s' (insufficient data: n=%d, classes=%d)",
                        cat, len(subset), n_unique)
            continue
        safe_name = f"sub_{cat.replace(' ', '_')}"
        clf = BaselineClassifier(target_name=safe_name)
        try:
            metrics = clf.fit(subset["complaint_text"].tolist(), subset["subcategory"].tolist())
            clf.save(out_dir)
            all_metrics[cat] = metrics
            logger.info("Trained subcategory classifier for '%s': acc=%.3f", cat, metrics["accuracy"])
        except ValueError as e:
            logger.warning("Could not train subcategory classifier for '%s': %s", cat, e)
    return all_metrics


def train_severity_model(df: pd.DataFrame) -> dict:
    """Optional XGBoost severity model, trained on rule-derived features
    against the (synthetic ground-truth) severity labels."""
    import numpy as np
    from src.severity.features import build_severity_features
    from src.severity.predictor import MLSeverityEngine, FEATURE_ORDER

    if settings.get("severity.model_type") != "xgboost":
        logger.info("severity.model_type is not 'xgboost'; skipping ML severity training "
                    "(rule-based engine will be used - this is the default and recommended mode).")
        return {}

    logger.info("Building severity features for %d records", len(df))
    feature_rows = []
    for _, row in df.iterrows():
        feats = build_severity_features(
            row["category"], row["subcategory"], row["sentiment"], row["emotion"],
            row["amount"], recent_similar_count=0, resolution_status=row["resolution_status"],
        )
        feature_rows.append([feats[k] for k in FEATURE_ORDER])
    X = np.array(feature_rows)
    y = df["severity"].tolist()

    engine = MLSeverityEngine()
    engine.fit(X, y)
    engine.save(settings.path("paths.severity_dir"))
    logger.info("Trained and saved XGBoost severity model")
    return {"trained": True, "n_samples": len(df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer", action="store_true",
                         help="Also attempt to train the optional transformer classifier if available")
    args = parser.parse_args()

    csv_path = settings.path("paths.complaints_csv")
    if not csv_path.exists():
        logger.error("No dataset found at %s. Run scripts/generate_data.py first.", csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df["complaint_text"] = df["complaint_text"].fillna("")

    metrics = {}
    metrics["category"] = train_category_classifier(df)
    metrics["subcategory"] = train_subcategory_classifiers(df)
    metrics["severity"] = train_severity_model(df)

    if args.transformer:
        try:
            from src.classification.transformer import TransformerClassifier
            logger.info("Attempting transformer classifier training...")
            tclf = TransformerClassifier(target_name="category")
            tmetrics = tclf.fit(df["complaint_text"].tolist(), df["category"].tolist())
            tclf.save(settings.path("paths.classification_dir"))
            metrics["transformer_category"] = tmetrics
        except RuntimeError as e:
            logger.warning("Transformer training skipped: %s", e)

    metrics_path = settings.path("paths.models_dir") / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
        except Exception:
            pass
        return str(o)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_default)
    logger.info("Saved training metrics to %s", metrics_path)
    logger.info("Category classifier: accuracy=%.3f macro_f1=%.3f",
                metrics["category"]["accuracy"], metrics["category"]["macro_f1"])
    print("\n=== Training complete ===")
    print(f"Category accuracy: {metrics['category']['accuracy']:.3f}")
    print(f"Category macro F1: {metrics['category']['macro_f1']:.3f}")
    print(f"Subcategory models trained: {len(metrics['subcategory'])}")


if __name__ == "__main__":
    main()
