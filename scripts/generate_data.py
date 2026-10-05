"""
Generates the synthetic complaint dataset.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --n 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings, get_logger
from src.ingestion.generators import generate_dataset

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Number of records to generate")
    args = parser.parse_args()

    n = args.n or settings.get("data_generation.num_records", 10000)
    print(f"Generating {n} synthetic complaint records (this is 100% synthetic demo data)...")
    df = generate_dataset(n=n, save=True)
    print(f"Done. Saved {len(df)} records to {settings.path('paths.complaints_csv')}")
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    if "injected_incident" in df.columns:
        print("\nInjected incidents:")
        print(df["injected_incident"].value_counts(dropna=True))


if __name__ == "__main__":
    main()
