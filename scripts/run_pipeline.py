"""
Convenience script that runs the entire local pipeline end-to-end from a
clean environment:
  1. generate synthetic data
  2. train classification models
  3. build FAISS indexes (complaints + knowledge base)
  4. run the full analysis pipeline (classification/sentiment/emotion/NER/
     severity/clustering/anomaly-detection/root-cause analysis) and
     persist everything to the database

After this completes, start the API and dashboard separately:
    python -m uvicorn src.api.main:app --reload
    streamlit run dashboard/app.py

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --n 3000 --limit 3000   # faster dev run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"Step failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Number of complaint records to generate")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows processed in process_complaints.py")
    parser.add_argument("--transformer", action="store_true", help="Also attempt transformer classifier training")
    args = parser.parse_args()

    run([sys.executable, "scripts/check_environment.py"])

    gen_cmd = [sys.executable, "scripts/generate_data.py"]
    if args.n:
        gen_cmd += ["--n", str(args.n)]
    run(gen_cmd)

    train_cmd = [sys.executable, "scripts/train_models.py"]
    if args.transformer:
        train_cmd.append("--transformer")
    run(train_cmd)

    run([sys.executable, "scripts/build_index.py"])

    process_cmd = [sys.executable, "scripts/process_complaints.py"]
    if args.limit:
        process_cmd += ["--limit", str(args.limit)]
    run(process_cmd)

    print("\n=== Pipeline complete ===")
    print("Next steps:")
    print("  python -m uvicorn src.api.main:app --reload")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
