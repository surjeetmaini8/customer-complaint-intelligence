"""Check required and optional dependencies before running the project."""
from __future__ import annotations

import importlib.util
import sys

REQUIRED = {
    "numpy": "numpy", "pandas": "pandas", "yaml": "pyyaml",
    "dotenv": "python-dotenv", "pydantic": "pydantic",
    "sklearn": "scikit-learn", "joblib": "joblib",
    "fastapi": "fastapi", "uvicorn": "uvicorn",
    "sqlalchemy": "sqlalchemy",
    "plotly": "plotly", "requests": "requests",
}
OPTIONAL = {
    "faiss": "faiss-cpu", "hdbscan": "hdbscan", "streamlit": "streamlit",
    "transformers": "transformers", "torch": "torch",
    "sentence_transformers": "sentence-transformers", "spacy": "spacy",
}

def main() -> int:
    missing = []
    optional_missing = []
    print("Customer Complaint Intelligence - environment check")
    print("=" * 58)
    for module, package in REQUIRED.items():
        ok = importlib.util.find_spec(module) is not None
        print(f"[{'OK' if ok else 'MISSING':7}] {package}")
        if not ok:
            missing.append(package)
    print("\nOptional components:")
    for module, package in OPTIONAL.items():
        ok = importlib.util.find_spec(module) is not None
        print(f"[{'OK' if ok else 'SKIP':7}] {package}")
        if not ok:
            optional_missing.append(package)
    if missing:
        print("\nMissing required packages:")
        print("  " + ", ".join(missing))
        print("Install them with: pip install -r requirements.txt")
        return 1
    print("\nEnvironment is ready for the full project pipeline.")
    if optional_missing:
        print("Optional packages not installed: " + ", ".join(optional_missing))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
