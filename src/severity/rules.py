"""Rule definitions used by the transparent, human-readable severity engine."""
from __future__ import annotations

FRAUD_CATEGORIES = {"Fraud"}
FRAUD_SUBCATEGORIES = {"unauthorized transaction", "suspicious activity", "account takeover"}

HIGH_IMPACT_SUBCATEGORIES = {
    "account locked", "refund missing", "duplicate charge",
    "unauthorized transaction", "account takeover",
}

FREQUENCY_SENSITIVE_SUBCATEGORIES = {
    "OTP failure", "app crash", "API error", "page not loading", "failed transaction",
}
