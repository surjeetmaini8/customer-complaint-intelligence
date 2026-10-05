"""
Synthetic customer complaint dataset generator.

IMPORTANT: All data produced here is 100% SYNTHETIC / FAKE. It is generated
for demonstration and ML experimentation purposes only and must never be
represented as real customer data.

Design goals:
  - realistic *variation* (phrase templates + random slot-filling + noise),
    not verbatim repeated strings
  - class imbalance across categories
  - temporal patterns (daily/hourly seasonality)
  - deliberately injected "incidents" (sudden spikes in a specific
    category/subcategory/platform/region combo) so the anomaly detector
    has something real to find
"""
from __future__ import annotations

import random
import string
import datetime as dt
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from src.config import settings, get_logger
from src.classification.taxonomy import load_taxonomy

logger = get_logger(__name__)

SOURCES = ["email", "support_ticket", "chat", "app_review", "social_media", "call_transcript"]
PLATFORMS = ["Android", "iOS", "Web"]
REGIONS = ["North India", "South India", "East India", "West India", "Central India"]
PRODUCTS = ["Prime Wallet", "QuickPay", "ShopEase App", "FastCart", "PayNow Card",
            "CloudStore Subscription", "InstaDeliver", "SecureBank App", "MegaMart Online", "TravelGo"]
PAYMENT_METHODS = ["UPI", "credit card", "debit card", "net banking", "wallet balance", "COD"]

# Category base weights -> deliberate class imbalance
CATEGORY_WEIGHTS = {
    "Payment": 0.18,
    "Delivery": 0.16,
    "Refund": 0.13,
    "Account": 0.10,
    "Product Quality": 0.09,
    "Fraud": 0.03,
    "Technical Issue": 0.14,
    "Subscription": 0.06,
    "Customer Service": 0.08,
    "Other": 0.03,
}

SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]
EMOTION_LABELS = ["Angry", "Frustrated", "Confused", "Disappointed", "Satisfied", "Neutral"]

# ------------------------------------------------------------------ templates
# {product} {amount} {order_id} {payment_method} {days} {agent} are slots.
TEMPLATES: Dict[str, List[str]] = {
    "failed transaction": [
        "I tried to pay for my order on {product} using {payment_method} but the transaction failed. Amount {amount} was mentioned but the order never went through.",
        "My {payment_method} payment of {amount} on {product} keeps failing even though my bank shows sufficient balance.",
        "Transaction failed multiple times while paying {amount} via {payment_method} on {product}. Very frustrating experience.",
    ],
    "duplicate charge": [
        "I was charged twice for the same order on {product}, order id {order_id}. Please refund the duplicate {amount} charge.",
        "{product} deducted {amount} two times for a single purchase using {payment_method}. This needs to be fixed immediately.",
    ],
    "payment deducted but order failed": [
        "Amount of {amount} was deducted from my {payment_method} but the order {order_id} shows as failed on {product}.",
        "My money got deducted but {product} says the order didn't place. Order id {order_id}, amount {amount}.",
    ],
    "payment reversed": [
        "The payment I made on {product} was reversed without any explanation, order {order_id}.",
        "I see a payment reversal of {amount} on my {payment_method} statement from {product} but no reason was given.",
    ],
    "payment pending": [
        "My payment of {amount} on {product} has been stuck in pending status for {days} days now.",
        "Order {order_id} shows payment pending even though I completed the {payment_method} transaction {days} days back.",
    ],
    "late delivery": [
        "My order {order_id} from {product} was supposed to arrive {days} days ago and it still hasn't shown up.",
        "Delivery for order {order_id} is delayed by {days} days with no updates from {product}.",
    ],
    "missing package": [
        "The courier marked my {product} order {order_id} as delivered but I never received the package.",
        "I never got my package for order {order_id}, but {product} shows it as delivered.",
    ],
    "damaged package": [
        "The package for order {order_id} from {product} arrived completely damaged and unusable.",
        "My order {order_id} was delivered broken and leaking, very poor packaging by {product}.",
    ],
    "wrong item": [
        "I ordered a different product but received something else entirely for order {order_id} on {product}.",
        "{product} sent me the wrong item for order {order_id}. I need the correct product urgently.",
    ],
    "delivery partner issue": [
        "The delivery agent for order {order_id} was rude and refused to deliver to the correct address.",
        "Delivery partner for {product} order {order_id} never attempted delivery despite marking multiple attempts.",
    ],
    "refund delayed": [
        "I returned my item {days} days ago but the refund of {amount} still hasn't been processed by {product}.",
        "Refund for order {order_id} is delayed beyond the promised {days} day window.",
    ],
    "refund missing": [
        "I was promised a refund of {amount} for order {order_id} but nothing has been credited to my account.",
        "{product} confirmed my refund but I never received the {amount} in my {payment_method} account.",
    ],
    "partial refund": [
        "I only received a partial refund of part of {amount} for my returned order {order_id} on {product}.",
        "The refund amount credited doesn't match the {amount} I paid for order {order_id}.",
    ],
    "refund status unclear": [
        "I can't find any clear status on my refund request for order {order_id} on {product}.",
        "Customer support couldn't explain the refund status for order {order_id}, very confusing process.",
    ],
    "login issue": [
        "I am unable to log into my {product} account even with the correct password.",
        "The {product} app keeps showing a login error every time I try to sign in.",
    ],
    "password reset": [
        "The password reset link from {product} never arrives in my email.",
        "I requested a password reset on {product} {days} times but the link doesn't work.",
    ],
    "account locked": [
        "My {product} account got locked for no reason and I can't access my orders or wallet.",
        "{product} locked my account after a failed login attempt and support hasn't unlocked it in {days} days.",
    ],
    "profile issue": [
        "My profile details on {product} keep resetting to old information every time I update them.",
        "I can't update my address on {product}, the profile page keeps showing an error.",
    ],
    "defective product": [
        "The product I received from {product} stopped working within {days} days of purchase.",
        "Order {order_id} arrived defective, the item doesn't turn on at all.",
    ],
    "not as described": [
        "The item from {product} order {order_id} looks nothing like what was shown on the listing.",
        "Product quality is far below what was advertised on {product} for order {order_id}.",
    ],
    "expired product": [
        "I received an expired product from {product}, order {order_id}, this is a serious safety concern.",
        "The item delivered for order {order_id} was already past its expiry date.",
    ],
    "quality below expectation": [
        "The build quality of the product from {product} order {order_id} is very poor for the price paid.",
        "Not happy with the quality of my recent {product} purchase, order {order_id}.",
    ],
    "unauthorized transaction": [
        "There is an unauthorized transaction of {amount} on my {product} account that I never made.",
        "I see a suspicious charge of {amount} via {payment_method} on {product} that I did not authorize.",
    ],
    "suspicious activity": [
        "I noticed suspicious login activity on my {product} account from an unknown device.",
        "There have been multiple suspicious attempts to access my {product} account in the last {days} days.",
    ],
    "account takeover": [
        "My {product} account was taken over and the registered email/phone was changed without my consent.",
        "Someone accessed my {product} account, changed my password, and made purchases using my {payment_method}.",
    ],
    "app crash": [
        "The {product} app crashes every time I try to open the checkout page on {platform}.",
        "{product} keeps crashing randomly on my {platform} device, making it unusable.",
    ],
    "OTP failure": [
        "I am not receiving the OTP to verify my login on {product}, tried {days} times already.",
        "OTP verification keeps failing on {product}, the code never arrives on my phone.",
        "My OTP never arrives when trying to login to {product} on {platform}.",
        "Trying to login to {product} but OTP is not being delivered to my registered number.",
    ],
    "page not loading": [
        "The checkout page on {product} website just keeps loading forever and never completes.",
        "{product} pages are not loading at all on {platform}, tried clearing cache with no luck.",
    ],
    "API error": [
        "Getting a generic API error whenever I try to place an order on {product}.",
        "{product} shows 'something went wrong' error code repeatedly during checkout.",
    ],
    "performance issue": [
        "{product} has become extremely slow on {platform} after the latest update.",
        "The app takes forever to load any page, very poor performance on {product}.",
    ],
    "unwanted renewal": [
        "My {product} subscription auto-renewed for {amount} even though I had cancelled it {days} days ago.",
        "I was charged {amount} for a subscription renewal on {product} that I never approved.",
    ],
    "cancellation issue": [
        "I have been trying to cancel my {product} subscription for {days} days with no success.",
        "The cancel subscription button on {product} doesn't work at all.",
    ],
    "billing mismatch": [
        "The amount billed for my {product} subscription doesn't match the {amount} advertised plan price.",
        "{product} billed me {amount} more than my selected subscription plan.",
    ],
    "plan downgrade issue": [
        "I tried downgrading my {product} plan but I'm still being charged the higher {amount} tier.",
        "My {product} subscription downgrade request from {days} days ago was never processed.",
    ],
    "no response": [
        "I have contacted {product} support {days} times about order {order_id} and received no response.",
        "No one from {product} customer service has responded to my emails in {days} days.",
    ],
    "unhelpful agent": [
        "The support agent I spoke to about order {order_id} on {product} was completely unhelpful.",
        "{product} customer service just kept repeating scripted responses without solving my issue.",
    ],
    "long wait time": [
        "I waited over {days} hours on hold with {product} customer service before giving up.",
        "The wait time to reach {product} support is unacceptably long, over {days} minutes.",
    ],
    "rude behavior": [
        "The {product} support agent was extremely rude while discussing my order {order_id}.",
        "I was spoken to very disrespectfully by a {product} representative regarding my complaint.",
    ],
    "general feedback": [
        "Just wanted to share some feedback about my recent experience with {product}.",
        "Overall my experience with {product} order {order_id} was okay, a few things could improve.",
    ],
    "feature request": [
        "It would be great if {product} added a way to track order {order_id} in real time.",
        "Please consider adding dark mode and better search to the {product} app.",
    ],
    "miscellaneous": [
        "Not sure which category this falls under but I had an issue with {product}.",
        "General query regarding my account on {product}, order {order_id}.",
    ],
}

OPENERS = ["", "Hi team, ", "Hello, ", "To whom it may concern, ", "Dear support, ", "Hey, "]
CLOSERS = ["", " Please help asap.", " Kindly resolve this soon.", " This is very disappointing.",
           " Waiting for your response.", " Thanks in advance.", " Please look into this."]


def _rand_amount(rng: random.Random) -> float:
    buckets = [(50, 500), (500, 2000), (2000, 8000), (8000, 30000), (30000, 100000)]
    weights = [0.35, 0.30, 0.20, 0.10, 0.05]
    lo, hi = rng.choices(buckets, weights=weights, k=1)[0]
    return round(rng.uniform(lo, hi), 2)


def _rand_order_id(rng: random.Random) -> str:
    return "ORD" + "".join(rng.choices(string.digits, k=8))


def _rand_customer_id(rng: random.Random) -> str:
    return "CUST" + "".join(rng.choices(string.digits, k=6))


def _fill_template(template: str, rng: random.Random, platform: str) -> str:
    text = template.format(
        product=rng.choice(PRODUCTS),
        amount=f"Rs. {int(_rand_amount(rng))}",
        order_id=_rand_order_id(rng),
        payment_method=rng.choice(PAYMENT_METHODS),
        days=rng.randint(1, 14),
        platform=platform,
    )
    if rng.random() < 0.6:
        text = rng.choice(OPENERS) + text
    if rng.random() < 0.5:
        text = text + rng.choice(CLOSERS)
    # light noise: occasional lowercase/typo-ish variation for realism
    if rng.random() < 0.05:
        text = text.replace(".", "..")
    return text.strip()


def _sentiment_emotion_for(category: str, subcategory: str, rng: random.Random):
    if category in ("Fraud",) or subcategory in ("account takeover", "unauthorized transaction"):
        sentiment = rng.choices(SENTIMENT_LABELS, weights=[0.02, 0.08, 0.90])[0]
        emotion = rng.choices(EMOTION_LABELS, weights=[0.45, 0.30, 0.10, 0.10, 0.0, 0.05])[0]
    elif category == "Other" and subcategory in ("general feedback", "feature request"):
        sentiment = rng.choices(SENTIMENT_LABELS, weights=[0.45, 0.40, 0.15])[0]
        emotion = rng.choices(EMOTION_LABELS, weights=[0.05, 0.05, 0.10, 0.05, 0.55, 0.20])[0]
    else:
        sentiment = rng.choices(SENTIMENT_LABELS, weights=[0.08, 0.22, 0.70])[0]
        emotion = rng.choices(EMOTION_LABELS, weights=[0.28, 0.30, 0.15, 0.18, 0.02, 0.07])[0]
    return sentiment, emotion


def _severity_seed(category: str, subcategory: str, amount: float, sentiment: str) -> str:
    if category == "Fraud" or subcategory in ("unauthorized transaction", "account takeover"):
        return "CRITICAL"
    if amount >= 20000:
        return "CRITICAL"
    if amount >= 5000 or sentiment == "Negative":
        return "HIGH" if amount >= 5000 else "MEDIUM"
    return "LOW" if sentiment != "Negative" else "MEDIUM"


class ComplaintGenerator:
    def __init__(self, seed: int = None):
        cfg = settings
        self.seed = seed if seed is not None else cfg.get("data_generation.random_seed", 42)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.taxonomy = load_taxonomy()
        self.start_date = pd.Timestamp(cfg.get("data_generation.start_date", "2024-01-01"))
        self.end_date = pd.Timestamp(cfg.get("data_generation.end_date", "2024-04-30"))
        self.incidents = cfg.get("data_generation.incidents", [])

    def _random_timestamp(self) -> pd.Timestamp:
        total_seconds = int((self.end_date - self.start_date).total_seconds())
        offset = self.rng.randint(0, max(total_seconds, 1))
        ts = self.start_date + pd.Timedelta(seconds=offset)
        # simple daily seasonality: bias towards business hours
        hour_weights = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 9,
                                  8, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        hour_weights = hour_weights / hour_weights.sum()
        hour = self.np_rng.choice(24, p=hour_weights)
        ts = ts.replace(hour=int(hour), minute=self.rng.randint(0, 59), second=self.rng.randint(0, 59))
        return ts

    def _random_category_subcategory(self):
        cats = list(CATEGORY_WEIGHTS.keys())
        weights = list(CATEGORY_WEIGHTS.values())
        category = self.rng.choices(cats, weights=weights, k=1)[0]
        subs = self.taxonomy.get(category, ["miscellaneous"])
        subcategory = self.rng.choice(subs)
        return category, subcategory

    def _make_text(self, subcategory: str, platform: str) -> str:
        templates = TEMPLATES.get(subcategory)
        if not templates:
            templates = ["I am facing an issue with {product}, order {order_id}."]
        template = self.rng.choice(templates)
        return _fill_template(template, self.rng, platform)

    def _base_record(self, idx: int, forced_ts: pd.Timestamp = None,
                      forced_category: str = None, forced_subcategory: str = None,
                      forced_platform: str = None, forced_region: str = None) -> Dict[str, Any]:
        ts = forced_ts if forced_ts is not None else self._random_timestamp()
        category, subcategory = (forced_category, forced_subcategory) if forced_category else self._random_category_subcategory()
        platform = forced_platform or self.rng.choice(PLATFORMS)
        region = forced_region or self.rng.choice(REGIONS)
        source = self.rng.choice(SOURCES)
        text = self._make_text(subcategory, platform)
        amount = _rand_amount(self.rng) if category in ("Payment", "Refund", "Subscription", "Fraud") else round(self.rng.uniform(0, 1500), 2)
        sentiment, emotion = _sentiment_emotion_for(category, subcategory, self.rng)
        severity = _severity_seed(category, subcategory, amount, sentiment)
        resolution_status = self.rng.choices(
            ["open", "in_progress", "resolved", "escalated"], weights=[0.30, 0.25, 0.35, 0.10]
        )[0]

        return {
            "complaint_id": f"CMP-{idx:06d}",
            "timestamp": ts.isoformat(),
            "source": source,
            "complaint_text": text,
            "category": category,
            "subcategory": subcategory,
            "sentiment": sentiment,
            "emotion": emotion,
            "severity": severity,
            "product": self.rng.choice(PRODUCTS),
            "region": region,
            "amount": amount,
            "order_id": _rand_order_id(self.rng),
            "customer_id": _rand_customer_id(self.rng),
            "platform": platform,
            "resolution_status": resolution_status,
            "is_synthetic": True,
        }

    def _generate_incident_records(self, start_idx: int) -> List[Dict[str, Any]]:
        """Injects sudden spikes for specific category/subcategory/platform/region
        combinations within a tight time window, to demonstrate anomaly detection."""
        records = []
        idx = start_idx
        for incident in self.incidents:
            window_start = pd.Timestamp(incident["start"])
            window_end = pd.Timestamp(incident["end"])
            duration_hours = max(int((window_end - window_start).total_seconds() // 3600), 1)
            baseline_per_hour = 5
            peak = int(baseline_per_hour * incident.get("peak_multiplier", 6))
            for h in range(duration_hours):
                bucket_start = window_start + pd.Timedelta(hours=h)
                # ramp up then down (triangular profile) for realism
                progress = h / max(duration_hours - 1, 1)
                intensity = 1 - abs(progress - 0.5) * 2  # 0..1..0
                volume = int(baseline_per_hour + (peak - baseline_per_hour) * intensity)
                for _ in range(volume):
                    ts = bucket_start + pd.Timedelta(seconds=self.rng.randint(0, 3599))
                    rec = self._base_record(
                        idx,
                        forced_ts=ts,
                        forced_category=incident["category"],
                        forced_subcategory=incident["subcategory"],
                        forced_platform=incident.get("platform"),
                        forced_region=incident.get("region"),
                    )
                    rec["complaint_id"] = f"CMP-{idx:06d}"
                    rec["injected_incident"] = incident["name"]
                    records.append(rec)
                    idx += 1
        logger.info("Injected %d incident-related complaints across %d incidents", len(records), len(self.incidents))
        return records

    def generate(self, n: int = None) -> pd.DataFrame:
        n = n or settings.get("data_generation.num_records", 10000)
        incident_records = self._generate_incident_records(start_idx=1)
        n_normal = max(n - len(incident_records), 0)
        normal_records = [self._base_record(idx) for idx in range(len(incident_records) + 1, len(incident_records) + 1 + n_normal)]
        all_records = incident_records + normal_records
        self.rng.shuffle(all_records)
        # reassign sequential complaint ids after shuffle for cleanliness
        for i, r in enumerate(all_records, start=1):
            r["complaint_id"] = f"CMP-{i:06d}"
        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        for i, idx in enumerate(df.index, start=1):
            df.at[idx, "complaint_id"] = f"CMP-{i:06d}"
        logger.info("Generated %d synthetic complaint records (%d from injected incidents)",
                    len(df), len(incident_records))
        return df


def generate_dataset(n: int = None, save: bool = True) -> pd.DataFrame:
    gen = ComplaintGenerator()
    df = gen.generate(n)
    if save:
        out_path = settings.path("paths.complaints_csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved synthetic dataset to %s", out_path)
    return df
