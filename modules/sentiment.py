"""Lightweight, reliable sentiment analysis for customer conversations."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


POSITIVE_WORDS = {
    "good", "great", "excellent", "awesome", "amazing", "helpful", "interested",
    "love", "like", "nice", "perfect", "thanks", "thank", "sure", "yes",
    "definitely", "excited", "happy", "satisfied", "impressive", "works",
    "affordable", "reasonable", "book", "schedule", "demo", "proceed",
    "accha", "acha", "badhiya", "shukriya", "dhanyavad", "pasand", "haan",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "angry",
    "frustrated", "annoyed", "useless", "waste", "expensive", "costly",
    "overpriced", "slow", "broken", "confusing", "disappointed", "cancel",
    "stop", "spam", "scam", "unhelpful", "problem", "issue", "fail", "wrong",
    "bura", "bekar", "mehenga", "gussa", "bakwas", "nahi",
}

NEGATORS = {"not", "no", "never", "dont", "don't", "didnt", "didn't", "isnt", "isn't", "cant", "can't"}


def analyze_sentiment(text: str) -> Dict[str, object]:
    """Classify text sentiment as Positive, Neutral, or Negative with a score in [-1.0, 1.0]."""
    if not text or not text.strip():
        return {"label": "Neutral", "score": 0.0, "reason": "Empty input"}

    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z\u0900-\u097F']+", lowered)
    if not tokens:
        return {"label": "Neutral", "score": 0.0, "reason": "No lexical tokens"}

    pos_hits = 0.0
    neg_hits = 0.0

    for idx, tok in enumerate(tokens):
        prev_tok = tokens[idx - 1] if idx > 0 else ""
        is_negated = prev_tok in NEGATORS

        if tok in POSITIVE_WORDS:
            if is_negated:
                neg_hits += 1.0
            else:
                pos_hits += 1.0
        elif tok in NEGATIVE_WORDS:
            if is_negated:
                pos_hits += 0.5
            else:
                neg_hits += 1.0

    # Phrase-level boosts
    if any(p in lowered for p in ["not interested", "too expensive", "waste of time", "call back later", "no thanks"]):
        neg_hits += 1.5
    if any(p in lowered for p in ["very interested", "looking forward", "sounds good", "book a demo", "schedule a call"]):
        pos_hits += 1.5

    total = pos_hits + neg_hits
    if total == 0:
        return {"label": "Neutral", "score": 0.0, "reason": "Informational or neutral phrasing"}

    raw_score = (pos_hits - neg_hits) / max(total, 1.0)
    score = round(max(-1.0, min(1.0, raw_score)), 2)

    if score >= 0.2:
        label = "Positive"
    elif score <= -0.2:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "score": score,
        "reason": f"pos={pos_hits:.1f}, neg={neg_hits:.1f}",
    }


def aggregate_conversation_sentiment(messages: List[Dict[str, str]]) -> Tuple[str, float]:
    """Compute overall customer sentiment across user turns, weighting recent turns slightly higher."""
    user_msgs = [m.get("message") or m.get("content") or "" for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return "Neutral", 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    for idx, msg in enumerate(user_msgs, start=1):
        res = analyze_sentiment(msg)
        weight = 1.0 + 0.25 * idx
        weighted_sum += float(res["score"]) * weight
        weight_total += weight

    avg_score = round(weighted_sum / max(weight_total, 1.0), 2)
    if avg_score >= 0.18:
        return "Positive", avg_score
    if avg_score <= -0.18:
        return "Negative", avg_score
    return "Neutral", avg_score
