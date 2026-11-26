"""Lead extraction, BANT/completeness qualification, scoring, and SQLite persistence."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from database.database import DatabaseManager


REQUIRED_LEAD_FIELDS = [
    "name",
    "email",
    "phone",
    "product_interest",
    "budget",
    "business_requirement",
    "purchase_timeline",
]

FIELD_LABELS = {
    "name": "Customer Name",
    "email": "Email Address",
    "phone": "Phone Number",
    "product_interest": "Product / Service Interest",
    "budget": "Budget",
    "business_requirement": "Business Requirement",
    "purchase_timeline": "Purchase Timeline",
}


class LeadManager:
    """Extracts, scores, validates, and persists sales leads in SQLite."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    @staticmethod
    def extract_lead_fields_from_text(text: str) -> Dict[str, str]:
        """Extract lead details deterministically from natural-language user messages."""
        if not text or not text.strip():
            return {}

        extracted: Dict[str, str] = {}
        clean = text.strip()
        lower = clean.lower()

        # Email
        email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", clean)
        if email_match:
            extracted["email"] = email_match.group(0)

        # Phone (10-15 digits with optional + and separators)
        phone_match = re.search(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?){1,2}\d{4,6}\b", clean)
        if phone_match:
            digits = re.sub(r"\D", "", phone_match.group(0))
            if 10 <= len(digits) <= 15:
                extracted["phone"] = phone_match.group(0).strip()

        # Name patterns (careful not to match "I am interested..." or trailing conjunctions like "and")
        name_patterns = [
            (r"(?:my name is|i am|i'm|this is|call me|name[:\s]+)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", 0),
            (r"(?:my name is|name[:\s]+)\s*([a-z]{2,20}(?:\s+[a-z]{2,20}){0,1})", re.IGNORECASE),
            (r"mera naam\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+hai", re.IGNORECASE),
        ]
        stop_names = {
            "interested", "looking", "calling", "trying", "asking", "here", "ready",
            "from", "good", "fine", "happy", "concerned", "building", "running",
        }
        trailing_stop_words = {"and", "or", "from", "at", "with", "for", "email", "phone", "my"}
        for pat, pat_flags in name_patterns:
            m = re.search(pat, clean, flags=pat_flags)
            if m:
                words = m.group(1).strip(" .,!?").split()
                while words and words[-1].lower() in trailing_stop_words:
                    words.pop()
                candidate = " ".join(words)
                if words and words[0].lower() not in stop_names and len(candidate) >= 2:
                    extracted["name"] = candidate.title()
                    break

        # Budget patterns
        budget_patterns = [
            r"(?:budget(?:\s+is|\s+around|\s+of)?\s*(?:is|around|about)?\s*)((?:rs\.?|inr|₹|usd|\$)?\s*\d[\d,]*(?:\.\d+)?\s*(?:k|lakh|lakhs|crore|m|million|thousand)?(?:\s*(?:/|per)?\s*(?:month|mo|year|yr|annum))?)",
            r"((?:rs\.?|inr|₹|\$|usd)\s*\d[\d,]*(?:\.\d+)?\s*(?:k|lakh|lakhs|crore|m)?(?:\s*(?:/|per)?\s*(?:month|mo|year|yr|annum))?)",
            r"(\d[\d,]*(?:\.\d+)?\s*(?:k|lakh|lakhs)\s*(?:per\s+month|monthly|per\s+year|annually|/month|/mo)?)",
            r"(\d[\d,]*(?:\.\d+)?\s*(?:per\s+month|monthly|per\s+year|annually|/month|/mo))",
        ]
        for pat in budget_patterns:
            m = re.search(pat, clean, flags=re.IGNORECASE)
            if m:
                val = m.group(1).strip(" .,")
                if any(ch.isdigit() for ch in val):
                    extracted["budget"] = val
                    break

        # Purchase timeline patterns
        timeline_patterns = [
            r"(?:timeline(?:\s+is)?[:\s]*|within\s+|in\s+the\s+next\s+|by\s+)([a-z0-9\s-]{3,30}?(?:days?|weeks?|months?|quarter|q[1-4]|year))",
            r"\b(immediately|asap|right away|this week|next week|this month|next month|within a month|next quarter|q[1-4])\b",
            r"\b(agale mahine|is mahine|turant|jaldi)\b",
        ]
        for pat in timeline_patterns:
            m = re.search(pat, lower)
            if m:
                extracted["purchase_timeline"] = m.group(1).strip(" .,")
                break

        # Product or service interest
        interests = []
        interest_map = [
            ("AI Automation Services", ["ai automation", "automation service", "automate", "workflow automation"]),
            ("SalesPilot AI Voice Agent", ["voice agent", "ai voice", "sales agent", "voice bot", "calling agent", "salespilot"]),
            ("RAG Knowledge Assistant", ["rag", "knowledge base", "document qa", "knowledge system"]),
            ("CRM & Calendar Integration", ["crm integration", "zoho", "hubspot", "salesforce"]),
            ("Starter Plan", ["starter plan", "starter package"]),
            ("Growth Plan", ["growth plan", "growth package"]),
            ("Enterprise Plan", ["enterprise plan", "enterprise custom"]),
        ]
        for label, keywords in interest_map:
            if any(kw in lower for kw in keywords):
                interests.append(label)
        if interests:
            extracted["product_interest"] = ", ".join(dict.fromkeys(interests))

        # Business requirement detection
        req_triggers = [
            "need ", "want to ", "looking to ", "looking for ", "automate ",
            "require ", "problem is ", "goal is ", "use case", "we have ", "our team ",
            "chahiye", "chahte",
        ]
        if any(t in lower for t in req_triggers) and len(clean) >= 15:
            # Avoid capturing pure greeting or pure contact info as requirement
            if not (len(extracted) >= 2 and len(clean) < 45 and ("@" in clean or "name" in lower)):
                extracted["business_requirement"] = clean[:280]

        return extracted

    @staticmethod
    def calculate_score_and_status(lead: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic lead qualification scoring (0-100)."""
        weights = {
            "name": 10,
            "email": 15,
            "phone": 15,
            "product_interest": 15,
            "business_requirement": 15,
            "budget": 15,
            "purchase_timeline": 15,
        }
        score = 0
        for field, weight in weights.items():
            val = str(lead.get(field) or "").strip()
            if val:
                score += weight

        has_contact = bool(str(lead.get("email") or "").strip() or str(lead.get("phone") or "").strip())
        has_intent = bool(str(lead.get("product_interest") or "").strip() or str(lead.get("business_requirement") or "").strip())
        has_commercial = bool(str(lead.get("budget") or "").strip() and str(lead.get("purchase_timeline") or "").strip())

        if score >= 70 and has_contact and has_intent:
            status = "Qualified"
        elif score >= 45 and (has_contact or has_commercial or has_intent):
            status = "Warm (In Progress)"
        elif score > 0:
            status = "New"
        else:
            status = "Unqualified"

        return {"lead_score": score, "qualification_status": status}

    @staticmethod
    def get_missing_fields(lead: Optional[Dict[str, Any]]) -> List[str]:
        """Return list of missing qualification fields in priority conversation order."""
        lead = lead or {}
        missing = []
        for field in REQUIRED_LEAD_FIELDS:
            if not str(lead.get(field) or "").strip():
                missing.append(field)
        return missing

    def update_from_message(
        self,
        session_id: str,
        user_message: str,
        sentiment_label: str = "Neutral",
        sentiment_score: float = 0.0,
        language: str = "English",
        extra_fields: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Extract lead attributes from message, merge with existing DB record, score, and persist."""
        existing = self.db.get_lead_by_session(session_id) or {}
        extracted = self.extract_lead_fields_from_text(user_message)
        if extra_fields:
            for k, v in extra_fields.items():
                if v and str(v).strip():
                    extracted[k] = str(v).strip()

        merged: Dict[str, Any] = dict(existing)
        for k, v in extracted.items():
            if k == "business_requirement" and existing.get("business_requirement"):
                prev = str(existing["business_requirement"])
                if v not in prev and len(prev) < 350:
                    merged[k] = f"{prev}; {v}"[:400]
            elif k == "product_interest" and existing.get("product_interest"):
                prev_items = [x.strip() for x in str(existing["product_interest"]).split(",") if x.strip()]
                new_items = [x.strip() for x in str(v).split(",") if x.strip()]
                combined = list(dict.fromkeys(prev_items + new_items))
                merged[k] = ", ".join(combined)[:300]
            else:
                merged[k] = v

        merged["sentiment"] = sentiment_label
        merged["sentiment_score"] = sentiment_score
        merged["language"] = language

        scoring = self.calculate_score_and_status(merged)
        merged.update(scoring)

        for non_updatable in ("id", "session_id", "created_at", "updated_at"):
            merged.pop(non_updatable, None)

        return self.db.upsert_lead(session_id, **merged)
