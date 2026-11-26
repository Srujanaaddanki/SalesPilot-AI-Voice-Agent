"""Call summary generation and SQLite persistence."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from database.database import DatabaseManager
from modules.sentiment import aggregate_conversation_sentiment


class CallSummarizer:
    """Generates structured post-call summaries using LLM when available or deterministic fallback."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    @staticmethod
    def recommend_next_action(lead: Dict[str, Any], has_appointment: bool, sentiment: str) -> str:
        status = lead.get("qualification_status", "New")
        if sentiment == "Negative":
            return "Assign to senior account manager for empathetic follow-up and objection resolution."
        if has_appointment and status == "Qualified":
            return "Confirm requested demo slot via email/phone and prepare tailored AI automation proposal."
        if has_appointment:
            return "Confirm requested appointment slot and gather remaining budget/timeline details on call."
        if status == "Qualified":
            return "Send tailored pricing proposal and propose 2 concrete demo slots within 24 hours."
        if lead.get("email") or lead.get("phone"):
            return "Follow up with case study and clarify budget & implementation timeline."
        return "Request customer contact information and confirm business requirements."

    def generate_and_save_summary(
        self,
        session_id: str,
        ai_engine: Any = None,
    ) -> Dict[str, Any]:
        """Build a structured call summary from conversation history, lead data, and appointments."""
        history: List[Dict[str, Any]] = self.db.get_conversation_history(session_id)
        lead: Dict[str, Any] = self.db.get_lead_by_session(session_id) or {}
        appointments = [a for a in self.db.get_all_appointments() if a.get("session_id") == session_id]
        has_appointment = len(appointments) > 0

        sentiment_label, _ = aggregate_conversation_sentiment(history)
        if lead.get("sentiment") and not history:
            sentiment_label = str(lead["sentiment"])

        customer_turns = [m["message"] for m in history if m.get("role") == "user"]
        requirements = (
            lead.get("business_requirement")
            or ("; ".join(customer_turns[:3])[:300] if customer_turns else "Not specified")
        )
        product_interest = lead.get("product_interest") or "General AI Automation Inquiry"
        budget = lead.get("budget") or "Not disclosed"
        timeline = lead.get("purchase_timeline") or "Not specified"
        qual_status = lead.get("qualification_status") or "New"
        lead_score = int(lead.get("lead_score") or 0)
        next_action = self.recommend_next_action(lead, has_appointment, sentiment_label)

        # Optionally enrich narrative & next action with LLM if engine is live
        llm_summary_text = ""
        if ai_engine is not None and history and getattr(ai_engine, "active_provider", "local_fallback") != "local_fallback":
            llm_result = ai_engine.generate_structured_summary(history, lead, appointments)
            if isinstance(llm_result, dict):
                requirements = llm_result.get("customer_requirements") or requirements
                product_interest = llm_result.get("product_interest") or product_interest
                budget = llm_result.get("budget") or budget
                timeline = llm_result.get("purchase_timeline") or timeline
                next_action = llm_result.get("recommended_next_action") or next_action
                llm_summary_text = llm_result.get("summary_text") or ""

        if not llm_summary_text:
            appt_note = (
                f"Appointment requested for {appointments[0]['preferred_date']} at {appointments[0]['preferred_time']}."
                if has_appointment
                else "No appointment requested yet."
            )
            cust_name = lead.get("name") or "Prospective customer"
            llm_summary_text = (
                f"{cust_name} inquired about {product_interest}. "
                f"Requirements: {requirements}. Budget: {budget}. Timeline: {timeline}. "
                f"Overall sentiment: {sentiment_label}. Status: {qual_status} ({lead_score}/100). "
                f"{appt_note} Recommended next action: {next_action}"
            )

        summary_payload = {
            "customer_requirements": requirements,
            "product_interest": product_interest,
            "budget": budget,
            "purchase_timeline": timeline,
            "sentiment": sentiment_label,
            "qualification_status": qual_status,
            "lead_score": lead_score,
            "recommended_next_action": next_action,
            "summary_text": llm_summary_text,
        }
        return self.db.save_call_summary(session_id, summary_payload)
