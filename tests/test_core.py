"""Unit, API, and end-to-end workflow tests for SalesPilot AI."""

from __future__ import annotations

import io
import wave
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app as st_app
from backend import main as api_main
from database.database import DatabaseManager
from modules.ai_engine import SalesAIEngine
from modules.appointment_manager import AppointmentManager
from modules.lead_manager import LeadManager
from modules.rag_engine import RAGEngine
from modules.sentiment import analyze_sentiment
from modules.speech_engine import MAX_AUDIO_BYTES, SpeechEngine
from modules.summarizer import CallSummarizer


def _make_temp_db(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(db_path=str(tmp_path / "test_salespilot.db"))


def test_database_creation_and_lead_crud(tmp_path: Path) -> None:
    db = _make_temp_db(tmp_path)
    lead = db.upsert_lead(
        "sess-1",
        name="Riya Kapoor",
        email="riya@example.com",
        phone="9876543210",
        product_interest="AI Automation Services",
        budget="INR 50,000/month",
        business_requirement="Automate inbound lead qualification",
        purchase_timeline="this month",
        sentiment="Positive",
        lead_score=100,
        qualification_status="Qualified",
    )
    assert lead["name"] == "Riya Kapoor"
    assert lead["qualification_status"] == "Qualified"

    fetched = db.get_lead_by_session("sess-1")
    assert fetched is not None
    assert fetched["email"] == "riya@example.com"
    assert len(db.get_all_leads()) == 1


def test_appointment_validation_and_persistence(tmp_path: Path) -> None:
    db = _make_temp_db(tmp_path)
    mgr = AppointmentManager(db)

    bad = mgr.submit_appointment_request(
        session_id="sess-appt",
        customer_name="Arjun",
        contact_details="arjun@example.com",
        preferred_date="2020-01-01",
        preferred_time="15:00",
    )
    assert bad["success"] is False

    bad_contact = mgr.submit_appointment_request(
        session_id="sess-appt",
        customer_name="Arjun",
        contact_details="not-valid",
        preferred_date=(date.today() + timedelta(days=2)).isoformat(),
        preferred_time="15:00",
    )
    assert bad_contact["success"] is False

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    ok = mgr.submit_appointment_request(
        session_id="sess-appt",
        customer_name="Arjun Verma",
        contact_details="arjun@example.com",
        preferred_date=tomorrow,
        preferred_time="03:00 PM",
    )
    assert ok["success"] is True
    assert "Pending Confirmation" in ok["appointment"]["status"]
    assert len(mgr.list_appointment_requests()) == 1


def test_rag_retrieval_and_honesty_on_unknown_query() -> None:
    rag = RAGEngine(knowledge_path=Path("data/company_knowledge.txt"))
    assert len(rag.chunks) >= 5

    context, hits = rag.get_context("What are your AI automation services and pricing plans?")
    assert hits
    assert "Starter AI Automation Plan" in context or "AI Automation" in context

    _, unknown_hits = rag.get_context("quantum teleportation submarine warranty policy")
    assert unknown_hits == []


def test_sentiment_analysis() -> None:
    pos = analyze_sentiment("This looks great and very helpful, I am interested in booking a demo!")
    neg = analyze_sentiment("This is too expensive and a waste of time, I am frustrated.")
    neu = analyze_sentiment("My office is located in Bengaluru.")
    assert pos["label"] == "Positive"
    assert neg["label"] == "Negative"
    assert neu["label"] == "Neutral"


def test_env_missing_keys_uses_local_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "auto")

    db = _make_temp_db(tmp_path)
    rag = RAGEngine(knowledge_path=Path("data/company_knowledge.txt"))
    engine = SalesAIEngine(db=db, rag=rag)
    assert engine.active_provider == "local_fallback"

    res = engine.process_turn("no-key-sess", "What pricing plans do you offer?", "English")
    assert res["is_fallback"] is True
    assert "Starter AI Automation Plan" in res["reply"] or "25,000" in res["reply"]


def test_end_to_end_workflow_in_local_fallback_mode(tmp_path: Path) -> None:
    db = _make_temp_db(tmp_path)
    rag = RAGEngine(knowledge_path=Path("data/company_knowledge.txt"))
    engine = SalesAIEngine(db=db, rag=rag, force_fallback=True)
    appt_mgr = AppointmentManager(db)
    summarizer = CallSummarizer(db)
    session_id = "e2e-session-01"

    turn1 = engine.process_turn(
        session_id=session_id,
        user_message="I am interested in your AI automation services.",
        language="English",
    )
    assert turn1["is_fallback"] is True
    assert turn1["provider_used"] == "local_fallback"
    assert turn1["rag_hits"]
    assert "budget" in turn1["reply"].lower() or "automate" in turn1["reply"].lower()

    engine.process_turn(
        session_id=session_id,
        user_message=(
            "My name is Neha Kulkarni, email neha@finscale.in, phone +91 9811223344. "
            "We need to automate lead follow-ups for our sales team. "
            "Our budget is INR 65,000/month and timeline is this month."
        ),
        language="English",
    )
    lead = db.get_lead_by_session(session_id)
    assert lead is not None
    assert lead["name"] == "Neha Kulkarni"
    assert lead["email"] == "neha@finscale.in"
    assert lead["qualification_status"] == "Qualified"
    assert lead["lead_score"] >= 70

    turn3 = engine.process_turn(
        session_id=session_id,
        user_message="Do you offer nuclear reactor hardware maintenance?",
        language="English",
    )
    assert "do not have verified information" in turn3["reply"].lower()

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    appt_res = appt_mgr.submit_appointment_request(
        session_id=session_id,
        customer_name=lead["name"],
        contact_details=lead["email"],
        preferred_date=tomorrow,
        preferred_time="11:30",
    )
    assert appt_res["success"] is True

    summary = summarizer.generate_and_save_summary(session_id, ai_engine=engine)
    assert summary["session_id"] == session_id
    assert summary["qualification_status"] == "Qualified"
    assert summary["recommended_next_action"]

    metrics = db.get_dashboard_metrics()
    assert metrics["total_leads"] == 1
    assert metrics["qualified_leads"] == 1
    assert metrics["total_appointments"] == 1
    assert metrics["total_summaries"] == 1
    assert metrics["total_messages"] == 6


def test_hindi_fallback_and_speech_graceful_handling(tmp_path: Path) -> None:
    db = _make_temp_db(tmp_path)
    rag = RAGEngine(knowledge_path=Path("data/company_knowledge.txt"))
    engine = SalesAIEngine(db=db, rag=rag, force_fallback=True)
    res = engine.process_turn("hindi-sess", "नमस्ते", language="Hindi")
    assert "SalesPilot AI" in res["reply"]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
    speech = SpeechEngine()
    transcript, err = speech.transcribe_audio_bytes(buf.getvalue(), language="Hindi")
    assert transcript is None
    assert err is not None


def test_fastapi_routes_isolated(tmp_path: Path) -> None:
    temp_db = _make_temp_db(tmp_path)
    temp_rag = RAGEngine(knowledge_path=Path("data/company_knowledge.txt"))
    temp_engine = SalesAIEngine(db=temp_db, rag=temp_rag, force_fallback=True)

    orig_engine = api_main.sales_agent.engine
    orig_appt = api_main.appointment_manager
    orig_sum = api_main.summarizer
    try:
        api_main.sales_agent.engine = temp_engine
        api_main.appointment_manager = AppointmentManager(temp_db)
        api_main.summarizer = CallSummarizer(temp_db)

        health = api_main.health()
        assert health["status"] == "healthy"

        chat_resp = api_main.chat(
            api_main.ChatRequest(
                message="I am interested in your AI automation services.",
                session_id="api-sess-1",
                language="English",
            )
        )
        assert chat_resp.provider_used == "local_fallback"
        assert chat_resp.ai_response

        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        appt_resp = api_main.request_appointment(
            api_main.AppointmentPayload(
                session_id="api-sess-1",
                customer_name="Kiran Rao",
                contact_details="kiran@example.com",
                preferred_date=tomorrow,
                preferred_time="14:00",
            )
        )
        assert appt_resp["success"] is True

        summary_resp = api_main.generate_summary("api-sess-1")
        assert summary_resp["session_id"] == "api-sess-1"

        dash = api_main.dashboard()
        assert dash["metrics"]["total_appointments"] == 1
        assert dash["metrics"]["total_summaries"] == 1
    finally:
        api_main.sales_agent.engine = orig_engine
        api_main.appointment_manager = orig_appt
        api_main.summarizer = orig_sum


def test_contextual_memory_and_session_isolation(tmp_path: Path) -> None:
    db = _make_temp_db(tmp_path)
    rag = RAGEngine(knowledge_path=Path("data/company_knowledge.txt"))
    engine = SalesAIEngine(db=db, rag=rag, force_fallback=True)

    # Customer A: Turn 1 (name + initial interest), Turn 2 (budget), Turn 3 (product question)
    engine.process_turn("cust-a", "Hi, my name is Sneha Iyer and I am interested in AI automation services.", "English")
    engine.process_turn("cust-a", "Our budget is around 50,000 per month.", "English")
    turn3 = engine.process_turn("cust-a", "Tell me about the Growth Plan pricing.", "English")

    lead_a = db.get_lead_by_session("cust-a")
    assert lead_a is not None
    assert lead_a["name"] == "Sneha Iyer"
    assert "50,000" in lead_a["budget"]
    assert "AI Automation Services" in lead_a["product_interest"]
    assert "Growth Plan" in lead_a["product_interest"]
    assert "Sneha Iyer" in turn3["reply"]

    # Customer B in separate session must not inherit Customer A's details
    engine.process_turn("cust-b", "My name is Vikram Das.", "English")
    lead_b = db.get_lead_by_session("cust-b")
    assert lead_b is not None
    assert lead_b["name"] == "Vikram Das"
    assert lead_b["budget"] == ""
    assert len(db.get_conversation_history("cust-a")) == 6
    assert len(db.get_conversation_history("cust-b")) == 2


def test_security_input_validation_and_admin_controls(tmp_path: Path, monkeypatch) -> None:
    db = _make_temp_db(tmp_path)
    rag = RAGEngine(knowledge_path=Path("data/company_knowledge.txt"))
    engine = SalesAIEngine(db=db, rag=rag, force_fallback=True)

    # Oversized chat message rejected
    with pytest.raises(ValueError, match="maximum allowed length"):
        engine.process_turn("sec-sess", "A" * (engine.MAX_MESSAGE_CHARS + 1), "English")

    # Invalid/non-WAV upload and oversized audio rejected
    speech = SpeechEngine()
    txt_bad, err_bad = speech.transcribe_audio_bytes(b"NOT_A_WAV_FILE" * 10, "English")
    assert txt_bad is None and "Unsupported audio format" in str(err_bad)

    txt_big, err_big = speech.transcribe_audio_bytes(b"RIFF" + b"\x00" * (MAX_AUDIO_BYTES + 10), "English")
    assert txt_big is None and "10 MB" in str(err_big)

    # Admin passcode helper & CSV formula neutralization (preserving +91 phone numbers)
    monkeypatch.setenv("ADMIN_PASSCODE", "fortivex-admin-2026")
    expected = st_app.get_configured_admin_passcode()
    assert expected == "fortivex-admin-2026"
    assert st_app.verify_admin_passcode("fortivex-admin-2026", expected) is True
    assert st_app.verify_admin_passcode("wrong-pass", expected) is False

    df = pd.DataFrame([{"name": "=CMD()", "phone": "+91 9876543210", "email": "@bad"}])
    clean_df = st_app.sanitize_csv_dataframe(df)
    assert clean_df.iloc[0]["name"] == "'=CMD()"
    assert clean_df.iloc[0]["email"] == "'@bad"
    assert clean_df.iloc[0]["phone"] == "+91 9876543210"

