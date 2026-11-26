"""SalesPilot AI – Autonomous AI Sales Voice Agent (Streamlit Application)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, time, timedelta
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from database.database import DatabaseManager
from modules.ai_engine import SalesAIEngine
from modules.appointment_manager import AppointmentManager
from modules.lead_manager import FIELD_LABELS, LeadManager
from modules.rag_engine import RAGEngine
from modules.speech_engine import SpeechEngine
from modules.summarizer import CallSummarizer


st.set_page_config(
    page_title="SalesPilot AI – Fortivex AI Voice Sales Agent",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource
def get_core_services() -> Dict[str, Any]:
    db = DatabaseManager()
    rag = RAGEngine()
    speech = SpeechEngine()
    lead_mgr = LeadManager(db)
    appt_mgr = AppointmentManager(db)
    summarizer = CallSummarizer(db)
    return {
        "db": db,
        "rag": rag,
        "speech": speech,
        "lead_mgr": lead_mgr,
        "appt_mgr": appt_mgr,
        "summarizer": summarizer,
    }


def get_ai_engine(services: Dict[str, Any], force_fallback: bool) -> SalesAIEngine:
    cache_key = f"engine_{'fallback' if force_fallback else 'auto'}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = SalesAIEngine(
            db=services["db"],
            rag=services["rag"],
            force_fallback=force_fallback,
        )
    return st.session_state[cache_key]


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess-{uuid.uuid4().hex[:8]}"
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    if "last_tts_audio" not in st.session_state:
        st.session_state.last_tts_audio = None
    if "last_tts_error" not in st.session_state:
        st.session_state.last_tts_error = None
    if "last_rag_hits" not in st.session_state:
        st.session_state.last_rag_hits = []
    if "last_sentiment" not in st.session_state:
        st.session_state.last_sentiment = {"label": "Neutral", "score": 0.0}
    if "last_summary" not in st.session_state:
        st.session_state.last_summary = None
    if "last_provider_info" not in st.session_state:
        st.session_state.last_provider_info = ""


def handle_user_turn(
    ai_engine: SalesAIEngine,
    speech_engine: SpeechEngine,
    session_id: str,
    user_text: str,
    language: str,
    enable_tts: bool,
) -> None:
    result = ai_engine.process_turn(
        session_id=session_id,
        user_message=user_text,
        language=language,
    )
    st.session_state.last_rag_hits = result.get("rag_hits", [])
    st.session_state.last_sentiment = result.get("sentiment", {"label": "Neutral", "score": 0.0})
    st.session_state.last_provider_info = (
        f"{result['provider_used']}"
        + (f" ({result['fallback_reason']})" if result.get("fallback_reason") else "")
    )

    if enable_tts:
        audio_bytes, tts_err = speech_engine.synthesize_speech(result["reply"], language=language)
        st.session_state.last_tts_audio = audio_bytes
        st.session_state.last_tts_error = tts_err
    else:
        st.session_state.last_tts_audio = None
        st.session_state.last_tts_error = None


def render_sentiment_badge(label: str, score: float) -> None:
    color_map = {
        "Positive": ("#e6f5ec", "#1d7a4d"),
        "Neutral": ("#eef3fb", "#1f5fbf"),
        "Negative": ("#fdecea", "#b3261e"),
    }
    bg, fg = color_map.get(label, ("#eef3fb", "#1f5fbf"))
    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {fg}44;border-radius:8px;padding:10px 14px;margin-bottom:10px;">
            <span style="font-size:12px;color:#555;text-transform:uppercase;">Customer Sentiment</span><br>
            <strong style="font-size:18px;color:{fg};">{label}</strong>
            <span style="font-size:13px;color:#555;"> (score: {score:+.2f})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_page(services: Dict[str, Any], language: str, enable_tts: bool, force_fallback: bool) -> None:
    db: DatabaseManager = services["db"]
    speech: SpeechEngine = services["speech"]
    appt_mgr: AppointmentManager = services["appt_mgr"]
    summarizer: CallSummarizer = services["summarizer"]
    ai_engine = get_ai_engine(services, force_fallback=force_fallback)
    session_id = st.session_state.session_id

    st.title("🎙️ SalesPilot AI – Autonomous AI Sales Voice Agent")
    st.caption(
        "Fortivex AI Private Limited | Grounded RAG Sales Assistant, Lead Qualification, "
        "Sentiment Analysis, Appointment Requests & Call Summaries"
    )

    if ai_engine.active_provider == "local_fallback":
        st.info(
            "**Local Fallback Mode Active**: Responding using local RAG knowledge retrieval and "
            "deterministic sales qualification rules (no external LLM API required)."
        )
    else:
        st.success(f"**Live AI Engine**: Connected via `{ai_engine.active_provider.upper()}` with RAG grounding.")

    col_chat, col_intel = st.columns([1.35, 1.0], gap="large")

    with col_chat:
        st.subheader("1. Voice & Text Sales Conversation")

        # Quick demo starters for easy interview walkthrough
        st.markdown("**Quick Demo Prompts:**")
        q1, q2, q3 = st.columns(3)
        if q1.button("AI Automation Inquiry", use_container_width=True):
            handle_user_turn(
                ai_engine,
                speech,
                session_id,
                "I am interested in your AI automation services.",
                language,
                enable_tts,
            )
            st.rerun()
        if q2.button("Provide Lead Details", use_container_width=True):
            sample = (
                "My name is Aarav Mehta, email aarav@techcraft.in, phone +91 9876543210. "
                "We want to automate inbound lead qualification for 15 sales reps. "
                "Budget is INR 65,000/month and timeline is this month."
            )
            handle_user_turn(ai_engine, speech, session_id, sample, language, enable_tts)
            st.rerun()
        if q3.button("Ask Out-of-Scope Test", use_container_width=True):
            handle_user_turn(
                ai_engine,
                speech,
                session_id,
                "Do you sell physical telecom SIM boxes or hardware servers?",
                language,
                enable_tts,
            )
            st.rerun()

        # Voice input controls
        with st.expander("🎤 Voice Input (Microphone or WAV Upload)", expanded=True):
            vcol1, vcol2 = st.columns(2)
            recorded_audio = None
            if hasattr(st, "audio_input"):
                recorded_audio = vcol1.audio_input("Record voice message")
            else:
                vcol1.caption("Live mic recording requires Streamlit >= 1.40; use WAV upload.")
            uploaded_audio = vcol2.file_uploader("Or upload WAV audio", type=["wav"])

            audio_source = recorded_audio or uploaded_audio
            if audio_source is not None:
                raw_bytes = audio_source.getvalue()
                audio_hash = hashlib.md5(raw_bytes).hexdigest()
                if audio_hash != st.session_state.last_audio_hash:
                    st.session_state.last_audio_hash = audio_hash
                    with st.spinner("Transcribing voice input..."):
                        transcript, stt_err = speech.transcribe_audio_bytes(raw_bytes, language=language)
                    if stt_err:
                        st.warning(stt_err)
                    elif transcript:
                        st.success(f"Transcribed ({language}): \"{transcript}\"")
                        handle_user_turn(ai_engine, speech, session_id, transcript, language, enable_tts)
                        st.rerun()

        # Render conversation history
        history = db.get_conversation_history(session_id)
        if not history:
            with st.chat_message("assistant"):
                if language == "Hindi":
                    st.write(
                        "नमस्ते! मैं SalesPilot AI हूँ। आप हमारे AI ऑटोमेशन, प्राइसिंग या डेमो के बारे में पूछ सकते हैं।"
                    )
                else:
                    st.write(
                        "Hello! I am SalesPilot AI from Fortivex AI Private Limited. "
                        "Ask me about our AI automation services, pricing, or share your requirements to get started."
                    )
        else:
            for item in history:
                with st.chat_message(item["role"]):
                    st.write(item["message"])
                    st.caption(f"{item['created_at']} | Language: {item['language']} | Sentiment: {item['sentiment']}")

        # Audio playback control for latest AI response
        if enable_tts:
            if st.session_state.last_tts_audio:
                st.markdown("🔊 **Spoken AI Response (Text-to-Speech)**")
                st.audio(st.session_state.last_tts_audio, format="audio/mp3")
            elif st.session_state.last_tts_error:
                st.caption(f"TTS notice: {st.session_state.last_tts_error}")

        # Text input
        user_prompt = st.chat_input(
            "Type your message (e.g., 'I am interested in your AI automation services.')..."
        )
        if user_prompt:
            handle_user_turn(ai_engine, speech, session_id, user_prompt, language, enable_tts)
            st.rerun()

    with col_intel:
        st.subheader("2. Live Lead, Sentiment & Booking")

        render_sentiment_badge(
            str(st.session_state.last_sentiment.get("label", "Neutral")),
            float(st.session_state.last_sentiment.get("score", 0.0)),
        )

        lead = db.get_lead_by_session(session_id) or {}
        score = int(lead.get("lead_score") or 0)
        status = lead.get("qualification_status") or "New"

        m1, m2 = st.columns(2)
        m1.metric("Lead Score", f"{score}/100")
        m2.metric("Qualification Status", status)

        with st.expander("📋 Captured Lead Details (SQLite CRM)", expanded=True):
            for field_key, label in FIELD_LABELS.items():
                val = lead.get(field_key) or "—"
                st.write(f"**{label}:** {val}")
            missing = LeadManager.get_missing_fields(lead)
            if missing:
                st.caption("Missing fields: " + ", ".join(FIELD_LABELS[m] for m in missing))
            else:
                st.success("All qualification fields captured!")

        with st.expander("📚 RAG Knowledge Sources Retrieved", expanded=False):
            hits = st.session_state.last_rag_hits
            if not hits:
                st.caption("No knowledge chunks matched the last message above threshold.")
            else:
                for h in hits:
                    st.markdown(f"**{h['section']}** (relevance: `{h['score']}`)")
                    st.caption(h["content"])

        with st.expander("📅 Request an Appointment", expanded=True):
            st.caption(
                "⚠️ Saves an **appointment request** in SQLite for sales team confirmation "
                "(not a confirmed external calendar booking)."
            )
            with st.form("appointment_form", clear_on_submit=False):
                default_name = lead.get("name") or ""
                default_contact = lead.get("email") or lead.get("phone") or ""
                appt_name = st.text_input("Customer Name *", value=default_name)
                appt_contact = st.text_input("Email or Phone *", value=default_contact)
                appt_date = st.date_input(
                    "Preferred Date *",
                    value=date.today() + timedelta(days=1),
                    min_value=date.today(),
                )
                appt_time = st.time_input("Preferred Time *", value=time(15, 0))
                appt_notes = st.text_input("Notes (optional)", placeholder="Product demo / pricing review")
                submitted = st.form_submit_button("Submit Appointment Request", use_container_width=True)

            if submitted:
                res = appt_mgr.submit_appointment_request(
                    session_id=session_id,
                    customer_name=appt_name,
                    contact_details=appt_contact,
                    preferred_date=appt_date.isoformat(),
                    preferred_time=appt_time.strftime("%H:%M"),
                    notes=appt_notes,
                )
                if res["success"]:
                    st.success(res["message"])
                else:
                    st.error(res["error"])

        st.divider()
        if st.button("📝 End Call & Generate Summary", type="primary", use_container_width=True):
            summary = summarizer.generate_and_save_summary(session_id, ai_engine=ai_engine)
            st.session_state.last_summary = summary

        saved_summary = st.session_state.last_summary or db.get_call_summary(session_id)
        if saved_summary:
            with st.expander("🧾 Saved Call Summary", expanded=True):
                st.write(f"**Requirements:** {saved_summary['customer_requirements']}")
                st.write(f"**Product Interest:** {saved_summary['product_interest']}")
                st.write(f"**Budget:** {saved_summary['budget']}")
                st.write(f"**Timeline:** {saved_summary['purchase_timeline']}")
                st.write(f"**Sentiment:** {saved_summary['sentiment']}")
                st.write(f"**Qualification:** {saved_summary['qualification_status']} ({saved_summary['lead_score']}/100)")
                st.write(f"**Recommended Next Action:** {saved_summary['recommended_next_action']}")
                st.info(saved_summary["summary_text"])


def render_admin_dashboard(services: Dict[str, Any]) -> None:
    db: DatabaseManager = services["db"]
    st.title("📊 SalesPilot AI – Admin CRM & Insights Dashboard")
    st.caption("Live data backed by local SQLite CRM database")

    metrics = db.get_dashboard_metrics()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Leads", metrics["total_leads"])
    k2.metric("Qualified Leads", metrics["qualified_leads"])
    k3.metric("Appointment Requests", metrics["total_appointments"])
    k4.metric("Call Summaries", metrics["total_summaries"])
    k5.metric("Conversation Turns", metrics["total_messages"])

    leads = db.get_all_leads()
    appointments = db.get_all_appointments()
    summaries = db.get_all_summaries()

    # Business Insights Charts
    st.subheader("Business Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Customer Sentiment Distribution**")
        sent_data = metrics["sentiment_breakdown"]
        if sent_data:
            fig, ax = plt.subplots(figsize=(5, 3))
            labels = list(sent_data.keys())
            values = list(sent_data.values())
            colors = [
                "#1d7a4d" if l == "Positive" else "#b3261e" if l == "Negative" else "#1f5fbf"
                for l in labels
            ]
            ax.bar(labels, values, color=colors)
            ax.set_ylabel("Leads")
            ax.yaxis.get_major_locator().set_params(integer=True)
            st.pyplot(fig, clear_figure=True)
        else:
            st.caption("No lead sentiment data recorded yet.")

    with c2:
        st.markdown("**Lead Qualification Funnel**")
        status_data = metrics["status_breakdown"]
        if status_data:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.bar(list(status_data.keys()), list(status_data.values()), color="#3b6fb6")
            ax2.set_ylabel("Leads")
            ax2.yaxis.get_major_locator().set_params(integer=True)
            plt.xticks(rotation=15, ha="right")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.caption("No lead status data recorded yet.")

    tab_leads, tab_appts, tab_summaries, tab_history = st.tabs(
        ["Leads CRM", "Appointment Requests", "Call Summaries", "Conversation History"]
    )

    with tab_leads:
        if leads:
            df_leads = pd.DataFrame(leads)
            st.dataframe(df_leads, use_container_width=True)
            st.download_button(
                "Export Leads CSV",
                data=df_leads.to_csv(index=False).encode("utf-8"),
                file_name="salespilot_leads.csv",
                mime="text/csv",
            )
        else:
            st.info("No leads captured yet. Start a conversation on the Sales Agent page.")

    with tab_appts:
        st.caption("All entries below are **Appointment Requests** pending sales team confirmation.")
        if appointments:
            st.dataframe(pd.DataFrame(appointments), use_container_width=True)
        else:
            st.info("No appointment requests submitted yet.")

    with tab_summaries:
        if summaries:
            st.dataframe(pd.DataFrame(summaries), use_container_width=True)
            for s in summaries:
                with st.expander(f"Session `{s['session_id']}` — {s['qualification_status']} | {s['sentiment']}"):
                    st.write(s["summary_text"])
                    st.write(f"**Next Action:** {s['recommended_next_action']}")
        else:
            st.info("No call summaries generated yet.")

    with tab_history:
        sessions = db.get_all_sessions()
        if sessions:
            selected_sess = st.selectbox("Select Session ID", sessions)
            msgs = db.get_conversation_history(selected_sess)
            for m in msgs:
                with st.chat_message(m["role"]):
                    st.write(m["message"])
                    st.caption(f"{m['created_at']} | {m['language']} | Sentiment: {m['sentiment']}")
        else:
            st.info("No conversation history stored yet.")


def main() -> None:
    init_session_state()
    services = get_core_services()

    with st.sidebar:
        st.header("SalesPilot AI")
        st.caption("Fortivex AI Private Limited")
        page = st.radio(
            "Workspace View",
            ["Sales Voice & Chat Agent", "Admin CRM Dashboard"],
            index=0,
        )
        st.divider()
        language = st.selectbox(
            "Conversation Language (Tested)",
            SpeechEngine.get_supported_languages(),
            index=0,
        )
        enable_tts = st.checkbox("Enable Text-to-Speech (gTTS)", value=True)
        force_fallback = st.checkbox("Simulate Local Fallback Mode (No API Key)", value=False)
        st.divider()
        st.caption(f"Active Session: `{st.session_state.session_id}`")
        if st.button("🔄 Start New Session", use_container_width=True):
            st.session_state.session_id = f"sess-{uuid.uuid4().hex[:8]}"
            st.session_state.last_audio_hash = None
            st.session_state.last_tts_audio = None
            st.session_state.last_tts_error = None
            st.session_state.last_rag_hits = []
            st.session_state.last_sentiment = {"label": "Neutral", "score": 0.0}
            st.session_state.last_summary = None
            st.rerun()

    if page == "Sales Voice & Chat Agent":
        render_agent_page(services, language, enable_tts, force_fallback)
    else:
        render_admin_dashboard(services)


if __name__ == "__main__":
    main()
