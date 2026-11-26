"""AI Conversation Engine supporting Groq, Gemini, and grounded Local Fallback mode."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from database.database import DatabaseManager
from modules.appointment_manager import AppointmentManager
from modules.lead_manager import FIELD_LABELS, LeadManager
from modules.rag_engine import RAGEngine
from modules.sentiment import analyze_sentiment


load_dotenv()


def _is_valid_key(val: Optional[str]) -> bool:
    if not val:
        return False
    stripped = val.strip()
    if not stripped or stripped.startswith("your_") or stripped == "changeme":
        return False
    return True


class SalesAIEngine:
    """Orchestrates RAG retrieval, LLM generation (with fallback), lead qualification, and sentiment."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        rag: Optional[RAGEngine] = None,
        force_fallback: bool = False,
        provider_override: Optional[str] = None,
    ) -> None:
        self.db = db or DatabaseManager()
        self.rag = rag or RAGEngine()
        self.lead_manager = LeadManager(self.db)
        self.appointment_manager = AppointmentManager(self.db)

        self.force_fallback = force_fallback
        self.provider_pref = (provider_override or os.getenv("LLM_PROVIDER", "auto")).lower()

        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        self.groq_client = None
        if not self.force_fallback and _is_valid_key(self.groq_api_key):
            try:
                from groq import Groq

                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception:
                self.groq_client = None

        self.active_provider = self._resolve_provider()

    def _resolve_provider(self) -> str:
        if self.force_fallback or self.provider_pref == "fallback":
            return "local_fallback"
        if self.provider_pref == "groq":
            return "groq" if self.groq_client is not None else "local_fallback"
        if self.provider_pref == "gemini":
            return "gemini" if _is_valid_key(self.gemini_api_key) else "local_fallback"

        # Auto selection: prefer Groq if configured, else Gemini, else local_fallback
        if self.groq_client is not None:
            return "groq"
        if _is_valid_key(self.gemini_api_key):
            return "gemini"
        return "local_fallback"

    def _build_system_prompt(
        self,
        language: str,
        rag_context: str,
        lead: Dict[str, Any],
        missing_fields: List[str],
    ) -> str:
        captured_summary = {
            k: lead.get(k) for k in FIELD_LABELS if lead.get(k)
        }
        missing_labels = [FIELD_LABELS[f] for f in missing_fields[:3]]
        lang_instruction = (
            "Respond strictly in natural, polite Hindi (Devanagari or simple Hindi conversational phrasing)."
            if language == "Hindi"
            else "Respond in clear, professional, conversational English."
        )

        kb_block = (
            rag_context
            if rag_context
            else "NO RELEVANT KNOWLEDGE FOUND IN COMPANY DOCUMENTS."
        )

        return (
            "You are SalesPilot AI, an autonomous, professional AI Sales Representative for "
            "Fortivex AI Private Limited.\n\n"
            "CORE RULES:\n"
            "1. Be helpful, polite, concise (3-5 sentences max), and conversational.\n"
            "2. Ground ALL product, service, pricing, timeline, security, and FAQ claims ONLY in "
            "the VERIFIED COMPANY KNOWLEDGE below.\n"
            "3. HONESTY GUARDRAIL: If the customer asks a factual question about Fortivex AI, "
            "pricing, unsupported integrations, or policies that is NOT covered by the VERIFIED "
            "COMPANY KNOWLEDGE, honestly state that you do not have that verified information "
            "and offer to have a sales specialist confirm it. NEVER invent prices or facts.\n"
            "4. Never claim that an external calendar appointment has been confirmed; only refer "
            "to submitting an appointment request for sales team confirmation.\n"
            "5. Ask 1 or 2 relevant follow-up qualification questions naturally instead of giving "
            "generic answers.\n"
            f"6. Language requirement: {lang_instruction}\n\n"
            f"CAPTURED LEAD PROFILE SO FAR:\n{json.dumps(captured_summary, ensure_ascii=False)}\n\n"
            f"PRIORITY MISSING QUALIFICATION DETAILS TO ASK ABOUT NEXT:\n{missing_labels}\n\n"
            f"VERIFIED COMPANY KNOWLEDGE (RAG RETRIEVAL):\n{kb_block}"
        )

    def _call_groq(self, messages: List[Dict[str, str]], temperature: float = 0.4, max_tokens: int = 350) -> str:
        if self.groq_client is None:
            raise RuntimeError("Groq client not configured")
        completion = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = completion.choices[0].message.content if completion.choices else ""
        if not text or not text.strip():
            raise RuntimeError("Empty response from Groq")
        return text.strip()

    def _call_gemini(self, system_prompt: str, history: List[Dict[str, str]], user_message: str) -> str:
        if not _is_valid_key(self.gemini_api_key):
            raise RuntimeError("Gemini API key not configured")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        )
        contents = []
        for msg in history[-8:]:
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.get("message", "")}]})
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 350},
        }
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("No candidates returned by Gemini")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise RuntimeError("Empty response from Gemini")
        return text

    @staticmethod
    def _looks_like_unknown_factual_question(user_message: str) -> bool:
        lower = user_message.lower()
        question_signals = ["?", "do you", "what is", "how much", "price", "cost", "support", "integrate", "offer", "kya", "kitna"]
        return any(sig in lower for sig in question_signals)

    @staticmethod
    def _is_greeting_only(user_message: str) -> bool:
        clean = re.sub(r"[^a-zA-Z\u0900-\u097F\s]", "", user_message).strip().lower()
        return clean in {
            "hi", "hello", "hey", "namaste", "नमस्ते", "नमस्कार",
            "good morning", "good afternoon", "good evening",
        }

    def _generate_local_fallback_reply(
        self,
        user_message: str,
        language: str,
        rag_hits: List[Dict[str, Any]],
        lead: Dict[str, Any],
        missing_fields: List[str],
        appointment_created: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Deterministic, RAG-grounded conversational sales response when no external LLM API is used."""
        is_hindi = language == "Hindi"

        if appointment_created and appointment_created.get("success"):
            appt = appointment_created["appointment"]
            if is_hindi:
                return (
                    f"धन्यवाद {appt['customer_name']}! आपका अपॉइंटमेंट अनुरोध {appt['preferred_date']} को "
                    f"{appt['preferred_time']} बजे के लिए सेव कर लिया गया है (स्थिति: पुष्टि लंबित)। "
                    "क्या आपकी कंपनी की कोई और आवश्यकता है जिसे मैं समरी में जोड़ूं?"
                )
            return (
                f"Thank you, {appt['customer_name']}! I have saved your appointment request for "
                f"{appt['preferred_date']} at {appt['preferred_time']} (status: Pending Sales Team Confirmation — "
                "not an external calendar booking). Is there anything else about your requirements you would like me to note?"
            )

        if self._is_greeting_only(user_message):
            if is_hindi:
                return (
                    "नमस्ते! मैं SalesPilot AI हूँ, Fortivex AI Private Limited की ओर से। "
                    "मैं आपकी AI ऑटोमेशन और सेल्स वॉइस एजेंट ज़रूरतों में मदद कर सकता हूँ। "
                    "आप किस प्रकार की बिज़नेस प्रक्रिया को ऑटोमेट करना चाहते हैं?"
                )
            return (
                "Hello! Welcome to Fortivex AI Private Limited. I am SalesPilot AI, your sales assistant. "
                "I can share details on our AI Automation Services, Voice Agents, RAG systems, and pricing, "
                "and help submit a demo appointment request. What business challenge are you looking to automate?"
            )

        parts: List[str] = []
        cust_name = str(lead.get("name") or "").strip()
        greeting_prefix = f"{cust_name}, " if cust_name else ""

        # 1. Grounded RAG knowledge response
        if rag_hits:
            top_hit = rag_hits[0]
            # Extract clean bullet/sentence summary from top 1-2 chunks
            snippet_lines = [
                ln.strip("-# ").strip()
                for ln in str(top_hit["content"]).splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            concise_info = " ".join(snippet_lines[:3])[:380]
            if is_hindi:
                parts.append(
                    f"{greeting_prefix}हमारी कंपनी जानकारी ({top_hit['section']}) के अनुसार: {concise_info}"
                )
            else:
                parts.append(
                    f"{greeting_prefix}based on our verified company knowledge ({top_hit['section']}): {concise_info}"
                    if cust_name
                    else f"Based on our verified company knowledge ({top_hit['section']}): {concise_info}"
                )
        elif self._looks_like_unknown_factual_question(user_message):
            if is_hindi:
                parts.append(
                    f"{greeting_prefix}ईमानदारी से कहूँ तो मेरे पास इस विशिष्ट विषय पर सत्यापित कंपनी जानकारी उपलब्ध नहीं है, "
                    "इसलिए मैं कोई गलत जानकारी नहीं दूँगा। मैं इसे हमारी सेल्स टीम के फॉलो-अप के लिए नोट कर सकता हूँ।"
                )
            else:
                parts.append(
                    f"{greeting_prefix}to be completely transparent, I do not have verified information on that specific detail "
                    "in our company knowledge base, so I will not guess. I can note this for our sales specialist to confirm."
                    if cust_name
                    else "To be completely transparent, I do not have verified information on that specific detail "
                    "in our company knowledge base, so I will not guess. I can note this for our sales specialist to confirm."
                )
        else:
            known_context = []
            if lead.get("budget"):
                known_context.append(f"budget ({lead['budget']})")
            if lead.get("purchase_timeline"):
                known_context.append(f"timeline ({lead['purchase_timeline']})")
            if lead.get("product_interest"):
                known_context.append(f"interest ({lead['product_interest']})")
            ctx_note = f" (noted: {', '.join(known_context)})" if known_context else ""
            if is_hindi:
                parts.append(f"धन्यवाद {cust_name + ' जी, ' if cust_name else ''}मैंने आपकी जानकारी रिकॉर्ड कर ली है।")
            else:
                parts.append(f"Thank you{', ' + cust_name if cust_name else ''}! I have updated your inquiry profile{ctx_note}.")

        # 2. Targeted qualification follow-up questions
        if missing_fields:
            question_map_en = {
                "business_requirement": "Could you briefly share the main workflow or business requirement you want to automate?",
                "product_interest": "Which solution interests you most — AI Automation Services, the SalesPilot Voice Agent, or a RAG knowledge assistant?",
                "budget": "Do you have an estimated monthly or project budget in mind?",
                "purchase_timeline": "What is your target timeline to get started?",
                "name": "May I have your full name?",
                "email": "What is the best email address to reach you?",
                "phone": "Could you also share your phone number for follow-up?",
            }
            question_map_hi = {
                "business_requirement": "कृपया बताएं कि आप अपनी कंपनी में किस कार्य को ऑटोमेट करना चाहते हैं?",
                "product_interest": "आप किस सेवा में अधिक रुचि रखते हैं — AI ऑटोमेशन, वॉइस एजेंट, या RAG सिस्टम?",
                "budget": "इस प्रोजेक्ट के लिए आपका अनुमानित बजट क्या है?",
                "purchase_timeline": "आप इसे कब तक शुरू करने की योजना बना रहे हैं?",
                "name": "कृपया आपका शुभ नाम बताएं?",
                "email": "संपर्क के लिए आपका ईमेल पता क्या है?",
                "phone": "कृपया अपना फ़ोन नंबर भी साझा करें?",
            }
            q_map = question_map_hi if is_hindi else question_map_en
            priority_order = [
                "business_requirement",
                "budget",
                "purchase_timeline",
                "name",
                "email",
                "phone",
                "product_interest",
            ]
            next_qs = [q_map[f] for f in priority_order if f in missing_fields][:2]
            if next_qs:
                parts.append(" ".join(next_qs))
        else:
            if is_hindi:
                parts.append(
                    "हमारे पास आपकी सभी मुख्य जानकारी आ गई है! क्या आप किसी डेमो या कंसल्टेशन के लिए अपॉइंटमेंट अनुरोध सबमिट करना चाहेंगे? "
                    "कृपया अपनी पसंदीदा तारीख और समय बताएं या नीचे दिए गए अपॉइंटमेंट फ़ॉर्म का उपयोग करें।"
                )
            else:
                parts.append(
                    "I have all your key qualification details saved! Would you like to submit an appointment request for a consultation? "
                    "You can share your preferred date and time or use the Appointment Request panel."
                )

        return " ".join(parts)

    def _try_inline_appointment_capture(
        self, session_id: str, user_message: str, lead: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect explicit date + time + appointment intent inside chat and save validated request."""
        lower = user_message.lower()
        appt_words = ["appointment", "book", "schedule", "demo", "meeting", "slot", "call on"]
        if not any(w in lower for w in appt_words):
            return None

        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b", user_message)
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|\d{1,2}\s*(?:AM|PM|am|pm))\b", user_message)
        if not (date_match and time_match):
            return None

        name = lead.get("name") or ""
        contact = lead.get("email") or lead.get("phone") or ""
        if not (name and contact):
            return None

        return self.appointment_manager.submit_appointment_request(
            session_id=session_id,
            customer_name=name,
            contact_details=contact,
            preferred_date=date_match.group(1),
            preferred_time=time_match.group(1),
            notes="Captured inline from conversation",
        )

    MAX_MESSAGE_CHARS = 2000

    def process_turn(
        self,
        session_id: str,
        user_message: str,
        language: str = "English",
    ) -> Dict[str, Any]:
        """Execute a complete conversational turn: sentiment, lead capture, RAG, LLM/fallback, persistence."""
        clean_msg = (user_message or "").strip()
        if not clean_msg:
            raise ValueError("Customer message cannot be empty.")
        if len(clean_msg) > self.MAX_MESSAGE_CHARS:
            raise ValueError(f"Message exceeds maximum allowed length of {self.MAX_MESSAGE_CHARS} characters.")

        # 1. Sentiment analysis
        sentiment_res = analyze_sentiment(clean_msg)
        sentiment_label = str(sentiment_res["label"])
        sentiment_score = float(sentiment_res["score"])

        # 2. Save user message
        self.db.save_message(
            session_id=session_id,
            role="user",
            message=clean_msg,
            language=language,
            sentiment=sentiment_label,
        )

        # 3. Extract & persist lead fields
        lead = self.lead_manager.update_from_message(
            session_id=session_id,
            user_message=clean_msg,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            language=language,
        )
        missing_fields = self.lead_manager.get_missing_fields(lead)

        # 4. Inline appointment request check (if customer gave full details + date + time)
        appointment_result = self._try_inline_appointment_capture(session_id, clean_msg, lead)

        # 5. RAG retrieval
        rag_context, rag_hits = self.rag.get_context(clean_msg, top_k=3)

        # 6. History before current message for LLM context
        full_history = self.db.get_conversation_history(session_id)
        prior_history = full_history[:-1]

        # 7. Generate response via configured provider with graceful fallback
        provider_used = self.active_provider
        is_fallback = provider_used == "local_fallback"
        fallback_reason = "No external LLM API key configured (Local Fallback Mode)." if is_fallback else ""
        reply = ""

        if provider_used == "groq":
            try:
                sys_prompt = self._build_system_prompt(language, rag_context, lead, missing_fields)
                llm_msgs = [{"role": "system", "content": sys_prompt}]
                for h in prior_history[-10:]:
                    llm_msgs.append({"role": h["role"], "content": h["message"]})
                extra_note = ""
                if appointment_result and appointment_result.get("success"):
                    extra_note = f"\n[SYSTEM NOTE: {appointment_result['message']}]"
                llm_msgs.append({"role": "user", "content": clean_msg + extra_note})
                reply = self._call_groq(llm_msgs)
            except Exception as exc:
                fallback_reason = f"Groq API unavailable ({exc}); switched to Local Fallback."
                provider_used = "local_fallback"
                is_fallback = True

        elif provider_used == "gemini":
            try:
                sys_prompt = self._build_system_prompt(language, rag_context, lead, missing_fields)
                reply = self._call_gemini(sys_prompt, prior_history, clean_msg)
            except Exception as exc:
                fallback_reason = f"Gemini API unavailable ({exc}); switched to Local Fallback."
                provider_used = "local_fallback"
                is_fallback = True

        if not reply:
            reply = self._generate_local_fallback_reply(
                user_message=clean_msg,
                language=language,
                rag_hits=rag_hits,
                lead=lead,
                missing_fields=missing_fields,
                appointment_created=appointment_result,
            )

        # 8. Save assistant response
        self.db.save_message(
            session_id=session_id,
            role="assistant",
            message=reply,
            language=language,
            sentiment=sentiment_label,
        )

        return {
            "session_id": session_id,
            "reply": reply,
            "provider_used": provider_used,
            "is_fallback": is_fallback,
            "fallback_reason": fallback_reason,
            "sentiment": sentiment_res,
            "lead": lead,
            "missing_fields": missing_fields,
            "rag_hits": rag_hits,
            "appointment_result": appointment_result,
        }

    def generate_structured_summary(
        self,
        history: List[Dict[str, Any]],
        lead: Dict[str, Any],
        appointments: List[Dict[str, Any]],
    ) -> Optional[Dict[str, str]]:
        """Use LLM to produce a structured JSON call summary if available; returns None on fallback/error."""
        if self.active_provider == "local_fallback":
            return None

        transcript = "\n".join(f"{m['role'].upper()}: {m['message']}" for m in history)
        prompt = (
            "Summarize this sales conversation into valid JSON with EXACT keys: "
            "customer_requirements, product_interest, budget, purchase_timeline, "
            "recommended_next_action, summary_text.\n"
            f"Lead record: {json.dumps(lead, ensure_ascii=False)}\n"
            f"Appointments: {json.dumps(appointments, ensure_ascii=False)}\n"
            f"Transcript:\n{transcript}"
        )
        try:
            if self.active_provider == "groq" and self.groq_client is not None:
                completion = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "Return ONLY a valid JSON object."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=400,
                    response_format={"type": "json_object"},
                )
                raw = completion.choices[0].message.content or "{}"
                return json.loads(raw)
            if self.active_provider == "gemini" and _is_valid_key(self.gemini_api_key):
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.gemini_model}:generateContent?key={self.gemini_api_key}"
                )
                payload = {
                    "systemInstruction": {"parts": [{"text": "Return ONLY a valid JSON object."}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 400,
                        "responseMimeType": "application/json",
                    },
                }
                resp = requests.post(url, json=payload, timeout=15)
                resp.raise_for_status()
                parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                raw = "".join(p.get("text", "") for p in parts).strip() or "{}"
                return json.loads(raw)
        except Exception:
            return None
        return None
