# SalesPilot AI – Autonomous AI Sales Voice Agent: Project Report

**Organization Submission:** Fortivex AI Private Limited  
**Project Title:** SalesPilot AI – Autonomous AI Sales Voice Agent  

---

## 1. Business Problem & Proposed Solution
- **Problem Statement:** Sales teams lose qualified inbound and outbound leads due to slow follow-up, limited business hours, inconsistent qualification scripts, manual CRM entry errors, and lack of structured visibility into customer objections and sentiment.
- **Proposed Solution:** **SalesPilot AI** is an autonomous AI sales assistant that interacts with prospects via **voice and text**, grounds product and pricing answers in company documents via **RAG**, qualifies leads across 7 structured fields, tracks customer **sentiment**, captures validated **appointment requests**, and generates **structured call summaries** in SQLite for an **Admin CRM & Business Insights Dashboard**.

## 2. Core Functional Architecture (Implemented MVP)
- **Autonomous AI Sales Agent (`modules/ai_engine.py`):** Supports Groq (`llama-3.3-70b-versatile`) and Google Gemini (`gemini-2.0-flash`) via environment variables, plus a deterministic **Local Retrieval Fallback Mode** when external API keys are unavailable. Guides conversations with targeted follow-up qualification questions rather than generic responses.
- **Speech-to-Text & Text-to-Speech (`modules/speech_engine.py`):** Uses `SpeechRecognition` (`recognize_google`) for microphone/WAV transcription and `gTTS` for MP3 spoken playback, with graceful offline error handling.
- **Contextual Memory & Multilingual Communication:** Maintains persistent turn-by-turn conversation history and evolving lead profiles in SQLite (`database/database.py`), with tested **English** (`en-IN`/`en`) and **Hindi** (`hi-IN`/`hi`) support across text, STT, and TTS.
- **RAG-Based Knowledge Retrieval (`modules/rag_engine.py`):** Splits `data/company_knowledge.txt` into section-aware chunks, indexes vectors with `scikit-learn` TF-IDF + `faiss-cpu` (cosine similarity), and enforces an anti-hallucination threshold so unavailable facts are never fabricated.
- **Lead Qualification & Sentiment Analysis (`modules/lead_manager.py`, `modules/sentiment.py`):** Captures Customer Name, Email, Phone, Product/Service Interest, Budget, Business Requirement, and Purchase Timeline; computes a deterministic 0–100 score (`Qualified`, `Warm (In Progress)`, `New`) alongside per-turn and conversation-level sentiment (`Positive`, `Neutral`, `Negative`).
- **Appointment Booking & CRM Integration (`modules/appointment_manager.py`):** Validates customer name, email/phone, non-past date, and time, storing requests in SQLite as `Requested (Pending Confirmation)`—explicitly distinguished from a live external calendar booking.
- **Call Summarisation & Business Insights (`modules/summarizer.py`, `app.py`):** Produces structured summaries (requirements, interest, budget, timeline, sentiment, qualification status, recommended next action) and visualizes KPIs, sentiment distribution, and qualification funnel charts in the Admin CRM Dashboard.
- **Optional REST API Layer (`backend/main.py`):** FastAPI service exposing `/health`, `/chat`, `/appointments`, `/summary/{session_id}`, and `/dashboard` bridged to the same modules.

## 3. Commercial Feasibility & Operational Costs
- **Business Value:** 24×7 instant lead engagement, consistent qualification, automated CRM logging, and actionable analytics so human sales reps focus on high-intent Qualified leads.
- **Operational Cost Model:** `Cost/min = STT + LLM tokens + TTS + Telephony + Infrastructure`. In production, indicative API voice costs (~USD 0.03–0.15/min depending on model, TTS voice tier, and telephony) can be comparable to Indian telecaller hourly wages; therefore, commercial viability relies on **speed-to-lead conversion uplift**, 24×7 scalability, short prompts/RAG chunks, FAQ caching, and Indian STT/TTS optimization.
- **Pricing Model Options:** SaaS monthly tiers (Starter / Growth / Enterprise with included conversation minutes) or usage-based per-minute billing.

## 4. Data Security & Infrastructure
- **Implemented Controls:** Secrets loaded exclusively from `.env` (excluded via `.gitignore`), strict server-side input & date validation, RAG anti-hallucination guardrails, and zero PII console logging.
- **Production Infrastructure & Compliance Roadmap:** TLS 1.3 in transit, AES-256 at rest (AWS KMS), JWT/RBAC multi-tenant isolation (`tenant_id` on all rows/vectors), PII redaction, retention policies, explicit AI & recording disclosure, and alignment with India's **DPDP Act 2023** and **TRAI** commercial communication regulations.

## 5. Proposed Scalable Architecture for 100,000 Users
> **MVP vs. Production Distinction:** The current Streamlit + SQLite prototype is a **single-node local MVP** and does **not** support 100,000 concurrent users.

Using Little's Law (`Concurrent Calls = Arrival Rate × Avg Duration`), 100,000 registered users (assuming 20% daily active callers, 3-minute average call duration, and 15% peak-hour load = 3,000 peak-hour calls) yields **~150 peak concurrent calls**, provisioned with 3× headroom (~500 concurrent calls):
- **Stateless Edge & Voice Workers:** Decouple UI (React/Next.js) and deploy containerized FastAPI + WebSocket voice gateways (AWS ECS/EKS in Mumbai `ap-south-1`) behind an HTTPS/WSS load balancer with horizontal autoscaling and Twilio/Exotel Media Streams.
- **Memory, Data & Async Queue:** Replace local state/SQLite with **Redis** (session memory & FAQ cache), **PostgreSQL + PgBouncer + pgvector** (multi-tenant CRM & vector search), and **Celery / AWS SQS** workers for post-call summaries and HubSpot/Zoho/Calendar sync.
- **Reliability & Monitoring:** Multi-provider LLM/STT/TTS fallbacks with timeouts, rate limiting, and Prometheus/Grafana or CloudWatch observability.

## 6. Current Limitations & Future Scope
- **Limitations:** Local SQLite single-writer scope; turn-based voice recording (`SpeechRecognition` & `gTTS` require internet); appointment requests are not synced to an external Google Calendar.
- **Future Scope:** Full-duplex WebSocket telephony with barge-in, PostgreSQL + `pgvector` + Redis migration, HubSpot/Zoho/Google Calendar OAuth adapters, and automated RAG faithfulness evaluation.
