# SalesPilot AI – Autonomous AI Sales Voice Agent

An end-to-end AI-powered autonomous sales assistant built for **Fortivex AI Private Limited** that interacts with customers via **voice and text**, answers company questions using **RAG**, qualifies leads, analyzes customer **sentiment**, captures **appointment requests**, generates structured **call summaries**, and displays live metrics in an **Admin CRM Dashboard**.

---

## Business Problem
Sales teams lose high-intent leads because of slow response times, limited working hours, inconsistent qualification scripts, and incomplete CRM notes. Managers also lack structured visibility into customer sentiment, budgets, timelines, and objections.

## Proposed Solution
**SalesPilot AI** provides an always-on conversational sales assistant that:
1. Engages prospects in **English or Hindi** using text or voice input.
2. Grounds product, pricing, and policy answers in `data/company_knowledge.txt` using local **FAISS + TF-IDF RAG retrieval** (refusing honestly if facts are unavailable).
3. Extracts and scores **7 lead qualification fields** (`name`, `email`, `phone`, `product_interest`, `budget`, `business_requirement`, `purchase_timeline`) and asks targeted follow-up questions for missing details.
4. Tracks customer **sentiment** (`Positive`, `Neutral`, `Negative`).
5. Validates and logs **appointment requests** (`Requested (Pending Confirmation)`) and **structured call summaries** in **SQLite**.
6. Runs seamlessly with **Groq API**, **Gemini API**, or a clearly labeled **Local Fallback Mode** when no API key is configured.

---

## Key Features
- **Voice & Text Conversation:** Streamlit chat interface with microphone recording (`st.audio_input`), `.wav` upload, `SpeechRecognition` STT, and `gTTS` spoken audio playback with graceful offline error handling.
- **Grounded RAG Engine:** Section-aware chunking + `faiss-cpu` / `scikit-learn` TF-IDF cosine retrieval over `data/company_knowledge.txt`.
- **Lead Qualification & Scoring:** Automatic extraction + deterministic 0–100 qualification scoring stored in SQLite.
- **Sentiment Analysis:** Per-turn and overall conversation sentiment (`Positive` / `Neutral` / `Negative`).
- **Appointment Requests:** Validated date/time/contact capture clearly labeled as a pending request (not an external calendar booking).
- **Structured Call Summaries:** Post-call summary capturing requirements, interest, budget, timeline, sentiment, qualification status, and recommended next action.
- **Admin CRM Dashboard:** Live KPI metrics, sentiment & funnel charts, leads CSV export, appointment requests table, summaries, and full conversation transcripts.
- **Multilingual Support:** Tested support for **English** and **Hindi** (text, STT, and TTS).

---

## Technology Stack
- **Frontend / UI:** Streamlit, Pandas, Matplotlib
- **AI / LLM:** Groq API (`llama-3.3-70b-versatile`), Google Gemini REST API (`gemini-2.0-flash`), Local Retrieval Fallback
- **RAG / Vector Search:** `scikit-learn` (TF-IDF) + `faiss-cpu`
- **Speech (STT & TTS):** `SpeechRecognition`, `gTTS`
- **Database:** SQLite (`database/salespilot.db`)
- **Optional REST API Layer:** FastAPI + Uvicorn (`backend/main.py`)
- **Testing:** `pytest`

---

## System Architecture
See the full architecture diagram in [architecture/architecture.png](architecture/architecture.png) and detailed technical report in [report/project_report.md](report/project_report.md).

**Local MVP Flow:**  
`Customer (Voice/Text)` → `Streamlit UI` → `STT / Text Input` → `AI Conversation Engine` → `RAG Knowledge Retrieval` ↔ `LLM / Local Fallback` → `Lead Qualification & Sentiment` → `SQLite CRM` & `Appointment Request` → `Call Summary` → `TTS (gTTS)` → `Customer` & `Admin Dashboard`

**Scalability Note (100,000 Users):**  
The current implementation is a local laptop MVP using Streamlit and SQLite. Production scaling toward 100,000 users (estimated ~150 peak concurrent calls under Little's Law, designed for 500 concurrent calls) is documented in `report/project_report.md` and `architecture/architecture.png` using stateless FastAPI WebSocket workers, Redis session state, PostgreSQL + pgvector, async Celery/SQS queues, and managed cloud telephony.

---

## Project Structure

```text
autonomous-ai-sales-voice-agent/
├── app.py                        # Main Streamlit UI (Sales Agent + Admin CRM Dashboard)
├── requirements.txt              # Python dependencies
├── README.md
├── .env.example                  # Sample environment variables (never commit .env)
├── .gitignore
├── data/
│   └── company_knowledge.txt     # Company products, pricing, FAQs, security info
├── database/
│   ├── __init__.py
│   └── database.py               # SQLite schema & CRUD operations
├── modules/
│   ├── __init__.py
│   ├── ai_engine.py              # Groq / Gemini / Local Fallback conversation engine
│   ├── rag_engine.py             # Chunking & FAISS/TF-IDF RAG retriever
│   ├── speech_engine.py          # SpeechRecognition STT + gTTS TTS (English & Hindi)
│   ├── lead_manager.py           # Lead extraction, scoring & SQLite persistence
│   ├── sentiment.py              # Positive/Neutral/Negative sentiment analyzer
│   ├── appointment_manager.py    # Validated appointment request capture
│   └── summarizer.py             # Structured call summary generator
├── backend/                      # Optional FastAPI REST API bridging to modules/
│   ├── config.py
│   ├── main.py
│   └── agents/sales_agent.py
├── architecture/
│   └── architecture.png          # MVP & 100k Production Architecture diagram
├── report/
│   └── project_report.md         # One-page technical & business evaluation report
└── tests/
    └── test_core.py              # Unit & end-to-end workflow tests
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Srujanaaddanki/SalesPilot-AI-Voice-Agent.git
cd SalesPilot-AI-Voice-Agent
```

### 2. Create and activate a virtual environment (recommended)
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env` and add your API key (optional — if omitted, the app runs in Local Fallback Mode):
```powershell
Copy-Item .env.example .env
```
Edit `.env`:
```env
LLM_PROVIDER=auto
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
SALESPILOT_DB_PATH=database/salespilot.db
```

---

## How to Run the Application

### Run the Streamlit App (Primary Demo)
```bash
streamlit run app.py
```
Then open the local URL printed by Streamlit (typically `http://localhost:8501`).

### Optional: Run the FastAPI REST Backend
```bash
uvicorn backend.main:app --reload
```

### Run Automated Tests
```bash
pytest -v
```

---

## End-to-End Demonstration Workflow
1. Open the Streamlit app (`streamlit run app.py`).
2. Enter or speak: `"I am interested in your AI automation services."`
3. SalesPilot AI retrieves relevant chunks from `data/company_knowledge.txt`, responds with service details, and asks qualification questions (requirements, budget, timeline, contact details).
4. Provide customer details (name, email, phone, budget, requirement, timeline).
5. The lead is scored, sentiment-tagged, and saved in SQLite.
6. Submit an appointment request via the **Request an Appointment** form (validated and saved as `Requested (Pending Confirmation)`).
7. Click **End Call & Generate Summary** to save a structured summary in SQLite.
8. Switch to **Admin CRM Dashboard** in the sidebar to inspect KPIs, charts, lead records, appointment requests, call summaries, and full conversation transcripts.

---

## Screenshots
*(No screenshots are included in this repository yet. Run `streamlit run app.py` locally to view the Sales Voice & Chat Agent interface and Admin CRM Dashboard.)*

---

## Current Limitations
- Uses local SQLite and single-process Streamlit state (suitable for single-user/demo workloads, not concurrent 100k production traffic).
- `SpeechRecognition` (`recognize_google`) and `gTTS` require an active internet connection; when offline, errors are caught and displayed gracefully while text chat and local RAG fallback continue to work.
- Voice input is turn-based (microphone recording or WAV upload) rather than continuous full-duplex phone streaming.
- Appointment requests are logged locally for human follow-up rather than synced to an external Google Calendar or CRM OAuth API.

## Future Improvements
- Real-time full-duplex telephony integration via Twilio / Exotel Media Streams with barge-in support.
- PostgreSQL + `pgvector` + Redis migration for multi-tenant horizontal scaling.
- Direct HubSpot, Zoho CRM, and Google Calendar API integrations.
- Automated RAG faithfulness evaluation and human-in-the-loop QA analytics.
