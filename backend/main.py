"""Optional FastAPI REST backend exposing SalesPilot AI conversation, CRM, and appointment endpoints."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agents.sales_agent import SalesAgent
from modules.appointment_manager import AppointmentManager
from modules.summarizer import CallSummarizer


app = FastAPI(
    title="SalesPilot AI - Autonomous AI Sales Voice Agent API",
    description="REST API layer for SalesPilot AI sales conversation, RAG, lead qualification, and CRM",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

sales_agent = SalesAgent()
appointment_manager = AppointmentManager(sales_agent.engine.db)
summarizer = CallSummarizer(sales_agent.engine.db)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    language: str = "English"


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    ai_response: str
    provider_used: str
    sentiment: str
    lead_score: int
    qualification_status: str


class AppointmentPayload(BaseModel):
    session_id: str = "default"
    customer_name: str
    contact_details: str
    preferred_date: str
    preferred_time: str
    notes: Optional[str] = ""


@app.get("/")
def home():
    return {
        "message": "SalesPilot AI - Autonomous AI Sales Voice Agent API is running!",
        "status": "online",
        "active_provider": sales_agent.engine.active_provider,
    }


@app.get("/health")
def health():
    return {"status": "healthy", "provider": sales_agent.engine.active_provider}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    result = sales_agent.get_response(
        message=request.message,
        session_id=request.session_id,
        language=request.language,
    )
    lead = result.get("lead", {})
    return ChatResponse(
        session_id=request.session_id,
        user_message=request.message,
        ai_response=result["reply"],
        provider_used=result["provider_used"],
        sentiment=result["sentiment"]["label"],
        lead_score=int(lead.get("lead_score", 0)),
        qualification_status=str(lead.get("qualification_status", "New")),
    )


@app.post("/appointments")
def request_appointment(payload: AppointmentPayload):
    res = appointment_manager.submit_appointment_request(
        session_id=payload.session_id,
        customer_name=payload.customer_name,
        contact_details=payload.contact_details,
        preferred_date=payload.preferred_date,
        preferred_time=payload.preferred_time,
        notes=payload.notes or "",
    )
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@app.post("/summary/{session_id}")
def generate_summary(session_id: str):
    return summarizer.generate_and_save_summary(session_id, ai_engine=sales_agent.engine)


@app.get("/dashboard")
def dashboard():
    db = sales_agent.engine.db
    return {
        "metrics": db.get_dashboard_metrics(),
        "leads": db.get_all_leads(),
        "appointments": db.get_all_appointments(),
        "summaries": db.get_all_summaries(),
    }
