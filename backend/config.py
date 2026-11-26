"""Backend configuration supporting Groq, Gemini, and graceful Local Fallback without crashing."""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
SALESPILOT_DB_PATH = os.getenv("SALESPILOT_DB_PATH", "database/salespilot.db")
