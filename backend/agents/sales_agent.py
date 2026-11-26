"""SalesAgent wrapper delegating to the unified modules.ai_engine.SalesAIEngine."""

from modules.ai_engine import SalesAIEngine


class SalesAgent:
    """FastAPI-compatible wrapper around SalesAIEngine."""

    def __init__(self) -> None:
        self.engine = SalesAIEngine()

    def get_response(self, message: str, session_id: str, language: str = "English") -> dict:
        return self.engine.process_turn(
            session_id=session_id,
            user_message=message,
            language=language,
        )
