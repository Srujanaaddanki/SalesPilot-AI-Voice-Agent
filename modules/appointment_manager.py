"""Appointment request validation and SQLite persistence (clearly distinct from external calendar sync)."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from database.database import DatabaseManager


class AppointmentManager:
    """Validates and stores customer appointment requests in SQLite."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    @staticmethod
    def validate_request(
        customer_name: str,
        contact_details: str,
        preferred_date: str,
        preferred_time: str,
    ) -> Tuple[bool, str, Dict[str, str]]:
        """Validate required fields, contact format, date (>= today), and time format."""
        name = (customer_name or "").strip()
        contact = (contact_details or "").strip()
        date_str = (preferred_date or "").strip()
        time_str = (preferred_time or "").strip()

        if len(name) < 2:
            return False, "Customer name is required (at least 2 characters).", {}

        has_email = bool(re.search(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", contact))
        digits = re.sub(r"\D", "", contact)
        has_phone = 7 <= len(digits) <= 15
        if not (has_email or has_phone):
            return False, "Valid contact details (email address or phone number) are required.", {}

        parsed_date: Optional[date] = None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                parsed_date = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue

        if parsed_date is None:
            return False, "Preferred date must be a valid date (e.g., YYYY-MM-DD).", {}

        if parsed_date < date.today():
            return False, "Preferred date cannot be in the past.", {}

        parsed_time = None
        for tfmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
            try:
                parsed_time = datetime.strptime(time_str.upper(), tfmt).time()
                break
            except ValueError:
                continue

        if parsed_time is None:
            return False, "Preferred time must be valid (e.g., 15:00 or 03:00 PM).", {}

        normalized = {
            "customer_name": name,
            "contact_details": contact,
            "preferred_date": parsed_date.isoformat(),
            "preferred_time": parsed_time.strftime("%H:%M"),
        }
        return True, "Validated", normalized

    def submit_appointment_request(
        self,
        session_id: str,
        customer_name: str,
        contact_details: str,
        preferred_date: str,
        preferred_time: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Validate and save an appointment request to SQLite."""
        ok, message, normalized = self.validate_request(
            customer_name, contact_details, preferred_date, preferred_time
        )
        if not ok:
            return {"success": False, "error": message}

        record = self.db.create_appointment(
            session_id=session_id,
            customer_name=normalized["customer_name"],
            contact_details=normalized["contact_details"],
            preferred_date=normalized["preferred_date"],
            preferred_time=normalized["preferred_time"],
            notes=notes or "Captured via SalesPilot AI (Pending sales team confirmation)",
        )
        return {
            "success": True,
            "message": (
                f"Appointment REQUEST saved for {record['customer_name']} on "
                f"{record['preferred_date']} at {record['preferred_time']} "
                f"(Status: {record['status']} — not an external calendar booking)."
            ),
            "appointment": record,
        }

    def list_appointment_requests(self) -> List[Dict[str, Any]]:
        return self.db.get_all_appointments()
