"""
FastAPI Backend
Mock calendar and booking API for the SME voice assistant demo.

No real database or Google Calendar -- uses in-memory dicts.
This is appropriate for the IGP portfolio demo and June 30th mock viva.

Run:
    pip install fastapi uvicorn
    uvicorn backend:app --reload --port 5005

Endpoints:
    POST /turn              Full pipeline turn (utterance -> spoken response)
    GET  /availability      Check available slots for a date + service
    POST /book              Book an appointment
    POST /cancel            Cancel an appointment
    GET  /appointments      List all booked appointments (debug)
    DELETE /appointments    Clear all bookings (debug / reset)
    GET  /metrics           Conversation metrics (accuracy, latency, per-action)
    POST /metrics/clear     Clear all logged metrics
    GET  /metrics-dashboard Serve the live HTML dashboard
"""

import uuid
from datetime import date, time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="SME Voice Assistant API", version="0.1.0")
DASHBOARD_PATH = Path(__file__).parent / "docs" / "dashboard.html"

from twilio_handler import router as twilio_router
app.include_router(twilio_router)

def _get_logger():
    try:
        from src.metrics_logger import log_turn, get_metrics, clear_metrics
    except ImportError:
        from metrics_logger import log_turn, get_metrics, clear_metrics  # type: ignore
    return log_turn, get_metrics, clear_metrics

# In-memory store: appointment_id -> appointment dict
BOOKINGS: dict[str, dict] = {}

# Business hours: slots available per day (24h format)
AVAILABLE_SLOTS = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
                   "14:00", "14:30", "15:00", "15:30", "16:00", "16:30"]

SERVICE_DURATION = {
    "general":      30,
    "consultation": 60,
    "follow_up":    15,
}

class TurnRequest(BaseModel):
    session_id: str
    utterance: str
    consent: bool = False
    source: str = "text"

class BookRequest(BaseModel):
    date: str          # ISO 8601
    time: str          # HH:MM
    service: str       # general | consultation | follow_up
    caller_name: Optional[str] = None

class CancelRequest(BaseModel):
    appointment_id: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None

class AvailabilityRequest(BaseModel):
    date: str
    service: Optional[str] = "general"

# Lazy-load the pipeline so the server starts fast even without a GPU.
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.inference import Pipeline
        _pipeline = Pipeline(mode="mock")
    return _pipeline

@app.post("/turn")
def pipeline_turn(req: TurnRequest):
    """
    Main endpoint. Takes a caller utterance, runs the full pipeline,
    returns the action JSON and the spoken confirmation string.
    """
    from src.session_manager import get_context, update

    ctx      = get_context(req.session_id)
    pipeline = get_pipeline()
    result   = pipeline.run(req.utterance, session_id=req.session_id, partial_context=ctx)

    if result["action"]:
        merged = update(req.session_id, result["action"], result["entities"])
        result["action"] = merged

        # If the action is a booking, execute it automatically.
        if merged.get("action") == "book_appointment":
            try:
                booking = _do_book(merged)
                result["booking_id"] = booking["appointment_id"]
            except HTTPException as e:
                result["booking_error"] = e.detail

    # Log metrics
    try:
        log_turn, _, _ = _get_logger()
        # Without consent, never persist the transcript text. Aggregate metrics
        # (action, latency) still count, but the caller's words are not stored.
        log_turn(
            session_id=req.session_id,
            utterance=req.utterance if req.consent else "[not recorded]",
            action=result.get("action"),
            validated=result.get("validated", False),
            latency_ms=result.get("latency_ms", 0),
        )
    except Exception:
        pass  # Never let metrics break the pipeline

    # Record the call locally (transcript + outcome) when the caller has consented.
    if req.consent:
        try:
            from src.call_log import log_call
            booking = None
            if result.get("booking_id"):
                booking = {"appointment_id": result["booking_id"],
                           "details": result.get("action")}
            log_call(
                session_id=req.session_id,
                transcript=req.utterance,
                action=result.get("action"),
                spoken_reply=result.get("spoken", ""),
                booking=booking,
                consent=req.consent,
                source=req.source,
            )
        except Exception:
            pass  # Never let call logging break the pipeline

    return result


@app.get("/calls")
def list_call_records(limit: int = 50):
    """Local call records (transcript and outcome). Personal data, never leaves the machine."""
    from src.call_log import list_calls
    return {"calls": list_calls(limit)}

@app.get("/metrics")
def metrics_endpoint():
    """Return conversation metrics: accuracy, latency P50/P95, per-action counts."""
    try:
        _, get_metrics, _ = _get_logger()
        return get_metrics()
    except Exception as e:
        return {"error": str(e)}

@app.post("/metrics/clear")
def metrics_clear():
    """Clear all logged turn data."""
    try:
        _, _, clear_metrics = _get_logger()
        clear_metrics()
        return {"message": "Metrics cleared"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/metrics-dashboard")
def metrics_dashboard():
    """Serve the live HTML metrics dashboard."""
    if DASHBOARD_PATH.exists():
        return FileResponse(str(DASHBOARD_PATH), media_type="text/html")
    return JSONResponse({"error": "dashboard.html not found in docs/"}, status_code=404)

@app.get("/availability")
def check_availability(date_str: str, service: str = "general"):
    """Return available slots for a given date, accounting for existing bookings."""
    booked_times = {
        b["time"] for b in BOOKINGS.values()
        if b["date"] == date_str
    }
    free = [s for s in AVAILABLE_SLOTS if s not in booked_times]
    return {
        "date":     date_str,
        "service":  service,
        "duration": SERVICE_DURATION.get(service, 30),
        "slots":    free,
        "booked":   len(AVAILABLE_SLOTS) - len(free),
    }

@app.post("/book")
def book_appointment(req: BookRequest):
    return _do_book(req.model_dump())

def _do_book(data: dict) -> dict:
    date_str = data.get("date")
    time_str = data.get("time")
    service  = data.get("service", "general")

    if not date_str or not time_str:
        raise HTTPException(status_code=400, detail="date and time are required")

    if service not in SERVICE_DURATION:
        raise HTTPException(status_code=400, detail=f"Unknown service: {service}")

    # Check for double-booking
    for b in BOOKINGS.values():
        if b["date"] == date_str and b["time"] == time_str:
            raise HTTPException(status_code=409, detail="Slot already booked")

    appt_id = f"APT-{uuid.uuid4().hex[:6].upper()}"
    BOOKINGS[appt_id] = {
        "appointment_id": appt_id,
        "date":           date_str,
        "time":           time_str,
        "service":        service,
        "caller_name":    data.get("caller_name"),
        "duration_min":   SERVICE_DURATION[service],
    }
    return BOOKINGS[appt_id]

@app.post("/cancel")
def cancel_appointment(req: CancelRequest):
    if req.appointment_id:
        if req.appointment_id not in BOOKINGS:
            raise HTTPException(status_code=404, detail="Appointment not found")
        del BOOKINGS[req.appointment_id]
        return {"cancelled": req.appointment_id}

    if req.date and req.time:
        to_cancel = [
            aid for aid, b in BOOKINGS.items()
            if b["date"] == req.date and b["time"] == req.time
        ]
        if not to_cancel:
            raise HTTPException(status_code=404, detail="No appointment found for that date/time")
        for aid in to_cancel:
            del BOOKINGS[aid]
        return {"cancelled": to_cancel}

    raise HTTPException(status_code=400, detail="Provide appointment_id or date+time")

@app.get("/appointments")
def list_appointments():
    return {"total": len(BOOKINGS), "appointments": list(BOOKINGS.values())}

@app.delete("/appointments")
def clear_appointments():
    BOOKINGS.clear()
    return {"message": "All appointments cleared"}

@app.get("/health")
def health():
    return {"status": "ok", "bookings": len(BOOKINGS)}
