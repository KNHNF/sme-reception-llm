"""
Twilio voice call webhook handler.

Adds two routes to the FastAPI app:
  POST /twilio/voice   - entry point when a call comes in
  POST /twilio/gather  - receives Twilio's speech transcription, runs pipeline, responds

No Twilio Python SDK needed. We respond with raw TwiML (XML).
Twilio does the STT (speech-to-text) via its Gather verb, so Faster-Whisper
is not used on this path. The LLM pipeline and session management are identical
to the /turn endpoint.

Setup:
  1. Start the API server: uvicorn backend:app --port 5005
  2. Expose it publicly: cloudflared tunnel --url http://localhost:5005
  3. In Twilio console, set your phone number's voice webhook to:
       https://<your-cloudflare-url>/twilio/voice   (HTTP POST)
  4. Call the number. That's it.

To use the real fine-tuned model instead of mock, change mode="mock"
to mode="phi3" or mode="llama3" in _get_pipeline() below, and make
sure the llama.cpp server is running on port 8080 first.
"""

import xml.sax.saxutils as saxutils
from fastapi import APIRouter, Form, Response
from typing import Optional

router = APIRouter(prefix="/twilio", tags=["Twilio"])

# Pipeline instance (lazy-loaded, separate from backend.py's instance)
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from src.inference import Pipeline
        except ImportError:
            from inference import Pipeline  # type: ignore
        _pipeline = Pipeline(mode="mock")  # change to "phi3" or "llama3" for real model
    return _pipeline


def _xml(text: str) -> str:
    """Escape text for safe embedding in TwiML."""
    return saxutils.escape(str(text))


def _gather_twiml(say_text: str) -> str:
    """TwiML that speaks a message then listens for speech input."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" action="/twilio/gather" timeout="5" speechTimeout="auto" language="en-GB">
    <Say voice="alice">{_xml(say_text)}</Say>
  </Gather>
  <Redirect>/twilio/voice</Redirect>
</Response>"""


@router.post("/voice")
async def twilio_voice():
    """
    Called by Twilio when an inbound call starts.
    Greets the caller and starts listening.
    """
    return Response(
        content=_gather_twiml("Thank you for calling. Could I take your name please?"),
        media_type="application/xml",
    )


@router.post("/gather")
async def twilio_gather(
    SpeechResult: Optional[str] = Form(None),
    CallSid: str = Form(...),
):
    """
    Called by Twilio after it transcribes the caller's speech.
    SpeechResult: the transcribed text (Twilio STT).
    CallSid: unique call ID, used as the session ID so state persists across turns.
    """
    if not SpeechResult or not SpeechResult.strip():
        return Response(
            content=_gather_twiml("Sorry, I didn't catch that. Please go ahead."),
            media_type="application/xml",
        )

    try:
        from src.session_manager import get_context, update
    except ImportError:
        from session_manager import get_context, update  # type: ignore

    ctx = get_context(CallSid)
    pipeline = _get_pipeline()
    result = pipeline.run(SpeechResult, session_id=CallSid, partial_context=ctx)

    if result.get("action"):
        merged = update(CallSid, result["action"], result.get("entities", {}))
        result["action"] = merged

    spoken = result.get("spoken_text", "I'm sorry, I couldn't process that. Please try again.")
    action = result.get("action") or {}

    # Hang up cleanly if the pipeline signals end of call
    if isinstance(action, dict) and action.get("action") == "end_call":
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">{_xml(spoken)}</Say>
  <Hangup/>
</Response>"""
        return Response(content=twiml, media_type="application/xml")

    return Response(
        content=_gather_twiml(spoken),
        media_type="application/xml",
    )
