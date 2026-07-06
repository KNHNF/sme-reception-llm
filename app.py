"""
Streamlit frontend for the SME AI Voice Assistant.

EMBEDDED mode (default, no extra terminal needed):
    streamlit run app.py
    The pipeline runs inside this process. No uvicorn required.

API mode (if you prefer running FastAPI separately):
    Terminal 1:  uvicorn backend:app --port 5005
    Terminal 2:  streamlit run app.py
    The app detects the backend and calls it over HTTP.

MODEL SELECTOR (sidebar):
    Mock           -- rule-based, instant, no GPU/checkpoint needed. Good for UI demos.
    Phi-3 FT       -- QLoRA fine-tuned Phi-3 mini. Needs checkpoints/sme-phi3-qlora/.
    Llama 3.2 FT   -- QLoRA fine-tuned Llama 3.2 3B. Needs checkpoints/sme-llama3-qlora/.
    Phi-3 vanilla  -- base Phi-3 with no adapter. Shows the 0.4% baseline.
    Llama 3.2 van  -- base Llama 3.2 with no adapter. Shows the 0.0% baseline.
"""

import os
import sys
import uuid
import io

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pipeline loader (cached per mode string)
@st.cache_resource(show_spinner="Loading pipeline…")
def _load_pipeline(mode: str, family: str = "phi3"):
    from src.inference import Pipeline
    return Pipeline(mode=mode, model_family=family)

@st.cache_resource(show_spinner=False)
def _load_tts():
    try:
        from src.tts import TTS
        return TTS()
    except Exception:
        return None

_tts = _load_tts()

# API fallback (when embedded import fails)
import requests
API_URL = "http://localhost:5005"

def _api_turn(utterance, session_id):
    try:
        r = requests.post(f"{API_URL}/turn",
                          json={"utterance": utterance, "session_id": session_id},
                          timeout=15)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Backend offline. Run `uvicorn backend:app --port 5005` or use embedded mode."
    except Exception as e:
        return None, str(e)

def _embedded_turn(pipeline, utterance, session_id):
    result = pipeline.run(utterance, session_id=session_id)
    try:
        from src.metrics_logger import log_turn
        log_turn(
            session_id=session_id,
            utterance=utterance,
            action=result.get("action"),
            validated=result.get("validated", False),
            latency_ms=result.get("latency_ms", 0),
        )
    except Exception:
        pass
    action = result.get("action") or {}
    action_str = action.get("action", "") if isinstance(action, dict) else str(action)
    return {
        "spoken":      result.get("spoken", ""),
        "action":      action,
        "action_str":  action_str,
        "latency_ms":  result.get("latency_ms", 0),
        "entities":    result.get("entities", {}),
        "caller_name": result.get("caller_name"),
        "end_call":    result.get("end_call", False),
    }, None

import base64

def tts_audio(text: str) -> bytes | None:
    """Return raw WAV bytes."""
    if _tts and hasattr(_tts, "to_wav_bytes"):
        return _tts.to_wav_bytes(text)
    return None

def audio_html(wav_bytes: bytes) -> str:
    """Embed WAV as base64 data URI. Plays inline in all browsers, never downloads."""
    b64 = base64.b64encode(wav_bytes).decode()
    return (
        '<audio controls style="width:100%;margin-top:.3rem;height:32px">'
        f'<source src="data:audio/wav;base64,{b64}" type="audio/wav">'
        '</audio>'
    )

@st.cache_resource(show_spinner=False)
def _load_stt_model():
    """Load Faster-Whisper for browser mic transcription. Returns None if unavailable."""
    try:
        from faster_whisper import WhisperModel
        return WhisperModel("small", device="cpu", compute_type="int8")
    except Exception:
        return None

# Bias STT toward the words callers actually use in this domain, and enable VAD
# to trim silence (which stops the model repeating itself on trailing audio).
STT_DOMAIN_PROMPT = (
    "Appointment booking call for a clinic. Terms: book, cancel, reschedule, "
    "appointment, consultation, follow-up, availability, "
    "Monday Tuesday Wednesday Thursday Friday, morning, afternoon."
)

def transcribe_uploaded_audio(audio_bytes: bytes) -> str | None:
    """Transcribe WAV bytes from st.audio_input using Faster-Whisper."""
    model = _load_stt_model()
    if not model:
        return None
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        # Plain transcription: measured lowest WER on the real clips. VAD and a
        # domain initial_prompt both raised WER on this audio, so neither is used.
        segments, _ = model.transcribe(path, language="en")
        return " ".join(s.text for s in segments).strip()
    except Exception:
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

# Page config
st.set_page_config(
    page_title="SME AI Voice Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }

.header-bar {
    background: linear-gradient(135deg,#1A1A2E 0%,#028090 100%);
    padding: 1.1rem 1.5rem; border-radius: 12px; margin-bottom: 1rem;
}
.header-title { color:#fff; font-size:1.45rem; font-weight:700; margin:0; }
.header-sub   { color:rgba(255,255,255,.72); font-size:.8rem; margin:0; }

.pipeline-row { display:flex; align-items:center; gap:0; margin:.5rem 0 1rem; }
.phase-box    { background:#028090; color:#fff; padding:.25rem .65rem;
                font-size:.74rem; font-weight:600; border-radius:6px; }
.phase-star   { background:#02C39A; }
.phase-arrow  { color:#028090; font-size:.95rem; margin:0 .18rem; }

.user-bubble  { background:#EFF6FF; border-left:3px solid #028090;
                padding:.65rem 1rem; border-radius:0 10px 10px 10px; margin:.35rem 0; }
.asst-bubble  { background:#F0FBFC; border-left:3px solid #02C39A;
                padding:.65rem 1rem; border-radius:0 10px 10px 10px; margin:.35rem 0; }
.turn-meta    { font-size:.72rem; color:#64748B; font-weight:600; margin-bottom:.2rem; }

.badge { display:inline-block; padding:.12rem .55rem; border-radius:20px;
         font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.badge-book    { background:#DCFCE7; color:#166534; }
.badge-cancel  { background:#FEE2E2; color:#991B1B; }
.badge-check   { background:#DBEAFE; color:#1E40AF; }
.badge-end     { background:#F3F4F6; color:#374151; }
.badge-scope   { background:#FEF9C3; color:#713F12; }
.badge-clarify { background:#EDE9FE; color:#6B21A8; }
.badge-default { background:#F1F5F9; color:#475569; }

.entity-tag { display:inline-block; background:#EFF6FF; border:1px solid #BFDBFE;
              color:#1D4ED8; padding:.08rem .45rem; border-radius:4px;
              font-size:.7rem; font-weight:600; margin:.08rem; }

.metric-card { background:#fff; border:1px solid #E2E8F0; border-radius:10px;
               padding:.7rem .9rem; text-align:center; margin-bottom:.4rem; }
.metric-val  { font-size:1.5rem; font-weight:700; color:#028090; }
.metric-lbl  { font-size:.7rem; color:#64748B; }

.end-banner { background:linear-gradient(135deg,#1A1A2E,#028090); color:#fff;
              padding:.65rem 1rem; border-radius:8px; text-align:center;
              font-weight:600; margin-top:.5rem; }

.model-pill { display:inline-block; padding:.18rem .6rem; border-radius:20px;
              font-size:.7rem; font-weight:700; margin-left:.5rem; }
.pill-mock  { background:#E0E7FF; color:#3730A3; }
.pill-ft    { background:#DCFCE7; color:#166534; }
.pill-van   { background:#FEE2E2; color:#991B1B; }
</style>
""", unsafe_allow_html=True)

# Model selector options
# Modes must match Pipeline(mode=...) in src/inference.py.
# Fine-tuned options require checkpoints/ adapters on this machine (already present).
# No Kaggle needed. Kaggle was training only. Running uses the saved adapters locally.
# Vanilla modes are not in the Pipeline class (eval-only scripts); excluded here.
MODEL_OPTIONS = {
    "Mock (rule-based, instant)":          ("mock", "phi3",   "pill-mock", "Mock"),
    "Llama 3.2 CPU (Q4_K_M, real model)":  ("cpu",  "llama3", "pill-ft",   "Llama 3.2 CPU"),
    "Phi-3 mini CPU (real model)":         ("cpu",  "phi3",   "pill-ft",   "Phi-3 CPU"),
}

# Session state
_DEFAULT_MODEL = "Mock (rule-based, instant)"

def _init():
    defaults = {
        "messages":   [],
        "session_id": str(uuid.uuid4())[:8],
        "caller":     None,
        "turn_count": 0,
        "latencies":  [],
        "call_ended": False,
        "pending_in": None,
        "model_key":  _DEFAULT_MODEL,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Reset stale model_key if it no longer matches any option (e.g. after a rename)
    if st.session_state.model_key not in MODEL_OPTIONS:
        st.session_state.model_key = _DEFAULT_MODEL
_init()

# Booking toast, fired on the rerun after a slot was booked in the previous turn.
_just_booked = st.session_state.pop("just_booked", None)
if _just_booked:
    st.toast(f"Appointment booked: {_just_booked}", icon="✅")

def new_session():
    keys = ["messages","session_id","caller","turn_count","latencies","call_ended","pending_in"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    _init()

# Helpers
BADGES = {
    "book_appointment":   ("book",    "badge-book"),
    "cancel_appointment": ("cancel",  "badge-cancel"),
    "check_availability": ("avail.",  "badge-check"),
    "end_call":           ("end",     "badge-end"),
    "out_of_scope":       ("OOS",     "badge-scope"),
    "clarify":            ("clarify", "badge-clarify"),
}

def badge_html(action_str):
    if not action_str:
        return ""
    lbl, cls = BADGES.get(action_str, (action_str, "badge-default"))
    return f'<span class="badge {cls}">{lbl}</span>'

def avg_lat():
    lats = st.session_state.latencies
    return sum(lats) / len(lats) if lats else 0

# Calendar + metrics panels (demo evidence).
# Booking is performed inside the pipeline (src/inference.py calls book_slot),
# which writes data/calendar.json. We diff the booked slots around each turn to
# detect a new booking, so this works in mock, CPU and API mode alike.
import json
import datetime as _dt

_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calendar.json")

def _load_calendar() -> dict:
    try:
        with open(_CAL_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"slots": []}

def _booked_slots() -> set:
    data = _load_calendar()
    return {(s["date"], s["time"], s["service"])
            for s in data.get("slots", []) if not s.get("available", True)}

def _fmt_booking(date: str, time: str, service: str) -> str:
    try:
        d = _dt.datetime.strptime(date, "%Y-%m-%d")
        t = _dt.datetime.strptime(time, "%H:%M")
        day = f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"
        return f"{service.replace('_', ' ')} on {day} at {t.strftime('%I:%M %p').lstrip('0')}"
    except Exception:
        return f"{service} on {date} at {time}"

def render_calendar_panel():
    import pandas as pd
    data = _load_calendar()
    slots = data.get("slots", [])
    booked = [s for s in slots if not s.get("available", True)]
    st.caption(f"{data.get('business_name', 'The practice')} - "
               f"{data.get('opening_hours', 'Monday to Friday, 9am to 5pm')}")
    if not booked:
        st.info("No appointments booked yet. Book one in the conversation to see it appear here.")
        return
    rows = [{"Date": s["date"], "Time": s["time"], "Service": s["service"].replace("_", " ")}
            for s in sorted(booked, key=lambda x: (x["date"], x["time"]))]
    st.markdown(f"**Booked appointments ({len(rows)})**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_metrics_panel():
    import pandas as pd
    try:
        from src.metrics_logger import get_metrics
        m = get_metrics()
    except Exception as e:
        st.warning(f"Metrics unavailable: {e}")
        return
    if not m.get("total"):
        st.info("No turns logged yet. Have a conversation to populate metrics.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turns logged", m["total"])
    c2.metric("Schema-valid %", f"{m['accuracy']}%")
    c3.metric("P50 ms", f"{m['latency_p50']:.0f}")
    c4.metric("P95 ms", f"{m['latency_p95']:.0f}")
    st.caption("Schema-valid % is the share of turns that produced a schema-valid action. "
               "It is not action accuracy against ground truth, which needs a labelled test set.")
    pa = m.get("per_action", {})
    rows = [{"Action": k, "Count": v["count"], "Valid": v["correct"], "Valid %": v["accuracy"]}
            for k, v in pa.items() if v["count"]]
    if rows:
        st.markdown("**Per-action**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    recent = m.get("recent", [])
    if recent:
        st.markdown("**Recent turns**")
        rr = [{"Utterance": r["utterance"], "Action": r["action"],
               "Valid": "yes" if r["validated"] else "no",
               "ms": f"{(r['latency_ms'] or 0):.0f}"} for r in recent]
        st.dataframe(pd.DataFrame(rr), use_container_width=True, hide_index=True)

# Sidebar
with st.sidebar:
    st.markdown("### Model")
    prev_key = st.session_state.model_key
    chosen = st.radio(
        "Active model",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(st.session_state.model_key),
        label_visibility="collapsed",
    )
    if chosen != prev_key:
        st.session_state.model_key = chosen
        new_session()
        st.rerun()

    mode_str, family_str, pill_cls, pill_label = MODEL_OPTIONS[chosen]
    is_real_model = mode_str not in ("mock",)

    if is_real_model:
        st.warning(
            "Uses the local llama.cpp server. Start it first in a terminal:\n\n"
            "`python scripts/03_cpu_server.py --model llama3 --quant Q4_K_M`",
            icon="⚠️"
        )

    st.divider()
    st.markdown("**Session**")
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{st.session_state.turn_count}</div><div class="metric-lbl">Turns</div></div>', unsafe_allow_html=True)
    avg = avg_lat()
    col = "#02C39A" if avg < 1000 else "#028090" if avg < 4000 else "#EF4444"
    c2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{col}">{avg:.0f}</div><div class="metric-lbl">Avg ms</div></div>', unsafe_allow_html=True)
    if st.session_state.caller:
        st.info(f"Caller: {st.session_state.caller}")

    st.divider()
    st.markdown("**Examples**")
    EXAMPLES = [
        "I'd like to book a consultation next Monday 2pm",
        "Do you have slots Thursday for a follow-up?",
        "Cancel my Wednesday 10am appointment",
        "Book a general appointment next week",
        "How would I get a confirmation email?",
        "What are your opening hours?",
        "Goodbye, thanks for your help",
    ]
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:18]}"):
            st.session_state.pending_in = ex
            st.rerun()

    st.divider()
    if _tts and _tts.available:
        st.caption("Piper TTS ready. Audio plays after each response.")
    else:
        st.caption("Piper not found. Text only. See piper/ setup in README.")

    if not st.session_state.call_ended and st.session_state.turn_count > 0:
        if st.button("End call", use_container_width=True, type="primary"):
            st.session_state.call_ended = True
            st.rerun()

    if st.button("New call", use_container_width=True, type="secondary"):
        new_session(); st.rerun()

# Load pipeline for chosen mode
mode_str, family_str, pill_cls, pill_label = MODEL_OPTIONS[st.session_state.model_key]
pipeline = None
embed_err = None
try:
    pipeline = _load_pipeline(mode_str, family_str)
    EMBEDDED = True
except Exception as e:
    EMBEDDED = False
    embed_err = str(e)

def run_turn(utterance, session_id):
    if EMBEDDED and pipeline:
        return _embedded_turn(pipeline, utterance, session_id)
    return _api_turn(utterance, session_id)

# Header
st.markdown(f"""
<div class="header-bar">
  <p class="header-title">
    SME AI Voice Assistant
    <span class="model-pill {pill_cls}">{pill_label}</span>
  </p>
  <p class="header-sub">UWE Bristol · MSc Data Science · Group 6 · Offline · No cloud · GDPR-safe</p>
</div>""", unsafe_allow_html=True)

pipe_html = '<div class="pipeline-row">'
phases = [("STT","Faster-Whisper",False),("NER","spaCy",False),
          ("LLM","QLoRA fine-tuned",True),("TTS","Piper",False)]
for i,(label,tip,star) in enumerate(phases):
    cls = "phase-box phase-star" if star else "phase-box"
    pipe_html += f'<span class="{cls}" title="{tip}">{label}{"  ★" if star else ""}</span>'
    if i < 3: pipe_html += '<span class="phase-arrow"> -> </span>'
pipe_html += "</div>"
st.markdown(pipe_html, unsafe_allow_html=True)

# Main area: conversation on the left, calendar and metrics on the right
col_chat, col_side = st.columns([3, 2], gap="large")

with col_side:
    st.markdown("##### Calendar")
    render_calendar_panel()
    st.markdown("##### Live metrics")
    render_metrics_panel()

with col_chat:
    if embed_err:
        if "cannot import name" in embed_err and "transformers" in embed_err:
            st.error(
                f"Broken transformers install: `{embed_err}`\n\n"
                "Run this in a terminal, then restart Streamlit:\n"
                "```\npip install --upgrade transformers\n```"
            )
        elif "checkpoints" in embed_err or "No such file" in embed_err:
            st.error(
                f"Adapter weights not found: `{embed_err}`\n\n"
                "The fine-tuned adapters must be in `checkpoints/sme-phi3-qlora/` or "
                "`checkpoints/sme-llama3-qlora/`. Check the folder exists."
            )
        elif "gated" in embed_err.lower() or "401" in embed_err or "token" in embed_err.lower():
            st.error(
                f"HuggingFace authentication required: `{embed_err}`\n\n"
                "1. Accept the licence at huggingface.co/meta-llama/Llama-3.2-3B-Instruct\n"
                "2. Run `huggingface-cli login` in a terminal\n"
                "3. Re-select the model"
            )
        else:
            st.error(f"Could not load model: {embed_err}")

    _state = ("Call ended" if st.session_state.call_ended
              else "In call" if st.session_state.turn_count else "Ready")
    _state_color = ("#EF4444" if _state == "Call ended"
                    else "#02C39A" if _state == "In call" else "#94A3B8")
    st.markdown(
        f'<div style="font-size:.75rem;font-weight:700;color:{_state_color};'
        f'margin-bottom:.5rem">&#9679; {_state}</div>',
        unsafe_allow_html=True)

    # Voice input at the top of the conversation so it is visible without scrolling.
    if not st.session_state.call_ended and not (embed_err and not EMBEDDED):
        _stt_model = _load_stt_model()
        if _stt_model is not None:
            try:
                import hashlib as _hashlib
                _mic_key = f"mic_{st.session_state.get('turn_count', 0)}"
                audio_val = st.audio_input("Speak your message", key=_mic_key)
                if audio_val is not None:
                    _raw = audio_val.read()
                    _audio_hash = _hashlib.md5(_raw).hexdigest()
                    if _audio_hash != st.session_state.get("_last_audio_hash"):
                        st.session_state["_last_audio_hash"] = _audio_hash
                        with st.spinner("Transcribing..."):
                            _mt = transcribe_uploaded_audio(_raw)
                        if _mt:
                            st.session_state.pending_in = _mt
                            st.caption(f"Heard: {_mt}")
            except AttributeError:
                pass

    if not st.session_state.messages:
        st.markdown(
            '<div style="padding:2rem 0;color:#94A3B8">'
            'Speak above or type below to begin the call.</div>',
            unsafe_allow_html=True)

    # Newest exchange first, so the latest receptionist reply sits under the mic
    # and you do not scroll to follow the conversation.
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble"><div class="turn-meta">YOU</div>{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            action_str = msg.get("action_str", "")
            latency    = msg.get("latency_ms")
            entities   = msg.get("entities", {})
            lat_html   = (f'<span style="font-size:.7rem;color:#94A3B8;margin-left:.4rem">{latency:.0f} ms</span>'
                          if latency else "")
            ent_html   = ""
            if entities:
                tags = [f'<span class="entity-tag">{k}: {v}</span>' for k,v in entities.items() if v]
                if tags:
                    ent_html = '<div style="margin-top:.35rem">' + " ".join(tags) + "</div>"
            st.markdown(
                f'<div class="asst-bubble">'
                f'<div class="turn-meta">ASSISTANT {badge_html(action_str)}{lat_html}</div>'
                f'{msg["content"]}{ent_html}</div>',
                unsafe_allow_html=True)
            if msg.get("audio"):
                st.markdown(audio_html(msg["audio"]), unsafe_allow_html=True)

    if st.session_state.call_ended:
        st.markdown('<div class="end-banner">Call ended. Click New call to start again.</div>',
                    unsafe_allow_html=True)

# Input
if embed_err and not EMBEDDED:
    st.info("Fix the model error above, then re-select from the sidebar.")
elif not st.session_state.call_ended:
    pending = st.session_state.pending_in
    if pending:
        st.session_state.pending_in = None

    # Voice input is rendered at the top of the conversation column above and
    # feeds in via pending_in. The text box stays pinned at the bottom.
    user_input = st.chat_input("Type your message...") or pending

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.turn_count += 1

        booked_before = _booked_slots()
        with st.spinner("Processing..."):
            data, err = run_turn(user_input, st.session_state.session_id)
        _new_booked = _booked_slots() - booked_before
        if _new_booked:
            _bd, _bt, _bs = sorted(_new_booked)[0]
            st.session_state["just_booked"] = _fmt_booking(_bd, _bt, _bs)

        if err:
            st.session_state.messages.append({
                "role": "assistant", "content": f"Error: {err}", "action_str": ""
            })
        else:
            spoken     = data.get("spoken", "Sorry, I could not process that.")
            action_str = data.get("action_str") or ""
            if not action_str and isinstance(data.get("action"), dict):
                action_str = data["action"].get("action", "")
            latency    = data.get("latency_ms", 0)
            entities   = data.get("entities", {})
            caller     = data.get("caller_name")
            end_call   = data.get("end_call", False)

            if caller:
                st.session_state.caller = caller
            if latency:
                st.session_state.latencies.append(latency)

            audio = tts_audio(spoken)  # returns raw bytes or None

            st.session_state.messages.append({
                "role":       "assistant",
                "content":    spoken,
                "action_str": action_str,
                "latency_ms": latency,
                "entities":   entities,
                "audio":      audio,
            })
            if end_call:
                st.session_state.call_ended = True
        st.rerun()
else:
    st.info("Click New call in the sidebar to start again.")
