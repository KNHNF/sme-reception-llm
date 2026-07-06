"""
Inference Pipeline
Runs one full turn: utterance -> entity extraction -> LLM -> validated JSON action.

This is what the FastAPI backend calls on each customer utterance.
It does NOT include STT or TTS -- those are handled at the call layer.

Usage:
    python inference.py --mock                                    no GPU, mock outputs
    python inference.py --vanilla                                 Phi-3 mini, no adapter
    python inference.py --adapter checkpoints/sme-phi3-qlora     Phi-3 + QLoRA
    python inference.py --model llama3 --vanilla                  Llama 3.2 3B, no adapter
    python inference.py --model llama3 --adapter checkpoints/sme-llama3-qlora
    python inference.py --model ollama --ollama_model llama3.2    via local Ollama
"""

import argparse
import json
import re as _re_top
import sys
import time
from pathlib import Path
from typing import Optional

def _join_spelled_name(text: str) -> str:
    """Convert hyphen-spelled names to a proper word.

    'K-A-R-A-N'             -> 'Karan'
    'J-A-C-K R-E-A-C-H-E-R' -> 'Jack Reacher'  (two separate groups)
    'my name is K-A-R-A-N'  -> 'my name is Karan'

    Uses HYPHENS only as separators - spaces between groups mark word boundaries
    so 'J-A-C-K R-E-A-C-H-E-R' becomes two matches, not one big 'Jackreacher'.
    """
    def _join(m: _re_top.Match) -> str:
        letters = _re_top.findall(r"[A-Za-z]", m.group())
        return "".join(letters).title()

    return _re_top.sub(r"\b[A-Za-z](?:-[A-Za-z]){1,}\b", _join, text)


_SPELLED_RE = _re_top.compile(r"(?<![A-Za-z])[A-Za-z](?:[\s.\-]+[A-Za-z]){2,}(?![A-Za-z])")

def _extract_spelled_name(text: str):
    """Return a spelled-out name assembled into a word, or None.

    Callers spell their name to correct a mishearing ('it's Koran, K-A-R-A-N',
    'no, K A R A N', 'K. A. R. A. N.'), so when a run of at least three single
    letters is present it is the authoritative name and beats any word the model
    misheard. Handles hyphen, space and full-stop separators. Returns the longest
    such run titlecased, or None if there is no spelled run.
    """
    best = None
    for m in _SPELLED_RE.finditer(text):
        run = m.group()
        letters = _re_top.findall(r"[A-Za-z]", run)
        if len(letters) < 3:
            continue
        if "-" in run:
            # Hyphens separate letters, spaces separate words:
            # 'J-A-C-K R-E-A-C-H-E-R' -> 'Jack Reacher'
            words = ["".join(_re_top.findall(r"[A-Za-z]", g)).title()
                     for g in run.split() if _re_top.search(r"[A-Za-z]", g)]
            cand = " ".join(words)
        else:
            # Pure space or full-stop spelling: one word. 'K A R A N' -> 'Karan'
            cand = "".join(letters).title()
        if best is None or len(cand) > len(best):
            best = cand
    return best

# Support both: `python src/inference.py` (script) and `from src.inference import Pipeline` (module)
try:
    from src.entity_extractor import extract, to_prompt_context
    from src.sme_action_schema import ActionOutput, render_confirmation
    from src.calendar_store import get_next_slot, book_slot, describe_slot, _ordinal
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from entity_extractor import extract, to_prompt_context
    from sme_action_schema import ActionOutput, render_confirmation
    from calendar_store import get_next_slot, book_slot, describe_slot, _ordinal


def _save_message(session_id: str, caller_name, text: str) -> None:
    """Append a caller message to a local file. Offline, no cloud, no Supabase."""
    import json, time
    path = Path(__file__).parent.parent / "data" / "messages.jsonl"
    row = {"ts": time.time(), "session_id": session_id,
           "caller": caller_name, "message": text}
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


SYSTEM_PROMPT = (
    "Appointment assistant. Output one JSON object only. "
    "Actions: book_appointment, check_availability, cancel_appointment, clarify, out_of_scope. "
    "Services: general, consultation, follow_up. "
    "Dates: YYYY-MM-DD. Times: HH:MM. "
    "If fields missing: {\"action\": \"clarify\", \"missing_fields\": [...]}."
)

MODEL_IDS = {
    "phi3":   "microsoft/Phi-3-mini-4k-instruct",
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
}

def build_prompt(utterance: str, entities: dict, partial_context: Optional[dict] = None,
                 model_family: str = "phi3") -> str:
    """
    Build the chat-format prompt for the chosen model family.
    Supported: "phi3" or "llama3".

    If spaCy extracted entities, they are prepended as a hint so the LLM
    does not need to resolve dates or times from natural language itself.

    If this is a clarification turn (caller answering a follow-up), the
    partial action context from the session is included too.
    """
    entity_hint = to_prompt_context(entities)

    user_content = utterance
    if entity_hint:
        user_content = f"{entity_hint}\n{utterance}"

    if partial_context and partial_context.get("partial_entities"):
        known = json.dumps(partial_context["partial_entities"])
        missing = partial_context.get("missing_fields", [])
        user_content = (
            f"[Already known: {known}] "
            f"[Still missing: {missing}]\n"
            + user_content
        )

    if model_family in ("llama3", "llama1b"):
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    if model_family in ("qwen0.5b", "qwen1.5b", "smol360"):
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # Default: Phi-3 format
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_content}<|end|>\n"
        f"<|assistant|>\n"
    )

def parse_llm_output(text: Optional[str]) -> Optional[dict]:
    """Extract JSON from LLM output, tolerating minor formatting noise."""
    if not text:
        return None
    text = text.strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

def validate_action(raw: Optional[dict]):
    """
    Validate against the Pydantic schema.
    Returns a validated model instance, or None on failure.
    """
    if raw is None:
        return None
    from pydantic import TypeAdapter, ValidationError
    adapter = TypeAdapter(ActionOutput)
    try:
        return adapter.validate_python(raw)
    except ValidationError:
        return None

class Pipeline:
    """
    Wraps model loading and inference.

    Modes:
      mock      -- hard-coded outputs, no GPU needed (for testing the pipeline structure)
      vanilla   -- base model with no adapter (baseline condition)
      finetuned -- base model + QLoRA adapter (primary condition)
      ollama    -- calls local Ollama API (use for Llama 3 vanilla if you have it installed)

    Model families: "phi3" or "llama3"
    """

    def __init__(self, mode: str = "mock",
                 model_family: str = "phi3",
                 adapter_path: Optional[str] = None,
                 ollama_model: str = "llama3.2",
                 ollama_url: str = "http://localhost:11434",
                 cpu_url: str = "http://127.0.0.1:8080"):

        self.mode         = mode
        self.model_family = model_family
        self.model        = None
        self.tokenizer    = None
        self.ollama_model = ollama_model
        self.ollama_url   = ollama_url
        self.cpu_url      = cpu_url
        self.cpu_timeout  = 60

        model_id = MODEL_IDS.get(model_family, MODEL_IDS["phi3"])

        if mode == "mock":
            print("Pipeline running in MOCK mode -- no GPU needed.")
            return

        if mode == "ollama":
            print(f"Pipeline using Ollama: {ollama_model} at {ollama_url}")
            return

        if mode == "cpu":
            import urllib.request as _u
            try:
                with _u.urlopen(f"{cpu_url}/health", timeout=3) as r:
                    reachable = (r.status == 200)
            except Exception:
                reachable = False
            if reachable:
                print(f"Pipeline using llama.cpp CPU server at {cpu_url}")
            else:
                print(f"WARNING: no llama.cpp server reachable at {cpu_url}.")
                print("Start it first: python scripts/03_cpu_server.py --model phi3")
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_token = __import__("os").environ.get("HF_TOKEN")

        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        ampere_plus = any(x in gpu_name for x in ["A100", "A10", "A30", "A40", "RTX 30", "RTX 40", "H100"])
        attn_impl = "flash_attention_2" if ampere_plus else "eager"

        print(f"Loading model: {model_id} (attn={attn_impl})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        self.model.eval()

        if mode == "finetuned" and adapter_path:
            from peft import PeftModel
            print(f"Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Model ready.")

    def run(self, utterance: str, session_id: str = "default",
            partial_context: Optional[dict] = None) -> dict:
        """
        Full pipeline turn.

        Returns:
            {
                "raw_output":      str    -- raw LLM text
                "action":          dict   -- parsed JSON (or None)
                "validated":       bool
                "spoken":          str    -- TTS-ready confirmation string
                "latency_ms":      float
                "entities":        dict   -- spaCy extraction
                "end_call":        bool   -- True if the call should terminate
                "caller_name":     str|None
            }
        """
        # Use src-prefixed imports to match how backend.py loads these modules,
        # preventing dual-identity bugs where two copies of _sessions exist.
        try:
            import src.session_manager as sm
            from src.profanity import contains_profanity, de_escalate, is_terminal_strike
        except ImportError:
            import session_manager as sm  # type: ignore[no-redef]
            from profanity import contains_profanity, de_escalate, is_terminal_strike  # type: ignore[no-redef]

        t0 = time.perf_counter()
        session = sm.get_or_create(session_id)

        # 0a. Empty / silent input guard
        if not utterance or not utterance.strip():
            spoken = "I'm sorry, I didn't catch that. Could you say that again?"
            return self._quick("", spoken, False, session, t0)

        # Cap input length - very long utterances can cause latency spikes or
        # overflow the model's context window.
        utterance = utterance[:500].strip()

        # 0b. Profanity gate - runs before anything else
        if contains_profanity(utterance):
            session.profanity_strikes += 1
            session.touch()
            spoken   = de_escalate(session.profanity_strikes)
            end_call = is_terminal_strike(session.profanity_strikes)
            return self._quick(utterance, spoken, end_call, session, t0)

        # 0b-2. Voicemail capture: if we asked for a message, this utterance IS it.
        if getattr(session, "awaiting_message", False):
            _save_message(session_id, session.caller_name, utterance)
            session.awaiting_message = False
            session.touch()
            name_part = f", {session.caller_name}" if session.caller_name else ""
            spoken = (f"Thank you{name_part}. I've taken your message and a member of the "
                      "team will call you back. Is there anything else I can help you with?")
            return self._quick(utterance, spoken, False, session, t0)

        # 0c. Human transfer request
        # If caller asks to speak to a real person, acknowledge and route out.
        # In a deployed system this would trigger a SIP transfer; in the demo
        # it closes the session and sets end_call so the loop terminates cleanly.
        _u = utterance.lower()
        _TRANSFER = {
            "speak to someone", "speak to a person", "real person", "actual person",
            "speak to staff", "member of staff", "speak to the team",
            "transfer me", "talk to someone", "talk to a person",
            "human", "operator", "receptionist please",
        }
        if any(t in _u for t in _TRANSFER):
            sm.close(session_id)
            spoken = (
                "Of course - I'll transfer you to a member of our team now. "
                "Please hold for a moment."
            )
            return self._quick(utterance, spoken, True, session, t0)

        # 0c-b. Reschedule / modify an existing booking.
        # The system does not change existing appointments, so instead of failing
        # with "I could not process that", offer the supported alternatives.
        # Guarded by pending_suggestion so it never hijacks live slot negotiation
        # (where "another day" means "offer me a different slot").
        _RESCHEDULE = {
            "reschedule", "move my appointment", "move the appointment",
            "move my booking", "change my appointment", "change the appointment",
            "change my booking", "rearrange", "make it later", "make it earlier",
            "a few days later", "do it later", "push it back", "push it to",
        }
        if not session.pending_suggestion and any(r in _u for r in _RESCHEDULE):
            spoken = (
                "I'm not able to change an existing booking on this automated line. "
                "I can cancel that appointment and book a new time for you, or take a "
                "message for a member of the team to call you back. "
                "Would you like me to cancel and rebook?"
            )
            return self._quick(utterance, spoken, False, session, t0)

        # 0c-c. Take a message for the human team (offline, stored locally).
        _MESSAGE_INTENT = {
            "leave a message", "take a message", "call me back", "have someone call",
            "leave a note", "pass a message", "get someone to call", "call me later",
        }
        if any(m in _u for m in _MESSAGE_INTENT):
            session.awaiting_message = True
            session.touch()
            spoken = ("Of course. Please tell me your message and the best number to reach "
                      "you on, and I'll pass it to the team.")
            return self._quick(utterance, spoken, False, session, t0)

        # 0d. Distress / emergency detection
        # Do NOT escalate these through the booking flow - acknowledge
        # urgency immediately.
        _DISTRESS = {
            "emergency", "it's urgent", "its urgent", "urgent appointment",
            "not well", "feeling unwell", "very unwell", "in pain", "severe pain",
            "chest pain", "can't breathe", "cannot breathe", "ambulance",
            "collapsed", "unconscious", "help me please",
        }
        if any(d in _u for d in _DISTRESS):
            spoken = (
                "I can hear this may be urgent. For a medical emergency, "
                "please hang up and call 999 immediately. "
                "If you need the next available appointment, "
                "say 'I need an urgent appointment' and I'll find you a slot right now."
            )
            return self._quick(utterance, spoken, False, session, t0)

        # 1. Name collection - intercept the first two turns
        if session.awaiting_name:
            # Whatever the caller said IS their name. Prefer spaCy PERSON if found.
            entities = extract(utterance)
            raw = utterance.strip()
            # Strip common preambles: "it's X", "its X", "I'm X", "my name is X", "this is X"
            import re as _re
            match = _re.search(
                r"(?:it'?s|i'?m|this is|my name is|name'?s|call me)\s+([A-Za-z][A-Za-z\s\-']+)",
                raw, _re.IGNORECASE
            )
            spelled = _extract_spelled_name(raw)
            if spelled:
                name = spelled
            elif match:
                name = _join_spelled_name(match.group(1).strip()).title()
            elif entities.get("person"):
                name = _join_spelled_name(entities["person"])
            else:
                # No name found via regex or spaCy.
                # Only re-ask if the utterance is clearly a booking request, not a name.
                # Positive name check: a name is short and contains no action verbs.
                # More reliable than a keyword blocklist.
                _action_words = {
                    "want", "like", "need", "book", "make", "cancel", "check",
                    "schedule", "available", "appointment", "consultation", "general",
                    "follow", "reschedule", "speak", "help", "calling", "hello",
                }
                _words = utterance.lower().split()
                _is_intent = (
                    len(_words) > 4
                    or any(w in _action_words for w in _words)
                )
                if _is_intent:
                    if session.name_reask_count >= 2:
                        # Caller has ignored the name prompt twice - proceed without name,
                        # store a placeholder so the rest of the pipeline works normally.
                        session.caller_name = "there"
                        session.awaiting_name = False
                        session.awaiting_name_confirm = False
                        session.touch()
                        # Fall through to normal turn processing below
                    else:
                        session.name_reask_count += 1
                        session.touch()
                        spoken = (
                            "I need your name before I can help with that. "
                            "Could you tell me your first and last name please?"
                        )
                        return self._quick(utterance, spoken, False, session, t0)
                name = _join_spelled_name(raw).title()
            # Guard: awaiting_name may have been cleared by the name_reask bypass above.
            # If so, skip name confirmation and fall through to normal turn processing.
            if not session.awaiting_name:
                pass  # bypass already set caller_name and cleared flags - continue below
            else:
                session.caller_name = name
                session.awaiting_name = False
                session.awaiting_name_confirm = True
                session.touch()
                spoken = (
                    f"Thank you. Just to confirm - did I catch that correctly as {name}?"
                )
                return self._quick(utterance, spoken, False, session, t0)

        if getattr(session, "awaiting_name_confirm", False):
            # Caller is confirming or correcting their name
            u = utterance.lower().strip()
            yes_words = {"yes", "yeah", "yep", "correct", "right", "that's right",
                         "yep that's right", "thats right", "that's correct", "sure"}
            no_words  = {"no", "nope", "wrong", "not quite", "actually", "it's",
                         "its", "i'm", "my name is"}
            session.awaiting_name_confirm = False
            if any(w in u for w in yes_words):
                session.touch()
                spoken = (
                    f"Great, thanks {session.caller_name}. How can I help you today? "
                    f"I can help with booking, cancellations, or checking availability."
                )
            else:
                # Check if they ignored the confirmation and sent booking intent again
                # Only re-ask if this looks like booking intent, NOT a name correction.
                # "No, it's Jack Richer" is a correction even though it's 4 words -
                # so rely on verb presence only, not word count.
                _reask_verbs = {"want", "like", "need", "book", "make", "cancel",
                                "check", "schedule", "appointment", "consultation",
                                "general", "follow", "available", "reschedule"}
                _uwords = u.split()
                _is_booking_intent = any(w in _reask_verbs for w in _uwords)
                if _is_booking_intent:
                    # Discard bad name, re-ask cleanly
                    session.caller_name = None
                    session.awaiting_name = True
                    session.awaiting_name_confirm = False
                    session.name_reask_count += 1
                    session.touch()
                    spoken = (
                        "I still need your name before I can help with that. "
                        "Could you tell me your first and last name please?"
                    )
                    return self._quick(utterance, spoken, False, session, t0)
                # They're correcting - re-capture from this utterance
                import re as _re2
                match2 = _re2.search(
                    r"(?:it'?s|i'?m|this is|my name is|name'?s|actually|call me)\s+"
                    r"([A-Za-z][A-Za-z\s\-']+)",
                    utterance, _re2.IGNORECASE
                )
                spelled2 = _extract_spelled_name(utterance)
                if spelled2:
                    session.caller_name = spelled2
                elif match2:
                    session.caller_name = _join_spelled_name(match2.group(1).strip()).title()
                session.touch()
                spoken = (
                    f"Apologies! Thanks {session.caller_name}. How can I help you today? "
                    f"I can help with booking, cancellations, or checking availability."
                )
            return self._quick(utterance, spoken, False, session, t0)

        if session.turn_count == 0 and session.caller_name is None:
            # First turn - greet and capture name
            session.awaiting_name = True
            session.touch()
            spoken = (
                "Thank you for calling City Medical Practice. "
                "Please note that this call may be recorded for quality and training purposes. "
                "Could I take your name please?"
            )
            return self._quick(utterance, spoken, False, session, t0)

        # 2. Pending slot confirmation - caller said yes/no to a suggestion
        if session.pending_suggestion:
            u = utterance.lower()
            yes_words = {"yes", "yeah", "yep", "sure", "ok", "okay", "please",
                         "that works", "that's fine", "sounds good", "sounds great",
                         "perfect", "great", "go on", "go ahead", "that'll do",
                         "that will do", "lovely", "brilliant", "works for me",
                         "do that", "book it", "book that", "let's do it",
                         "yes please", "sounds perfect"}
            no_words  = {"no", "nope", "nah", "different", "another", "else",
                         "not that", "other", "later", "earlier", "next",
                         "not now", "leave it", "forget it", "some other time"}
            # Words that mean "a different day entirely" (not just a later time slot)
            diff_day_words = {"date", "day", "week", "monday", "tuesday", "wednesday",
                              "thursday", "friday"}

            if any(w in u for w in yes_words):
                slot = session.pending_suggestion
                book_slot(slot["date"], slot["time"], slot["service"])
                session.clear()
                name_part = f", {session.caller_name}" if session.caller_name else ""
                from datetime import datetime as _dt
                d = _dt.strptime(slot["date"], "%Y-%m-%d")
                t = _dt.strptime(slot["time"], "%H:%M")
                day_str = f"{d.strftime('%A')} {_ordinal(d.day)} {d.strftime('%B')}"
                spoken = (
                    f"Brilliant{name_part}. I've booked your {slot['service'].replace('_', ' ')} "
                    f"for {day_str} at {t.strftime('%I:%M %p').lstrip('0')}. "
                    f"You'll receive a confirmation shortly. Is there anything else I can help you with?"
                )
                return self._quick(utterance, spoken, False, session, t0)

            # Check for a specific date request BEFORE the general no_words check.
            # "the 26th", "on Friday", "if possible on the 24th" - NER + regex fallback.
            # spaCy en_core_web_sm sometimes misses "the 26th of June" so we add a
            # direct ordinal-date regex as a safety net.
            _ents = extract(utterance)
            req_date = _ents.get("date_resolved")
            if not req_date:
                import re as _re_ord
                from datetime import date as _date_cls
                _MONTHS = {
                    "january": 1, "february": 2, "march": 3, "april": 4,
                    "may": 5, "june": 6, "july": 7, "august": 8,
                    "september": 9, "october": 10, "november": 11, "december": 12,
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
                    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
                }
                _om = _re_ord.search(
                    r"\b(\d{1,2})(?:st|nd|rd|th)(?:\s+of\s+|\s+)?(\w+)?",
                    utterance, _re_ord.IGNORECASE,
                )
                if _om:
                    _day = int(_om.group(1))
                    _mon_str = (_om.group(2) or "").lower().strip()
                    _today = _date_cls.today()
                    _month = _MONTHS.get(_mon_str, _today.month)
                    try:
                        from datetime import datetime as _dtord
                        _cand = _dtord(_today.year, _month, _day).date()
                        if _cand < _today:
                            _cand = _dtord(_today.year + 1, _month, _day).date()
                        req_date = _cand.isoformat()
                    except ValueError:
                        pass
            if req_date and req_date != session.pending_suggestion.get("date"):
                session.suggestion_index = 0
                slot = get_next_slot(
                    service=session.pending_suggestion.get("service"),
                    preferred_date=req_date,
                    skip=0,
                )
                if slot:
                    session.pending_suggestion = slot
                    session.touch()
                    if slot["date"] != req_date:
                        from datetime import datetime as _dtreq
                        req_d = _dtreq.strptime(req_date, "%Y-%m-%d")
                        req_day = f"{req_d.strftime('%A')} {_ordinal(req_d.day)} {req_d.strftime('%B')}"
                        spoken = (
                            f"I'm afraid I don't have anything available on {req_day}. "
                            f"The nearest I have is {describe_slot(slot)}. Would that work for you?"
                        )
                    else:
                        spoken = f"How about {describe_slot(slot)}? Would that suit you?"
                    return self._quick(utterance, spoken, False, session, t0)
                else:
                    session.pending_suggestion = None
                    spoken = (
                        "I'm afraid I don't have any availability around that date. "
                        "Could I take your number and have someone call you back?"
                    )
                    return self._quick(utterance, spoken, False, session, t0)

            if any(w in u for w in no_words):
                session.suggestion_index += 1
                # "later date" / "another day" = jump to NEXT calendar day entirely.
                # "later" / "earlier" alone = same day different time -> keep preferred_date.
                _wants_diff_day = any(w in u for w in diff_day_words)
                if _wants_diff_day:
                    # Filter all future slots to those strictly after the current suggested date
                    try:
                        from src.calendar_store import find_slots as _find_all
                    except ImportError:
                        from calendar_store import find_slots as _find_all
                    _current_date = session.pending_suggestion.get("date")
                    _future = [s for s in _find_all(service=session.pending_suggestion.get("service"))
                               if s["date"] > _current_date]
                    slot = _future[0] if _future else None
                else:
                    slot = get_next_slot(
                        service=session.pending_suggestion.get("service"),
                        preferred_date=session.pending_suggestion.get("date"),
                        skip=session.suggestion_index,
                    )
                if slot:
                    session.pending_suggestion = slot
                    session.touch()
                    spoken = f"How about {describe_slot(slot)}? Would that suit you?"
                    return self._quick(utterance, spoken, False, session, t0)
                else:
                    session.pending_suggestion = None
                    spoken = (
                        "I'm afraid I don't have any more available slots at the moment. "
                        "Could I take your number and have someone call you back?"
                    )
                    return self._quick(utterance, spoken, False, session, t0)

        # 3. End-call detection (model-independent). A caller saying goodbye should end the
        # call, not be sent to the model and misread as out_of_scope. Kept to clear terminal
        # phrases so it never hijacks a normal turn; declines to a proposed slot are handled
        # above via pending_suggestion, so "no" alone does not reach here mid-negotiation.
        _u = utterance.lower().strip().rstrip(".!?")
        _bye_exact = {"bye", "no thanks", "no thank you", "no that's all", "no thats all",
                      "that's it", "thats it"}
        _bye_phrases = ("goodbye", "good bye", "that's all", "thats all", "that is all",
                        "that will be all", "that'll be all", "nothing else", "nothing more",
                        "all done", "we're done", "were done", "i'm done", "im done",
                        "no that's all", "no thanks that")
        if _u in _bye_exact or any(ph in _u for ph in _bye_phrases):
            sm.close(session_id)
            return {
                "raw_output":  "",
                "action":      {"action": "end_call"},
                "validated":   True,
                "spoken":      "Thank you for calling. Have a good day. Goodbye.",
                "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
                "entities":    {},
                "end_call":    True,
                "caller_name": session.caller_name,
            }

        # 4. Normal LLM inference
        entities = extract(utterance)
        prompt   = build_prompt(utterance, entities, partial_context, self.model_family)

        if self.mode == "mock":
            raw_text = self._mock_output(utterance, entities, session)
        elif self.mode == "ollama":
            raw_text = self._ollama_generate(prompt)
        elif self.mode == "cpu":
            raw_text = self._cpu_generate(prompt)
        else:
            raw_text = self._hf_generate(prompt)

        t1 = time.perf_counter()

        if not raw_text:
            # Model returned nothing - fall back to confusion handler
            session.confusion_count += 1
            session.touch()
            spoken = "I'm sorry, I didn't quite catch that. Could you say that again?"
            return self._quick(utterance, spoken, False, session, t0)

        parsed    = parse_llm_output(raw_text)
        validated = validate_action(parsed)

        # 4. Post-processing: enrich check_availability with a real suggestion
        if validated and validated.action.value == "check_availability":
            preferred_date = getattr(validated, "date", None)
            service        = getattr(validated, "service", None)
            svc_str        = service.value if service else None
            slot = get_next_slot(service=svc_str, preferred_date=preferred_date,
                                 skip=session.suggestion_index)
            if slot:
                session.pending_suggestion = slot
                session.touch()
                name_part = f", {session.caller_name}" if session.caller_name else ""
                spoken = (
                    f"Of course{name_part}. The next available slot is "
                    f"{describe_slot(slot)}. Does that work for you?"
                )
            else:
                spoken = (
                    "I'm afraid there are no available slots matching that request right now. "
                    "Would you like me to check for a different date or service?"
                )
        elif validated and validated.action.value == "out_of_scope":
            session.confusion_count += 1
            c = session.confusion_count
            if c == 1:
                spoken = (
                    "I can help with booking, cancellations, or checking availability. "
                    "For example, you could say: 'I'd like to book a general appointment' "
                    "or 'Do you have any slots on Tuesday?'"
                )
            elif c == 2:
                spoken = (
                    "I'm sorry, I'm having trouble understanding. "
                    "Could you try saying the date like 'Tuesday the 8th', "
                    "the time like '10am' or '2:30pm', "
                    "and the type - general, consultation, or follow-up?"
                )
            elif c == 3:
                spoken = (
                    "I'm still not quite catching that. "
                    "Let me try once more - please say something like: "
                    "'Book a general appointment on Monday at 10am.'"
                )
            else:
                # 4th failure - end call gracefully
                spoken = (
                    "I'm afraid I'm not able to assist further on this call. "
                    "Please call back during office hours and a member of our team will be happy to help. "
                    "Goodbye."
                )
                sm.close(session_id)
                return {
                    "raw_output":  raw_text,
                    "action":      parsed,
                    "validated":   True,
                    "spoken":      spoken,
                    "latency_ms":  round((t1 - t0) * 1000, 2),
                    "entities":    entities,
                    "end_call":    True,
                    "caller_name": session.caller_name,
                }
        elif validated:
            session.confusion_count = 0   # reset on any successful action
            if validated.action.value == "book_appointment":
                # Guard against booking invented details. Only write to the calendar
                # when the caller actually gave a date AND a time (spaCy-resolved).
                # Otherwise propose a real slot and confirm first, reusing the
                # pending_suggestion yes/no flow. Stops the model booking a
                # hallucinated slot from a vague "book an appointment".
                has_date = bool(entities.get("date_resolved"))
                has_time = bool(entities.get("time_resolved"))
                if has_date and has_time:
                    book_slot(validated.date, validated.time, validated.service.value)
                    spoken = render_confirmation(validated)
                else:
                    svc = validated.service.value if getattr(validated, "service", None) else None
                    slot = get_next_slot(service=svc,
                                         preferred_date=(validated.date if has_date else None))
                    if slot:
                        session.pending_suggestion = slot
                        session.touch()
                        name_part = f", {session.caller_name}" if session.caller_name else ""
                        spoken = (f"Of course{name_part}. The next available slot is "
                                  f"{describe_slot(slot)}. Does that work for you?")
                    else:
                        spoken = ("Of course. Which day and time would you like, and is it a "
                                  "general appointment, a consultation, or a follow-up?")
            elif validated.action.value == "cancel_appointment":
                # An automated line cannot verify the caller's identity or that a booking is
                # theirs, so we never auto-cancel. Log the request for the reception team and
                # tell the caller honestly that a human will confirm it.
                _save_message(session_id, session.caller_name,
                              f"[cancellation request] {utterance}")
                name_part = f", {session.caller_name}" if session.caller_name else ""
                spoken = (
                    f"Thank you{name_part}. I've noted your cancellation request and passed it to "
                    f"our reception team, who will confirm it and be in touch. "
                    f"Is there anything else I can help you with?"
                )
            else:
                spoken = render_confirmation(validated)
        else:
            spoken = "I could not process that. Could you repeat?"

        end_call = (
            validated is not None
            and validated.action.value == "end_call"
        )
        if end_call:
            sm.close(session_id)

        return {
            "raw_output":   raw_text,
            "action":       parsed,
            "validated":    validated is not None,
            "spoken":       spoken,
            "latency_ms":   round((t1 - t0) * 1000, 2),
            "entities":     entities,
            "end_call":     end_call,
            "caller_name":  session.caller_name,
        }

    def _quick(self, utterance: str, spoken: str, end_call: bool,
               session, t0: float) -> dict:
        """Return a fast pipeline result bypassing LLM (name capture, profanity, etc.)."""
        return {
            "raw_output":  "",
            "action":      None,
            "validated":   True,
            "spoken":      spoken,
            "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
            "entities":    {},
            "end_call":    end_call,
            "caller_name": session.caller_name,
        }

    def _hf_generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # EOS token differs between model families
        if self.model_family == "llama3":
            eos_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        else:
            eos_id = self.tokenizer.convert_tokens_to_ids("<|end|>")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_id,
            )
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _ollama_generate(self, prompt: str) -> str:
        """
        Call the local Ollama API.
        Make sure Ollama is running: ollama serve
        And the model is pulled: ollama pull llama3.2
        """
        import json as _json
        import urllib.request

        payload = _json.dumps({
            "model":  self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 40},
        }).encode()

        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
            return data.get("response", "").strip()
        except Exception as e:
            return f'{{"action": "out_of_scope"}}  # ollama error: {e}'

    def _cpu_generate(self, prompt: str) -> str:
        """
        Call the local llama.cpp server (scripts/03_cpu_server.py) /completion endpoint.
        Sends the raw templated prompt so the input matches the fine-tuning format exactly.
        """
        import json as _json
        import urllib.request

        if self.model_family in ("llama3", "llama1b"):
            stop = ["<|eot_id|>"]
        elif self.model_family in ("qwen0.5b", "qwen1.5b", "smol360"):
            stop = ["<|im_end|>"]
        else:
            stop = ["<|end|>"]
        payload = _json.dumps({
            "prompt":      prompt,
            "n_predict":   40,
            "temperature": 0,
            "stop":        stop,
            "stream":      False,
            "cache_prompt": True,
        }).encode()

        req = urllib.request.Request(
            f"{self.cpu_url}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cpu_timeout) as resp:
                data = _json.loads(resp.read())
            return (data.get("content") or "").strip()
        except Exception as e:
            return f'{{"action": "out_of_scope"}}  # cpu server error: {e}'

    def _mock_output(self, utterance: str, entities: dict, session=None) -> str:
        """
        Returns plausible JSON without a real model.
        Used for testing the pipeline structure before training is done.
        """
        u = utterance.lower()
        date = entities.get("date_resolved") or "2026-06-25"
        time = entities.get("time_resolved") or "10:00"
        svc  = entities.get("service") or "general"

        # Bare negatives after booking confirmation ("anything else?" -> "no")
        if u.strip() in ("no", "nope", "nah", "nothing", "no thanks", "that's all",
                         "that is all", "all good", "all done", "no that's all"):
            return json.dumps({"action": "end_call"})

        if any(w in u for w in ["bye", "goodbye", "thank you", "thanks", "that's all",
                                   "thats all", "no thanks", "nothing else", "have a good"]):
            return json.dumps({"action": "end_call"})

        # Confused re-asks after a clarify - treat as continued booking intent
        if u.strip() in ("what time", "what date", "what service", "what appointment",
                          "what?", "sorry?", "pardon?", "excuse me", "i don't understand"):
            return json.dumps({"action": "clarify", "missing_fields": ["time"]})

        if any(w in u for w in ["cancel", "cancellation"]):
            return json.dumps({"action": "cancel_appointment", "date": date, "time": time})

        if any(w in u for w in ["available", "availability", "free", "have any", "any slots",
                                  "any appointments", "what times", "when can", "do you have"]):
            return json.dumps({"action": "check_availability", "date": date, "service": svc})

        if any(w in u for w in ["hours", "open", "close", "closing", "opening", "time do you",
                                  "when are you", "directions", "address", "email", "confirmation",
                                  "how do i", "how would", "will i receive", "get a"]):
            return json.dumps({"action": "out_of_scope"})

        # If specific date AND time are mentioned, go straight to book_appointment.
        # Otherwise use check_availability so the slot-suggestion -> yes/no flow runs
        # and book_slot() is actually called when the caller confirms.
        has_date = bool(entities.get("date_resolved"))
        has_time = bool(entities.get("time_resolved"))
        if has_date and has_time:
            return json.dumps({"action": "book_appointment", "date": date, "time": time, "service": svc})
        return json.dumps({"action": "check_availability", "date": date, "service": svc})
