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


# Lookbehind excludes both letters AND apostrophes. Without the apostrophe,
# "it's K-A-R-A-N" matched starting at the "s" in "it's" (apostrophe isn't a
# letter, so the old lookbehind let "s" start a run), swallowing it into the
# result as a bogus leading "S" - "it's K-A-R-A-N" became "S Karan" instead
# of "Karan". Same trap for any contraction ("that's", "let's", ...)
# immediately followed by a spelled name.
_SPELLED_RE = _re_top.compile(r"(?<![A-Za-z'])[A-Za-z](?:[\s.\-]+[A-Za-z]){2,}(?![A-Za-z])")

def _extract_spelled_name(text: str):
    """Return a spelled-out name assembled into a word (or words), or None.

    Callers spell their name to correct a mishearing ('it's Koran, K-A-R-A-N',
    'no, K A R A N', 'K. A. R. A. N.'), so when a run of at least three single
    letters is present it is the authoritative name and beats any word the model
    misheard. Handles hyphen, space and full-stop separators.

    Callers often spell first AND last name as two separate runs in one
    utterance ('J-O-H-N, and V-I-C-K'), the comma/'and' between them breaks
    the regex into two matches rather than one. Every matched run found is
    concatenated in the order spoken ('John Vick'), not just the longest one -
    keeping only the longest silently dropped whichever name part was shorter.
    Returns None if there is no spelled run at all.
    """
    parts = []
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
        parts.append(cand)
    if not parts:
        return None
    return " ".join(parts)

# Support both: `python src/inference.py` (script) and `from src.inference import Pipeline` (module)
try:
    from src.entity_extractor import extract, to_prompt_context
    from src.sme_action_schema import ActionOutput, render_confirmation
    from src.calendar_store import get_next_slot, book_slot, describe_slot, _ordinal, find_slots
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from entity_extractor import extract, to_prompt_context
    from sme_action_schema import ActionOutput, render_confirmation
    from calendar_store import get_next_slot, book_slot, describe_slot, _ordinal, find_slots


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


# Shortened from the original ~30-word version (dropped "quality and", "Please
# note that"). Previously ended with "After the beep, please tell me your
# name" - dropped that per Karan's call: it reads like a voicemail machine,
# not a live receptionist, and the beep was never the fix for the
# missing-first-word problem anyway (that was a stream-open race condition in
# STT.listen_vad, fixed separately). Listening still starts immediately after
# this line finishes, same as a human receptionist pausing for a reply; no
# audible cue is promised or required. This is a wording simplification, not
# legal advice: if the call-recording disclosure needs specific regulatory
# phrasing, restore the fuller sentence.
GREETING = (
    "Thank you for calling City Medical Practice. "
    "This call may be recorded for training purposes. "
    "Could I take your name, please?"
)

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

# Caller signals they're done. Module-level (was previously defined inline,
# twice in effect) so the pending_suggestion branch can check it too - "no
# thank you" while a slot is on offer used to be swallowed by no_words'
# bare "no" and treated as "reject this slot, offer another" instead of
# "I'm done", so the call never ended even though the caller said thanks
# and hung up in spirit.
_BYE_EXACT = {"bye", "that's it", "thats it"}
_BYE_PHRASES = (
    "goodbye", "good bye", "that's all", "thats all", "that is all",
    "that will be all", "that'll be all", "nothing else", "nothing more",
    "all done", "we're done", "were done", "i'm done", "im done",
    "no that's all", "no thanks", "no thank you", "thank you", "thanks",
)

# --- Third-party booking ("book an appointment for my son") -----------------
# Relation-word list only, not freeform names. Freeform "for <Name>" collides
# with too many other "for" usages in a booking sentence (dates, reasons,
# services) to detect reliably without false positives, so this only fires on
# an explicit family/relation word, which is unambiguous.
_ON_BEHALF_RE = _re_top.compile(
    r"\bfor (?:my |our )(son|daughter|wife|husband|partner|mother|father|"
    r"mum|mom|dad|child|kid|sister|brother|grandson|granddaughter|"
    r"grandmother|grandfather)\b",
    _re_top.IGNORECASE,
)


def _word_match(text: str, phrases) -> bool:
    """True if any phrase appears in `text` as a whole word/phrase, not as
    a raw substring.

    Found via a live test run, not hypothetical: yes_words' bare "ok" is a
    substring of "book" ("b-OO-K"), so "Can I book a consultation and a
    follow-up" matched yes_words and silently confirmed whatever slot was
    already pending - ignoring the caller's actual request and booking the
    wrong thing. no_words has the identical exposure via bare "no" (inside
    "know", "unknown", "annotate", etc). Word-boundaried regex per phrase
    fixes both while still matching multi-word phrases as a contiguous unit.
    """
    return any(_re_top.search(rf"\b{_re_top.escape(p)}\b", text) for p in phrases)


def _behalf_note(session) -> str:
    """' This booking is for your son.' or '' if booking for the caller."""
    rel = getattr(session, "on_behalf_of", None)
    return f" This booking is for your {rel}." if rel else ""


# --- Doctor/practitioner-specific requests -----------------------------------
# The service model has no practitioner field at all, so this used to fall
# through to a generic out_of_scope reply from the LLM - honest but not
# helpful, since it never acknowledges the preference was heard. Detected up
# front, logged for the human team, and answered plainly instead: no
# guarantee of a specific doctor on this line, but noted.
_DOCTOR_RE = _re_top.compile(
    r"\b(?:with|see|for)\s+(?:dr\.?|doctor)\s+([A-Za-z]+)"
    r"|\b(same doctor|usual doctor|usual gp)\b",
    _re_top.IGNORECASE,
)

# --- Multi-service requests ("a consultation and a follow-up") --------------
_SERVICE_KW_MAP = {
    "consultation": ("consultation", "consult"),
    "follow_up":    ("follow-up", "follow up", "followup"),
    "general":      ("general appointment", "general"),
}
_SERVICE_LABELS = {
    "general": "a general appointment", "consultation": "a consultation",
    "follow_up": "a follow-up",
}


def _all_services_mentioned(text: str) -> list:
    """Every service keyword-matched in the text, in the order checked.

    entity_extractor.extract() only ever returns ONE service (first keyword
    match, by design - a single action needs a single service). This is a
    separate scan used only to detect when a caller asks for two services in
    the same breath ("a consultation and a follow-up"), so the second one can
    be queued instead of being silently discarded when extract() picks one.
    """
    low = text.lower()
    return [svc for svc, kws in _SERVICE_KW_MAP.items() if any(kw in low for kw in kws)]


def _queue_second_service(utterance: str, primary_svc: Optional[str], session) -> str:
    """Queue a second mentioned service and return a short spoken addendum.

    "" if only one service was mentioned, or one is already queued (so a
    later, unrelated mention of a second service name later in the call
    doesn't clobber an earlier queue before it's been offered).
    """
    if session.queued_service:
        return ""
    mentioned = _all_services_mentioned(utterance)
    if len(mentioned) < 2:
        return ""
    other = next((s for s in mentioned if s != primary_svc), None)
    if not other:
        return ""
    session.queued_service = other
    return f" I'll come back to {_SERVICE_LABELS.get(other, other)} for you right after this."


# --- Name correction after it's already been confirmed ----------------------
# The awaiting_name_confirm branch handles corrections right after the name
# is first given, but nothing re-opens name capture once that's resolved -
# if a caller realises mid-call "actually my name's Karan, not Karen", there
# was no path back. Deliberately requires an explicit correction cue
# ("actually", "correction", "you have my name wrong") rather than matching
# any sentence containing "my name" - a caller referencing their name for an
# unrelated reason mid-conversation shouldn't silently overwrite it.
_NAME_CORRECTION_RE = _re_top.compile(
    r"my name'?s?\s+actually\s+([A-Za-z][A-Za-z\s\-']+)"
    r"|actually,?\s+my name'?s?\s+([A-Za-z][A-Za-z\s\-']+)"
    r"|my name is actually\s+([A-Za-z][A-Za-z\s\-']+)"
    r"|correction,?\s+(?:it'?s|my name'?s?)\s+([A-Za-z][A-Za-z\s\-']+)"
    r"|you (?:have|got) my name wrong,?\s+it'?s\s+([A-Za-z][A-Za-z\s\-']+)",
    _re_top.IGNORECASE,
)

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

def _safe_service(model_service_value: Optional[str], entities: dict) -> Optional[str]:
    """Resolve which service to actually search/book for.

    Prefers the caller's own words (spaCy keyword match on the literal
    utterance) over the model's guess. Never accepts "follow_up" from the
    model alone with no keyword evidence - a follow-up implies a prior
    visit, which a brand-new caller cannot have, so an unprompted follow_up
    guess for a vague "what's free?" is a model quirk, not a real request.
    Falls back to "general" in that case rather than leaving it unset (unset
    means "any service", which can surface a follow_up slot anyway just
    because it happens to be chronologically first).
    """
    keyword_service = entities.get("service")
    if keyword_service:
        return keyword_service
    if model_service_value == "follow_up":
        return "general"
    return model_service_value


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
            # "reception" alone catches natural phrasings the exact-match
            # list above missed ("put me through to reception", "can I speak
            # to reception") - found via test_pipeline_state_machine.py,
            # which used exactly this phrase and it fell through to the LLM
            # instead of transferring.
            "put me through", "to reception", "speak to reception",
        }
        if any(t in _u for t in _TRANSFER):
            sm.close(session_id)
            spoken = (
                "Of course. I'll transfer you to a member of our team now. "
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
                # Bare filler/yes/no answers are not names either ("yes", "no" said
                # to "Could I take your name please?" must not become caller_name).
                _BARE_NONNAME = {
                    "yes", "yeah", "yep", "no", "nope", "nah", "sure", "ok", "okay",
                    "correct", "right", "wrong", "not", "quite", "actually",
                    "hi", "hey", "hello", "hiya", "howdy", "sorry",
                }
                # Strip trailing punctuation before comparing - "hello?" split()
                # into ["hello?"] never matches the bare word "hello" in either
                # set above, letting a caller's greeting slip through as their
                # "name" (it's still spoken back with the "?" attached too,
                # doubling up against the confirm prompt's own "?").
                _PUNCT = ".,!?;:'\""
                raw_clean = raw.strip(_PUNCT)
                _words = [w.strip(_PUNCT) for w in utterance.lower().split()]
                _is_intent = (
                    len(_words) > 4
                    or any(w in _action_words for w in _words)
                    or raw_clean.lower() in _BARE_NONNAME
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
                        if session.name_reask_count >= 2:
                            # Free-form re-asks keep failing - usually STT
                            # mishearing the name, not the caller's fault -
                            # switch to spelling instead of asking the same
                            # open question a third time.
                            spoken = (
                                "Let's try that differently. Could you spell "
                                "your name for me, letter by letter?"
                            )
                        else:
                            spoken = (
                                "I need your name before I can help with that. "
                                "Could you tell me your name please?"
                            )
                        return self._quick(utterance, spoken, False, session, t0)
                name = _join_spelled_name(raw_clean).title()
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
                    f"Thank you. Just to confirm, did I catch that correctly as {name}?"
                )
                return self._quick(utterance, spoken, False, session, t0)

        if getattr(session, "awaiting_name_confirm", False):
            # Caller is confirming or correcting their name
            u = utterance.lower().strip()
            yes_words = {"yes", "yeah", "yep", "correct", "right", "that's right",
                         "yep that's right", "thats right", "that's correct", "sure"}
            session.awaiting_name_confirm = False
            if _word_match(u, yes_words):
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
                    if session.name_reask_count >= 2:
                        spoken = (
                            "Let's try that differently. Could you spell your "
                            "name for me, letter by letter?"
                        )
                    else:
                        spoken = (
                            "I still need your name before I can help with that. "
                            "Could you tell me your name please?"
                        )
                    return self._quick(utterance, spoken, False, session, t0)
                _BAD_NAME_WORDS = {"no", "nope", "not", "wrong", "nothing", "none",
                                   "script", "correct", "right", "sure", "actually",
                                   "quite", "yes", "yeah"}

                # Partial correction: "my last name is Vich" / "surname is Vich" /
                # "first name is Karan" only fixes that one part, keeping the rest
                # of the already-captured name, rather than discarding everything
                # and re-asking from scratch.
                import re as _re_part
                last_match = _re_part.search(
                    r"(?:last name|surname|family name)(?:'s|\s+is|\s+was)?\s+"
                    r"([A-Za-z][A-Za-z\-']*)",
                    utterance, _re_part.IGNORECASE)
                first_match = _re_part.search(
                    r"first name(?:'s|\s+is|\s+was)?\s+([A-Za-z][A-Za-z\-']*)",
                    utterance, _re_part.IGNORECASE)
                if last_match or first_match:
                    existing = (session.caller_name or "").split()
                    ok = True
                    if last_match:
                        new_last = _join_spelled_name(last_match.group(1)).title()
                        if new_last.lower() in _BAD_NAME_WORDS:
                            ok = False
                        else:
                            existing = (existing[:-1] if len(existing) > 1 else []) + [new_last]
                    if ok and first_match:
                        new_first = _join_spelled_name(first_match.group(1)).title()
                        if new_first.lower() in _BAD_NAME_WORDS:
                            ok = False
                        else:
                            existing = [new_first] + (existing[1:] if len(existing) > 1 else [])
                    if ok and existing:
                        session.caller_name = " ".join(existing)
                        session.awaiting_name_confirm = True
                        session.touch()
                        spoken = (
                            f"Thank you. Just to confirm, did I catch that correctly "
                            f"as {session.caller_name}?"
                        )
                        return self._quick(utterance, spoken, False, session, t0)

                # They're correcting the whole name - re-capture from this utterance.
                # Negative lookahead stops "it's not correct" / "it's not script"
                # (a rejection, not a name) from being parsed as a name at all.
                import re as _re2
                match2 = _re2.search(
                    r"(?:it'?s|i'?m|this is|my name is|name'?s|actually|call me)\s+"
                    r"(?!not\b|n't\b|no\b)([A-Za-z][A-Za-z\s\-']+)",
                    utterance, _re2.IGNORECASE
                )
                spelled2 = _extract_spelled_name(utterance)
                new_name = None
                if spelled2:
                    new_name = spelled2
                elif match2:
                    cand = _join_spelled_name(match2.group(1).strip()).title()
                    if cand and cand.lower() not in _BAD_NAME_WORDS:
                        new_name = cand
                if new_name:
                    session.caller_name = new_name
                    session.touch()
                    spoken = (
                        f"Apologies! Thanks {session.caller_name}. How can I help you today? "
                        f"I can help with booking, cancellations, or checking availability."
                    )
                else:
                    # No usable name in the correction - don't keep the rejected
                    # name or a garbage capture, ask again cleanly instead.
                    session.caller_name = None
                    session.awaiting_name = True
                    session.name_reask_count += 1
                    session.touch()
                    if session.name_reask_count >= 2:
                        # Free-form re-asks keep failing - usually STT
                        # mishearing the name, not the caller's fault -
                        # switch to spelling instead of asking the same
                        # open question a third time.
                        spoken = (
                            "Let's try that differently. Could you spell your "
                            "name for me, letter by letter?"
                        )
                    else:
                        spoken = "Sorry about that. Could you tell me your name again please?"
            return self._quick(utterance, spoken, False, session, t0)

        if session.turn_count == 0 and session.caller_name is None:
            # First turn - greet and capture name. Normally unreached in the
            # voice demos (call_ui.py / demo.py call greet() proactively
            # before the caller says anything - see that method). Kept here
            # as a fallback for entry points that pass the caller's own
            # first utterance straight into run() (e.g. backend.py's HTTP
            # /turn endpoint), so the greeting still fires even if nothing
            # called greet() first.
            session.awaiting_name = True
            session.touch()
            spoken = GREETING
            return self._quick(utterance, spoken, False, session, t0)

        # 1b. Name correction, mid-call. Only reachable here once name capture/
        # confirm has already resolved (both return early above otherwise).
        _name_fix = _NAME_CORRECTION_RE.search(utterance)
        if _name_fix and session.caller_name:
            new_name = next(g for g in _name_fix.groups() if g)
            new_name = _join_spelled_name(new_name.strip()).title()
            old_name = session.caller_name
            session.caller_name = new_name
            session.touch()
            spoken = (
                f"Apologies for the mix-up! I've corrected that from {old_name} to "
                f"{new_name}. Now, how can I help you today?"
            )
            return self._quick(utterance, spoken, False, session, t0)

        # 1c. Third-party booking ("for my son"). Side-effect only, no
        # return - runs every turn so it can be said at any point in the
        # call, and lets normal processing continue underneath it.
        _ob = _ON_BEHALF_RE.search(utterance)
        if _ob:
            session.on_behalf_of = _ob.group(1).lower()
            session.touch()

        # 1d. Doctor/practitioner request. Only intercepts the turn when
        # there's no active slot negotiation - mid pending_suggestion, a
        # doctor mention should be noted (see 2's own check below) without
        # derailing the yes/no flow already in progress.
        _doc = _DOCTOR_RE.search(utterance)
        if _doc and not session.pending_suggestion:
            name_group = _doc.group(1)
            practitioner = f"Dr {name_group.title()}" if name_group else "a specific doctor"
            session.requested_practitioner = practitioner
            session.touch()
            _save_message(session_id, session.caller_name,
                          f"[practitioner preference] caller requested {practitioner}")
            name_part = f", {session.caller_name}" if session.caller_name else ""
            spoken = (
                f"I'm not able to guarantee a specific doctor on this automated "
                f"line{name_part}, but I've noted that you'd like {practitioner} and "
                f"passed it to the team - they'll do their best to accommodate it. "
                f"In the meantime, I can still check availability or book you in. "
                f"What would you like?"
            )
            return self._quick(utterance, spoken, False, session, t0)

        # 2. Pending slot confirmation - caller said yes/no to a suggestion
        if session.pending_suggestion:
            u = utterance.lower()
            _u_stripped = u.strip().rstrip(".!?")

            # Caller is done, mid-negotiation. Checked before yes/no below:
            # "no thank you" / "no thanks" contain "no" and used to be read
            # as "reject this slot, offer me another" (no_words matched on
            # the bare "no" substring), so the call never actually ended -
            # it just kept fishing for a slot nobody wanted. A real declined
            # slot never gets confirmed, so there is nothing to un-book here.
            if _u_stripped in _BYE_EXACT or any(ph in _u_stripped for ph in _BYE_PHRASES):
                sm.close(session_id)
                return {
                    "raw_output":  "",
                    "action":      {"action": "end_call"},
                    "validated":   True,
                    "spoken":      "No problem. Thank you for calling. Have a good day. Goodbye.",
                    "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
                    "entities":    {},
                    "end_call":    True,
                    "caller_name": session.caller_name,
                }

            # Caller didn't hear the offer, mid-negotiation. Without this,
            # "sorry, what?" matched none of yes/no/date/service below, fell
            # straight out of this block, and got sent to the LLM as a brand
            # new out_of_scope-looking utterance - the pending offer was
            # still set internally but never re-spoken, so the caller was
            # just met with a confused "I can help with booking..." reply
            # instead of hearing the slot again.
            _REPEAT_EXACT = {"sorry", "pardon", "what", "huh", "come again", "one more time"}
            _REPEAT_PHRASES = ("say that again", "repeat that", "can you repeat",
                                "didn't catch that", "didnt catch that", "excuse me",
                                "say again")
            if _u_stripped in _REPEAT_EXACT or any(p in _u_stripped for p in _REPEAT_PHRASES):
                slot = session.pending_suggestion
                spoken = f"Sure - {describe_slot(slot)}. Does that work for you?"
                return self._quick(utterance, spoken, False, session, t0)

            yes_words = {"yes", "yeah", "yep", "sure", "ok", "okay", "please",
                         "that works", "that's fine", "sounds good", "sounds great",
                         "perfect", "great", "go on", "go ahead", "that'll do",
                         "that will do", "lovely", "brilliant", "works for me",
                         "do that", "book it", "book that", "let's do it",
                         "yes please", "sounds perfect"}
            # "next" (bare) was removed - it collided with "next consultation
            # available" / "next Tuesday" style requests, which are a service
            # or date change, not a plain decline. "next one"/"next slot"
            # keep the plain-decline meaning ("just give me the next slot").
            no_words  = {"no", "nope", "nah", "different", "another", "else",
                         "not that", "other", "later", "earlier", "next one",
                         "next slot", "not now", "leave it", "forget it",
                         "some other time"}
            # Words that mean "a different day entirely" (not just a later time slot)
            diff_day_words = {"date", "day", "week", "monday", "tuesday", "wednesday",
                              "thursday", "friday"}

            if _word_match(u, yes_words):
                slot = session.pending_suggestion
                name_part = f", {session.caller_name}" if session.caller_name else ""
                if book_slot(slot["date"], slot["time"], slot["service"]):
                    # Read before clear() - queued_service itself survives
                    # clear() by design (see session_manager.Session), but
                    # capturing it here keeps this branch readable either way.
                    _queued = session.queued_service
                    behalf_note = _behalf_note(session)
                    session.clear()
                    # Was hand-building the date string here (a second copy of
                    # calendar_store's _fmt_slot logic, already out of sync - it
                    # said "Monday 13th July" while the shared formatter now says
                    # "Monday the 13th of July"). Reusing describe_slot() also
                    # fixes "I've booked your general for..." (missing the word
                    # "appointment") since it uses the proper service labels.
                    _save_message(session_id, session.caller_name,
                                  f"[booking confirmed] {describe_slot(slot)}")
                    if _queued:
                        # Second service from an earlier "a consultation and a
                        # follow-up" request - chain straight into offering it
                        # instead of ending the flow, so it doesn't get
                        # silently dropped once this booking clears the session.
                        session.queued_service = None
                        next_slot = get_next_slot(service=_queued, skip=0)
                        svc_label = _SERVICE_LABELS.get(_queued, _queued)
                        if next_slot:
                            session.pending_suggestion = next_slot
                            session.touch()
                            spoken = (
                                f"Brilliant{name_part}. I've booked {describe_slot(slot)}."
                                f"{behalf_note} Now, for the {svc_label}, how about "
                                f"{describe_slot(next_slot)}? Would that work too?"
                            )
                        else:
                            _save_message(session_id, session.caller_name,
                                          f"[callback requested] no availability for "
                                          f"queued service={_queued}")
                            spoken = (
                                f"Brilliant{name_part}. I've booked {describe_slot(slot)}."
                                f"{behalf_note} I'm afraid I don't have anything available "
                                f"for the {svc_label} right now - I'll get a member of the "
                                f"team to call you back about that one. Is there anything "
                                f"else I can help you with?"
                            )
                            # Chain has ended (nothing more to offer for this
                            # person) even though it didn't end cleanly -
                            # same reset as the plain no-queue branch below.
                            session.on_behalf_of = None
                    else:
                        spoken = (
                            f"Brilliant{name_part}. I've booked {describe_slot(slot)}."
                            f"{behalf_note} Is there anything else I can help you with?"
                        )
                        # Booking flow for this person is fully resolved -
                        # clear so a later, unrelated request in the same
                        # call ("do you have anything on Monday?", nothing
                        # to do with the son) doesn't inherit a stale
                        # "This booking is for your son" note. Kept alive
                        # while a queued_service chain is still running
                        # (both branches above), since that's still the
                        # same person's booking.
                        session.on_behalf_of = None
                    return self._quick(utterance, spoken, False, session, t0)
                else:
                    # Someone else booked this exact slot between it being
                    # offered and confirmed (a real race, not hypothetical -
                    # book_slot() now actually checks and reports this
                    # instead of silently double-booking). Don't claim
                    # success, find the next real slot and offer that.
                    next_slot = get_next_slot(
                        service=slot["service"], skip=session.suggestion_index + 1)
                    if next_slot:
                        session.pending_suggestion = next_slot
                        session.suggestion_index += 1
                        session.touch()
                        spoken = (
                            f"I'm sorry{name_part}, that slot was just taken by another "
                            f"caller. The next available is {describe_slot(next_slot)}. "
                            f"Does that work for you?"
                        )
                    else:
                        session.pending_suggestion = None
                        spoken = (
                            f"I'm sorry{name_part}, that slot was just taken by another "
                            f"caller, and I don't have another one immediately available. "
                            f"I'll get a member of the team to call you back."
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
                        req_day = f"{req_d.strftime('%A')} the {_ordinal(req_d.day)} of {req_d.strftime('%B')}"
                        spoken = (
                            f"I'm afraid I don't have anything available on {req_day}. "
                            f"The nearest I have is {describe_slot(slot)}. Would that work for you?"
                        )
                    else:
                        spoken = f"How about {describe_slot(slot)}? Would that suit you?"
                    return self._quick(utterance, spoken, False, session, t0)
                else:
                    # Used to ask "could I take your number" here, but nothing
                    # in the app ever captured the reply - the next thing the
                    # caller said just got sent to the LLM as an unrelated
                    # new turn. Asking for information you then throw away is
                    # worse than not asking, so this no longer asks for
                    # anything new; it logs the request so a human has
                    # something to act on (same pattern as cancellation
                    # requests and booking confirmations). Read the service
                    # before clearing pending_suggestion below, not after.
                    _svc = session.pending_suggestion.get("service")
                    session.pending_suggestion = None
                    _save_message(session_id, session.caller_name,
                                  f"[callback requested] no availability near requested date "
                                  f"for service={_svc}")
                    spoken = (
                        "I'm afraid I don't have any availability around that date. "
                        "I'll get a member of the team to call you back during business hours."
                    )
                    return self._quick(utterance, spoken, False, session, t0)

            # Explicit service switch ("make it a consultation", "when's the
            # next consultation available", "actually a follow-up"). This is
            # the real bug from the 2026-07-14 transcript: "Can we make it a
            # consultation? When is the next consultation available?" has
            # "next" in it, which used to be a no_words match, so it fell
            # into the plain-decline branch below and searched the next slot
            # of the OLD service (general) - the caller's actual request to
            # switch service was silently dropped and they were offered a
            # general appointment again. Checked before no_words so a service
            # mention always wins over an incidental "next"/"different" in
            # the same sentence. Only fires when the requested service
            # differs from what's already pending, so "yes, a consultation
            # is fine" (restating the same service while accepting) doesn't
            # get reinterpreted as a switch - that's handled by yes_words above.
            _req_service = _ents.get("service")
            if _req_service and _req_service != session.pending_suggestion.get("service"):
                session.suggestion_index = 0
                slot = get_next_slot(
                    service=_req_service,
                    preferred_date=session.pending_suggestion.get("date"),
                    skip=0,
                )
                if not slot:
                    # Nothing on/near the currently offered date for the new
                    # service - fall back to the earliest slot of that
                    # service anywhere in the calendar, rather than giving up.
                    slot = get_next_slot(service=_req_service, skip=0)
                if slot:
                    session.pending_suggestion = slot
                    session.touch()
                    # "Can I book a consultation and a follow-up" said while
                    # a different service is already pending goes through
                    # this switch branch, not the fresh-request branches -
                    # needs the same second-service queue check or the
                    # follow-up half gets silently dropped again.
                    multi_note = _queue_second_service(utterance, _req_service, session)
                    spoken = f"How about {describe_slot(slot)}?{multi_note} Would that suit you?"
                else:
                    _save_message(session_id, session.caller_name,
                                  f"[callback requested] no availability for service={_req_service}")
                    session.pending_suggestion = None
                    spoken = (
                        "I'm afraid I don't have any availability for that at the moment. "
                        "I'll get a member of the team to call you back during business hours."
                    )
                return self._quick(utterance, spoken, False, session, t0)

            # Bare time-only correction ("do you have anything at 2pm",
            # "what about half past three") with no date change and no
            # service change. Previously fell through unhandled - not a
            # yes/no word, not a date entity, not a service switch - so it
            # either matched no_words on an unrelated substring or reached
            # the LLM as an unrelated new turn, losing the pending offer
            # either way. Same-day-first: try to find that exact time on the
            # currently offered date, then the nearest later time that same
            # day, then fall back to the first future slot at that exact
            # time on any day. Single find_slots() call (skip=0, first 10),
            # same effort level as the date-request check above it - not the
            # 30-page exhaustive scan the plain-decline branch uses below.
            _req_time = _ents.get("time_resolved")
            if _req_time and _req_time != session.pending_suggestion.get("time"):
                _svc = session.pending_suggestion.get("service")
                _pref_date = session.pending_suggestion.get("date")
                _candidates = find_slots(service=_svc, preferred_date=_pref_date, skip=0)
                _same_day = [s for s in _candidates if s["date"] == _pref_date]
                slot = next((s for s in _same_day if s["time"] == _req_time), None)
                if not slot:
                    _later_same_day = sorted(
                        (s for s in _same_day if s["time"] >= _req_time),
                        key=lambda s: s["time"])
                    slot = _later_same_day[0] if _later_same_day else None
                if not slot:
                    slot = next((s for s in _candidates if s["time"] == _req_time), None)
                if not slot and _candidates:
                    slot = _candidates[0]
                if slot:
                    session.pending_suggestion = slot
                    session.suggestion_index = 0
                    session.touch()
                    spoken = f"How about {describe_slot(slot)}? Would that suit you?"
                else:
                    _save_message(session_id, session.caller_name,
                                  f"[callback requested] no slot near time={_req_time} "
                                  f"for service={_svc}")
                    session.pending_suggestion = None
                    spoken = (
                        "I'm afraid I don't have anything at that time. "
                        "I'll get a member of the team to call you back during business hours."
                    )
                return self._quick(utterance, spoken, False, session, t0)

            if _word_match(u, no_words):
                session.suggestion_index += 1
                # Track exactly which slot just got turned down, by identity,
                # not by position. The old approach re-searched with a global
                # "skip" counter against a sort that re-pivots on
                # preferred_date every time (preferred_date follows whatever
                # was most recently suggested), so "skip N" didn't reliably
                # mean "N slots further on" - it could revisit the same
                # handful of slots forever without ever reaching a real
                # exhaustion, which is exactly the infinite-loop
                # test_pipeline.py caught (25 rejections, never ran out).
                # Filtering out explicitly-rejected slots guarantees forward
                # progress regardless of how the sort shifts underneath it.
                _rejected = session.pending_suggestion
                if _rejected:
                    session.rejected_slots.append(
                        (_rejected["date"], _rejected["time"], _rejected["service"]))

                def _first_unrejected(candidates):
                    for s in candidates:
                        if (s["date"], s["time"], s["service"]) not in session.rejected_slots:
                            return s
                    return None

                # "later date" / "another day" = jump to NEXT calendar day entirely.
                # "later" / "earlier" alone = same day different time -> keep preferred_date.
                _wants_diff_day = _word_match(u, diff_day_words)
                if _wants_diff_day:
                    # Filter all future slots to those strictly after the current suggested date
                    _current_date = session.pending_suggestion.get("date")
                    _future = [s for s in find_slots(service=session.pending_suggestion.get("service"))
                               if s["date"] > _current_date]
                    slot = _first_unrejected(_future)
                else:
                    # Bounded scan through find_slots' 10-at-a-time pages,
                    # skipping past anything already rejected, until either a
                    # fresh slot turns up or the calendar genuinely runs dry.
                    # 30 pages = 300 candidates, comfortably more than a
                    # 4-week calendar can ever hold for one service.
                    svc = session.pending_suggestion.get("service")
                    pref_date = session.pending_suggestion.get("date")
                    slot = None
                    batch_skip = 0
                    for _ in range(30):
                        batch = find_slots(service=svc, preferred_date=pref_date, skip=batch_skip)
                        if not batch:
                            break
                        slot = _first_unrejected(batch)
                        if slot:
                            break
                        batch_skip += 10
                if slot:
                    session.pending_suggestion = slot
                    session.touch()
                    spoken = f"How about {describe_slot(slot)}? Would that suit you?"
                    return self._quick(utterance, spoken, False, session, t0)
                else:
                    # Same dead-end fix as above - see comment there. Read
                    # the service before clearing pending_suggestion.
                    _svc = session.pending_suggestion.get("service")
                    session.pending_suggestion = None
                    _save_message(session_id, session.caller_name,
                                  f"[callback requested] no more available slots for service={_svc}")
                    spoken = (
                        "I'm afraid I don't have any more available slots at the moment. "
                        "I'll get a member of the team to call you back during business hours."
                    )
                    return self._quick(utterance, spoken, False, session, t0)

        # 3. End-call detection (model-independent). A caller saying goodbye should end the
        # call, not be sent to the model and misread as out_of_scope. Kept to clear terminal
        # phrases so it never hijacks a normal turn; declines to a proposed slot are handled
        # above via pending_suggestion, so "no" alone does not reach here mid-negotiation.
        # Substring-matched (not exact) so preambles like "I said no thanks" or
        # "um, thank you" still close the call. "thanks"/"thank you" bare is
        # deliberately included: after "anything else I can help with?" a bare
        # thanks means the caller is done, matching what _mock_output already
        # treats as end_call - keeps real-model and mock behaviour consistent
        # instead of the real model guessing clarify/out_of_scope on it.
        _u = utterance.lower().strip().rstrip(".!?")
        if _u in _BYE_EXACT or any(ph in _u for ph in _BYE_PHRASES):
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
            # Only trust the model's date field when the caller's own words
            # actually contained a date entity. Otherwise the model can (and
            # does) hallucinate a plausible-looking date for a request like
            # "check availability" where none was given, which then gets
            # treated as a preferred_date and skips over an earlier slot that
            # was genuinely free (e.g. caller asks generally, model guesses
            # the 15th, get_next_slot returns the 15th even though the 14th
            # was free and chronologically first). Mirrors the has_date guard
            # already used for the booking branch below.
            has_date = bool(entities.get("date_resolved"))
            preferred_date = getattr(validated, "date", None) if has_date else None
            service        = getattr(validated, "service", None)
            svc_str        = _safe_service(service.value if service else None, entities)
            slot = get_next_slot(service=svc_str, preferred_date=preferred_date,
                                 skip=session.suggestion_index)
            if slot:
                session.pending_suggestion = slot
                session.touch()
                # Keep the logged/displayed action JSON honest: `parsed` still
                # holds the model's raw fields (possibly a hallucinated date
                # the guard above just overrode), and left as-is it shows a
                # date in the console/transcript that disagrees with what's
                # actually spoken - looks like a bug even when the booking
                # logic itself is correct. Sync it to the real slot chosen.
                parsed["date"] = slot["date"]
                parsed["service"] = slot["service"]
                name_part = f", {session.caller_name}" if session.caller_name else ""
                multi_note = _queue_second_service(utterance, svc_str, session)
                spoken = (
                    f"Of course{name_part}. The next available slot is "
                    f"{describe_slot(slot)}.{_behalf_note(session)}{multi_note} "
                    f"Does that work for you?"
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
                    "Could you try saying the date, like Tuesday the 8th, "
                    "the time, like ten in the morning or half past two, "
                    "and the type of appointment: general, consultation, or follow-up?"
                )
            elif c == 3:
                spoken = (
                    "I'm still not quite catching that. "
                    "Let me try once more. Please say something like: "
                    "book a general appointment on Monday at ten in the morning."
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
                    booked_svc = _safe_service(validated.service.value, entities)
                    if booked_svc != validated.service.value:
                        # Keep the spoken confirmation in sync with what's
                        # actually booked - never say "follow-up" while
                        # booking "general" underneath it.
                        validated.service = validated.service.__class__(booked_svc)
                        parsed["service"] = booked_svc  # keep the logged action in sync too
                    if book_slot(validated.date, validated.time, booked_svc):
                        spoken = render_confirmation(validated)
                        note = _behalf_note(session)
                        multi_note = _queue_second_service(utterance, booked_svc, session)
                        _queued = session.queued_service
                        if _queued:
                            # Same chaining as the pending_suggestion yes-branch:
                            # don't let the second service get silently dropped.
                            session.queued_service = None
                            next_slot = get_next_slot(service=_queued, skip=0)
                            svc_label = _SERVICE_LABELS.get(_queued, _queued)
                            if next_slot:
                                session.pending_suggestion = next_slot
                                session.touch()
                                tail = (f"Now, for the {svc_label}, how about "
                                        f"{describe_slot(next_slot)}? Would that work too?")
                                # Chain continues - same person, keep the note.
                            else:
                                _save_message(session_id, session.caller_name,
                                              f"[callback requested] no availability for "
                                              f"queued service={_queued}")
                                tail = (f"I'm afraid I don't have anything available for "
                                        f"the {svc_label} right now - I'll get a member of "
                                        f"the team to call you back about that one. Is there "
                                        f"anything else I can help you with?")
                                # Chain ended (nothing left to offer) - clear so
                                # it doesn't stick to a later unrelated request.
                                session.on_behalf_of = None
                            spoken = spoken.replace(
                                "Is there anything else I can help you with?", tail)
                        else:
                            if note or multi_note:
                                spoken = spoken.replace(
                                    "Is there anything else I can help you with?",
                                    f"{note.strip()}{multi_note} Is there anything else "
                                    f"I can help you with?")
                            # No queued service - this booking is the whole
                            # flow for this person, fully resolved now. Clear
                            # so a later, unrelated request in the same call
                            # ("do you have anything on Monday?") doesn't
                            # inherit a stale "for your son" note.
                            session.on_behalf_of = None
                    else:
                        # Caller gave an exact date+time that's no longer free
                        # (someone else took it first, or it was never really
                        # available and the model didn't check) - same honesty
                        # fix as the pending_suggestion yes-branch above, offer
                        # a real alternative instead of confirming a phantom.
                        alt = get_next_slot(service=booked_svc, preferred_date=validated.date)
                        name_part = f", {session.caller_name}" if session.caller_name else ""
                        if alt:
                            session.pending_suggestion = alt
                            session.touch()
                            spoken = (
                                f"I'm sorry{name_part}, that slot isn't available. "
                                f"The nearest I have is {describe_slot(alt)}. "
                                f"Would that work for you?"
                            )
                        else:
                            spoken = (
                                f"I'm sorry{name_part}, that slot isn't available and I "
                                f"don't have another one nearby right now. I'll get a "
                                f"member of the team to call you back."
                            )
                else:
                    svc = _safe_service(
                        validated.service.value if getattr(validated, "service", None) else None,
                        entities)
                    slot = get_next_slot(service=svc,
                                         preferred_date=(validated.date if has_date else None))
                    if slot:
                        session.pending_suggestion = slot
                        session.touch()
                        # Same reasoning as the check_availability branch above:
                        # `parsed` still has the model's raw (possibly vague or
                        # hallucinated) date/service, sync it to the real slot
                        # being offered so the logged action matches the reply.
                        parsed["date"] = slot["date"]
                        parsed["time"] = slot["time"]
                        parsed["service"] = slot["service"]
                        name_part = f", {session.caller_name}" if session.caller_name else ""
                        multi_note = _queue_second_service(utterance, svc, session)
                        spoken = (f"Of course{name_part}. The next available slot is "
                                  f"{describe_slot(slot)}.{_behalf_note(session)}{multi_note} "
                                  f"Does that work for you?")
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
            elif validated.action.value == "clarify" and "date" in (validated.missing_fields or []):
                # Don't ask an open "which date would you like" question when
                # we can just offer a real slot instead - the caller doesn't
                # know the calendar, so a concrete first offer to accept or
                # reject is faster than a blind date request, and reuses the
                # same yes/no + "how about a specific day" flow as everywhere
                # else (see pending_suggestion handling above).
                svc_str = _safe_service(
                    validated.service.value if getattr(validated, "service", None) else None,
                    entities)
                slot = get_next_slot(service=svc_str, skip=session.suggestion_index)
                if slot:
                    session.pending_suggestion = slot
                    session.touch()
                    parsed["date"] = slot["date"]
                    parsed["time"] = slot["time"]
                    parsed["service"] = slot["service"]
                    name_part = f", {session.caller_name}" if session.caller_name else ""
                    multi_note = _queue_second_service(utterance, svc_str, session)
                    spoken = (
                        f"Of course{name_part}. The earliest I have available is "
                        f"{describe_slot(slot)}.{_behalf_note(session)}{multi_note} Would that "
                        f"work, or did you have a different date in mind?"
                    )
                else:
                    spoken = render_confirmation(validated)
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

    def greet(self, session_id: str = "default") -> dict:
        """Return the call-opening greeting without needing caller input.

        The caller shouldn't have to say something first before the
        assistant speaks - a real receptionist answers and talks
        immediately. Call this once, right when a call starts, before
        listening for anything; speak its `spoken` text, then start
        listening for the caller's name as normal.

        This does not go through run()'s empty-input guard (which would
        return "I didn't catch that" for a blank utterance) - it sets up
        the session directly, mirroring the turn_count==0 branch inside
        run() so the two stay in sync if the greeting logic changes.
        """
        try:
            import src.session_manager as sm
        except ImportError:
            import session_manager as sm  # type: ignore[no-redef]

        t0 = time.perf_counter()
        session = sm.get_or_create(session_id)
        session.awaiting_name = True
        session.touch()
        return self._quick("", GREETING, False, session, t0)

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
