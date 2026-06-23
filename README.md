# SME Reception LLM — IGP Private Dev Repo

LLM-based real-time voice assistant for SME appointment booking.  
UWE Bristol MSc Data Science — Group 6 (Karan Homayounfar, 25065219)

---

## What it does

A fully offline phone assistant for small businesses. Caller speaks → system books the appointment.

```
mic → Faster-Whisper STT → spaCy NER → fine-tuned LLM → JSON action → Piper TTS → spoken response
```

Handles: booking, cancellations, availability checks, name capture, calendar suggestions, profanity (3-strike), end-of-call detection. Runs on CPU, no cloud.

---

## Eval results

| Condition              | Action Accuracy | Exact Match | Latency P50 |
|------------------------|-----------------|-------------|-------------|
| Phi-3 mini (vanilla)   | 0.4%            | 0%          | 4410ms      |
| Phi-3 mini (fine-tuned)| 98.1%           | 70.6%       | 3556ms      |
| Llama 3.2 3B (vanilla) | 0.0%            | 0%          | 2791ms      |
| Llama 3.2 3B (fine-tuned) | **99.8%**   | 70.4%       | 3703ms      |

480-sample 4-condition eval. Training: 600 synthetic samples, Kaggle T4, ~60–70 min.

---

## Quick start

```bash
# Install (one time — already done on Karan's machine)
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run (choose one)
python demo.py --text --no-tts          # safest, no audio hardware
python demo.py --text --no-tts --record # saves TTS WAVs to recordings/
python demo.py                          # full voice mode

# API server
uvicorn backend:app --port 5005
# Swagger docs: http://localhost:5005/docs
```

---

## File map

```
src/
  inference.py          Pipeline class: profanity → name → slot confirm → LLM
  entity_extractor.py   spaCy NER: DATE, TIME, SERVICE, PERSON
  session_manager.py    In-memory session: name, partial_action, confusion_count
  sme_action_schema.py  Pydantic v2 discriminated union — 6 action types
  stt.py                Faster-Whisper microphone capture
  tts.py                Piper TTS subprocess wrapper
  profanity.py          3-strike keyword filter + de-escalation
  calendar_store.py     Reads data/calendar.json, find_next_slot(), book_slot()

data/
  calendar.json         Mock 3-week appointment schedule

evaluation/
  eval_phi3.py          Evaluation script — Phi-3 mini
  eval_llama3.py        Evaluation script — Llama 3.2 3B

checkpoints/            QLoRA adapter weights (gitignored — too large)
  sme-phi3-qlora/
  sme-llama3-qlora/

docs/                   (gitignored — private viva notes, paper, diagrams)
  pipeline_flowchart.html   Interactive pipeline + 39 papers mapped
  paper_final.html          Academic paper, 38 citations, SVG result charts
  SME_Viva_v2.pptx          20-slide viva deck (use this, not the old one)
  VIVA_GUIDE.md             Slide-by-slide guide + email drafts
  PROJECT_FULL.md           Full technical reference + YouTube links
  BUGS_AND_FIXES.md         7 documented bugs and fixes

backend.py              FastAPI app — /turn, /book, /cancel, /availability
demo.py                 End-to-end demo (text or voice, mock or real model)
```

---

## Conversation flow

1. Turn 0: greeting + "Could I take your name please?" + recording notice
2. Turn 1: name captured (regex strips "it's / I'm / my name is" preamble)
3. Normal turns: utterance → spaCy NER → LLM → JSON action
4. `check_availability` → `calendar_store` → "Next slot is X, does that work?"
5. Caller says yes → `book_slot()` → confirmation with name
6. Caller says no → suggest next slot
7. Confusion escalation: 4-step retry with format hints, ends call on 4th failure
8. Profanity: 3-strike, ends call on strike 3
9. "bye / goodbye / thanks" → `end_call` → loop terminates

---

## Docker

```bash
docker build -t sme-backend .
docker run -p 5005:5005 sme-backend
```

No GPU needed. Image ~500MB.

---

## Do NOT re-run

- `pip install` / `spacy download` — already done
- Training — checkpoints already saved
- Evaluation — 480-sample results already in `evaluation/`
- PPTX generation — use `docs/SME_Viva_v2.pptx` (20 slides)

---

## Stack

Python 3.11, Faster-Whisper, spaCy en_core_web_sm, Phi-3 mini, Llama 3.2 3B, QLoRA/PEFT, BitsAndBytes 4-bit NF4, FastAPI, Pydantic v2, Piper TTS, Docker
