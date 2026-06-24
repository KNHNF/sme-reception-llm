# SME Voice Assistant

Offline voice assistant for SME appointment booking. Caller speaks, system books the appointment. No cloud, no GPU required at inference time.

UWE Bristol MSc Data Science - Group 6

## What it does

```
mic -> Faster-Whisper STT -> spaCy NER -> fine-tuned LLM -> JSON action -> Piper TTS -> spoken response
```

Handles booking, cancellations, availability checks, name capture, calendar suggestions, profanity filtering, and end-of-call detection. Runs fully on CPU.

## Eval results

| Condition                 | Action Accuracy | Exact Match | Latency P50 |
|---------------------------|-----------------|-------------|-------------|
| Phi-3 mini (vanilla)      | 0.4%            | 0%          | 4410ms      |
| Phi-3 mini (fine-tuned)   | 98.1%           | 70.6%       | 3556ms      |
| Llama 3.2 3B (vanilla)    | 0.0%            | 0%          | 2791ms      |
| Llama 3.2 3B (fine-tuned) | **99.8%**       | 70.4%       | 3703ms      |

480-sample, 4-condition evaluation. Training: 600 synthetic samples, Kaggle T4, ~60-70 min per model.

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# text mode - safest, no audio hardware needed
python demo.py --text --no-tts

# full voice mode
python demo.py

# Streamlit UI
streamlit run app.py          # http://localhost:8501

# API server
uvicorn backend:app --port 5005
# Swagger:             http://localhost:5005/docs
# Metrics dashboard:   http://localhost:5005/metrics-dashboard
```

## Streamlit UI

The main demo interface. Imports the pipeline directly - no API server needed.

Features: chat bubbles, per-turn action badges (BOOK / CANCEL / AVAIL. / OOS / CLARIFY), latency counter, spaCy entity tags, Piper TTS inline audio playback, model selector (Mock / Phi-3 FT / Llama 3.2 FT), session reset.

Mock mode works on any machine. Fine-tuned models require an NVIDIA GPU.

## File map

```
src/
  inference.py          pipeline: profanity -> name capture -> slot confirm -> LLM
  entity_extractor.py   spaCy NER: DATE, TIME, SERVICE, PERSON
  session_manager.py    in-memory session state per call
  sme_action_schema.py  Pydantic v2 discriminated union, 6 action types
  stt.py                Faster-Whisper microphone capture
  tts.py                Piper TTS subprocess wrapper
  profanity.py          3-strike keyword filter + de-escalation messages
  calendar_store.py     JSON calendar: find_next_slot(), book_slot(), Mon-Fri filter
  metrics_logger.py     SQLite logger -> data/metrics.db

data/
  calendar.json         mock 3-week appointment schedule

evaluation/
  eval_phi3.py          evaluation script - Phi-3 mini
  eval_llama3.py        evaluation script - Llama 3.2 3B

checkpoints/            QLoRA adapter weights (not in repo - too large for git)
  sme-phi3-qlora/
  sme-llama3-qlora/

app.py                  Streamlit UI
backend.py              FastAPI: /turn, /metrics, /metrics/clear, /metrics-dashboard
demo.py                 end-to-end demo (text or voice, mock or real model)
test_pipeline.py        smoke + edge case tests (111 checks, mock mode)
```

## Conversation flow

1. Greeting: "Thank you for calling. Could I take your name please?"
2. Name capture: regex strips preambles ("it's / my name is"), spaCy PERSON fallback, spelled names joined (J-A-C-K R-E-A-C-H-E-R -> Jack Reacher)
3. Name confirmation: "Did I get that as Jack Reacher?" - handles yes / correction / booking intent
4. Normal turns: utterance -> spaCy NER -> LLM -> validated JSON action
5. Availability: suggests next slot ("Thursday 25th June at 9:00 AM. Does that work?")
6. Caller says yes: books slot, confirmation with ordinal date
7. Caller says no: next slot same day; "later date / another day" jumps to next calendar day
8. Specific date request ("the 26th"): NER + ordinal regex fallback, explains if nothing available
9. Profanity: 3-strike, call ends on strike 3
10. "bye / thanks": end_call action, loop terminates

Calendar: Mon-Fri only. Past same-day slots and weekend slots are excluded.

## Testing

```bash
python test_pipeline.py
```

111 checks covering name capture, spelled names, booking intent loop guard, empty/noise/emoji input, profanity 3-strike sequence, out-of-scope requests, date edge cases (past dates, weekends, specific date requests, later-date navigation), slot exhaustion, unknown services, ambiguous confirmation, session reuse after booking, and rapid-fire turns.

## Piper TTS setup

Create a `piper/` folder in the project root and put these two files in it:

1. Piper binary: download from [github.com/rhasspy/piper/releases](https://github.com/rhasspy/piper/releases) - pick the Windows or Linux build
2. Voice model: download `en_US-lessac-medium.onnx` (and its `.json` config) from [huggingface.co/rhasspy/piper-voices](https://huggingface.co