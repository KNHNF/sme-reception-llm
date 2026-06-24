# SME Reception LLM - IGP Private Dev Repo

LLM-based real-time voice assistant for SME appointment booking.  
UWE Bristol MSc Data Science 

---

## What it does

A fully offline phone assistant for small businesses. Caller speaks → system books the appointment.

```
mic → Faster-Whisper STT → spaCy NER → fine-tuned LLM → JSON action → Piper TTS → spoken response
```

Handles: booking, cancellations, availability checks, name capture, calendar suggestions, profanity (3-strike), end-of-call detection. Runs on CPU, no cloud.

---

## Eval results

| Condition                 | Action Accuracy | Exact Match | Latency P50 |
|---------------------------|-----------------|-------------|-------------|
| Phi-3 mini (vanilla)      | 0.4%            | 0%          | 4410ms      |
| Phi-3 mini (fine-tuned)   | 98.1%           | 70.6%       | 3556ms      |
| Llama 3.2 3B (vanilla)    | 0.0%            | 0%          | 2791ms      |
| Llama 3.2 3B (fine-tuned) | **99.8%**       | 70.4%       | 3703ms      |

480-sample 4-condition eval. Training: 600 synthetic samples, Kaggle T4, ~60–70 min.

---

## Quick start

```bash
# Install (one time - already done on Karan's machine)
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run (choose one)
python demo.py --text --no-tts          # safest, no audio hardware
python demo.py --text --no-tts --record # saves TTS WAVs to recordings/
python demo.py                          # full voice mode

# API server
uvicorn backend:app --port 5005
# Swagger docs:        http://localhost:5005/docs
# Metrics dashboard:   http://localhost:5005/metrics-dashboard
# Metrics JSON:        http://localhost:5005/metrics
```

---

## File map

```
src/
  inference.py          Pipeline class: profanity → name → slot confirm → LLM
  entity_extractor.py   spaCy NER: DATE, TIME, SERVICE, PERSON
  session_manager.py    In-memory session: name, partial_action, confusion_count
  sme_action_schema.py  Pydantic v2 discriminated union - 6 action types
  stt.py                Faster-Whisper microphone capture
  tts.py                Piper TTS subprocess wrapper (dual-flag, detailed errors)
  profanity.py          3-strike keyword filter + de-escalation
  calendar_store.py     Reads data/calendar.json, find_next_slot(), book_slot()
  metrics_logger.py     SQLite logger: per-turn accuracy + latency → data/metrics.db

data/
  calendar.json         Mock 3-week appointment schedule
  metrics.db            Auto-created on first /turn call (gitignored)

evaluation/
  eval_phi3.py          Evaluation script - Phi-3 mini
  eval_llama3.py        Evaluation script - Llama 3.2 3B

checkpoints/            QLoRA adapter weights (gitignored - too large)
  sme-phi3-qlora/
  sme-llama3-qlora/

docs/                   (gitignored - private viva notes, paper, diagrams)
  SME_Viva_v3.pptx          12-slide 5-min mock viva deck (current)
  pipeline_flowchart.html   Interactive pipeline + 39 papers mapped
  paper_final.html          Academic paper, 38 citations, SVG result charts
  dashboard.html            Live metrics dashboard (Chart.js, auto-refresh 5s)
  CallFlow_Emails.md        Email drafts - Iheanyi Ibe + Mark Corderoy
  VIVA_GUIDE.md             Slide-by-slide viva guide
  PROJECT_FULL.md           Full technical reference
  BUGS_AND_FIXES.md         Dev issues log

backend.py              FastAPI app - /turn, /book, /cancel, /availability,
                        /metrics, /metrics/clear, /metrics-dashboard
demo.py                 End-to-end demo (text or voice, mock or real model)
```

---

## Conversation flow

1. Turn 0: greeting + "Could I take your name please?"
2. Turn 1: name captured — regex strips preambles ("it's / I'm / my name is"), spaCy PERSON fallback, spelled-name join (K-A-R-A-N → Karan, J-A-C-K R-E-A-C-H-E-R → Jack Reacher)
3. Turn 2: name confirmation ("Did I get that as X?") — yes/no/correction/booking-intent all handled; booking intent re-asks (max 2 times then bypasses with placeholder)
4. Normal turns: utterance → spaCy NER → LLM → JSON action
5. `check_availability` → `calendar_store` → "Next slot is X, does that work?"
6. Caller says yes → `book_slot()` → ordinal date confirmation ("Thursday 25th June")
7. Caller says no → suggest next slot (same day) or next day if "later date / another day"
8. Caller requests specific date → NER + ordinal regex fallback → "nothing on the 24th, nearest is..."
9. Confusion escalation: 4-step retry with format hints, ends call on 4th failure
10. Profanity: 3-strike, ends call on strike 3
11. "bye / goodbye / thanks" → `end_call` → loop terminates

Calendar: Mon–Fri only, past same-day slots excluded, weekend slots filtered.

---

## Docker

```bash
docker build -t sme-backend .
docker run -p 5005:5005 sme-backend
```

No GPU needed. Image ~500MB.

---

## Piper TTS setup

The `piper/` folder contains the compiled binary and voice model (61MB ONNX file).
It is gitignored in this private repo to avoid committing large binaries to GitHub.

To set up Piper locally:
1. Download from [github.com/rhasspy/piper/releases](https://github.com/rhasspy/piper/releases)
2. Download voice model: `en_US-lessac-medium.onnx` from [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
3. Place both in `piper/`

For the GitLab repo: the ONNX model is 61MB and requires Git LFS if committed.
Alternative: add the download instructions to the team README and exclude from git.

---

## Testing

```bash
python test_pipeline.py          # 111 checks, mock mode, no GPU needed
```

Covers: name capture, spelled names, booking intent loop guard, empty/noise/emoji input,
profanity 3-strike, out-of-scope, date edge cases (past, weekend, specific date, later date),
slot exhaustion, unknown service, ambiguous confirmation, session reuse after booking, rapid fire.

---

## Do NOT re-run

- `pip install` / `spacy download` - already done
- Training - checkpoints already saved
- Evaluation - 480-sample results already in `evaluation/`
- PPTX generation - use `docs/SME_Viva_v3.pptx` (12 slides, current)

---

## Streamlit UI

Visual web frontend for demos. Runs self-contained — no uvicorn needed:

```bash
streamlit run app.py          # opens at http://localhost:8501
```

The app imports the pipeline directly (embedded mode). If that fails it falls back to calling
the FastAPI backend at `localhost:5005` (API mode — run `uvicorn backend:app --port 5005` first).

Features: chat bubbles, per-turn action badges (BOOK / CANCEL / AVAIL. / OOS / CLARIFY),
latency display, spaCy entity tags, Piper TTS audio playback via browser, example prompts
sidebar, session turn counter, caller name display, new-call reset.

---

## Stack

Python 3.11, Faster-Whisper, spaCy en_core_web_sm, Phi-3 mini, Llama 3.2 3B,
QLoRA/PEFT, BitsAndBytes 4-bit NF4, FastAPI, Pydantic v2, Piper TTS, SQLite, Docker

---

## Resources and citations

### Models
- **Llama 3.2 3B** — Meta AI (2024). [HuggingFace: meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Phi-3 mini** — Abdin et al. (2024). [HuggingFace: microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **Why these models?** Both are 3–4B parameter, instruction-tuned, quantisable to 4-bit NF4,
  and run inference on CPU. The research question is whether small offline models, after domain
  fine-tuning, can replace cloud APIs (GPT-4, Alexa) for structured NLU tasks.
- **Why not Gemma?** Gemma 1 (available at training time) had weaker instruction-following
  benchmarks than Phi-3 and Llama 3.2 at equivalent parameter counts, and less mature
  QLoRA support in the `trl`/`peft` stack. Gemma 2 was not yet released.
- **Why not GPT-4 / cloud models?** GDPR prevents sending patient/client data to third-party
  servers. The offline constraint is the research contribution, not a limitation.

### Training
- **QLoRA** — Dettmers et al. (2023). [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
  — quantises the frozen backbone to 4-bit NF4, trains only low-rank adapter matrices
  (rank 16, ~4M parameters). Enables fine-tuning a 3B model on a free Kaggle T4 GPU (~60 min).
- **LoRA** — Hu et al. (2022). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Training compute** — [Kaggle T4 GPU notebooks](https://www.kaggle.com/) (free tier, 30h/week)
- **Synthetic training data** — 600 samples generated with Claude claude-opus-4-8 (Anthropic, 2024),
  covering 6 action types with varied phrasings, entities, and edge cases.

### Speech recognition (STT)
- **Faster-Whisper** — [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  — CTranslate2 backend, 4× faster than OpenAI Whisper on CPU, same model weights.
- **Whisper** — Radford et al. (2022). [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- **STT benchmark** — [LibriSpeech dev-clean](https://www.openslr.org/12) (Panayotov et al., 2015)
  — tiny: 17.3% WER, small: 8.5% WER.

### Named entity recognition (NER)
- **spaCy** — Honnibal & Montani (2017). [spacy.io](https://spacy.io/)
  — `en_core_web_sm` pipeline, CPU-only, extracts DATE / TIME / SERVICE / PERSON.

### Text-to-speech (TTS)
- **Piper TTS** — [github.com/rhasspy/piper](https://github.com/rhasspy/piper)
  — local neural TTS, ONNX runtime, no internet, `en_US-lessac-medium` voice.
- **VITS** — Kim et al. (2021). [arXiv:2106.06103](https://arxiv.org/abs/2106.06103)
  — Piper's underlying architecture.

### Evaluation
- 480-sample 4-condition evaluation (Phi-3 vanilla/FT, Llama 3.2 vanilla/FT).
- Metrics: action accuracy (primary), exact JSON match (secondary), latency P50.
- Evaluation scripts: `evaluation/eval_phi3.py`, `evaluation/eval_llama3.py`
