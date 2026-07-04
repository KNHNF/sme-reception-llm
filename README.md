# SME Voice Assistant

Offline voice assistant for SME appointment booking. Caller speaks, system books the appointment. No cloud, no GPU required at inference time.

UWE Bristol MSc Data Science - Group 6

## What it does

```
mic -> Faster-Whisper STT -> spaCy NER -> fine-tuned LLM -> JSON action -> Piper TTS -> spoken response
```

Handles booking, cancellations, availability checks, name capture, calendar suggestions, profanity filtering, and end-of-call detection. Runs fully on CPU.

## Eval results

### GPU (Kaggle T4, 480 samples)

| Condition                 | Action Accuracy | Exact Match | Latency P50 |
|---------------------------|-----------------|-------------|-------------|
| Phi-3 mini (vanilla)      | 0.4%            | 0%          | 4410ms      |
| Phi-3 mini (fine-tuned)   | 98.1%           | 70.6%       | 3556ms      |
| Llama 3.2 3B (vanilla)    | 0.0%            | 0%          | 2791ms      |
| Llama 3.2 3B (fine-tuned) | **99.8%**       | 70.4%       | 3703ms      |

Training: 600 synthetic samples, Kaggle T4, ~60-70 min per model.

### CPU (no GPU, GGUF Q3_K_M, 30 samples)

| Model         | Quant   | Action Accuracy | Latency P50 |
|---------------|---------|-----------------|-------------|
| Phi-3 mini    | Q3_K_M  | 66.7%           | 2670ms      |
| Llama 3.2 3B  | Q3_K_M  | 86.7%           | 1938ms      |

Run via llama.cpp server. No GPU or internet required. Results in `evaluation/cpu_results/`.

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

## CPU inference (no GPU)

Runs the fine-tuned models locally via GGUF quantisation and llama.cpp. No GPU, no internet.

```bash
# 1. Merge QLoRA adapter into base model
python scripts/01_merge_adapter.py --model phi3
python scripts/01_merge_adapter.py --model llama3

# 2. Convert to GGUF and quantise
python scripts/02_convert_gguf.py --model phi3 --quant Q3_K_M
python scripts/02_convert_gguf.py --model llama3 --quant Q3_K_M

# 3. Start the llama.cpp server (port 8080)
python scripts/03_cpu_server.py --model phi3
python scripts/03_cpu_server.py --model llama3

# 4. Run evaluation (in a second terminal)
python scripts/04_cpu_eval.py --model phi3 --quant Q3_K_M
python scripts/04_cpu_eval.py --model llama3 --quant Q3_K_M
```

Requires llama.cpp binaries in `tools/llama_cpp/bin/`. Download `llama-bXXXX-bin-win-cpu-x64.zip` from [github.com/ggerganov/llama.cpp/releases](https://github.com/ggerganov/llama.cpp/releases) and extract into `tools/llama_cpp/bin/`. The `tools/` folder is gitignored.

## File map

```
src/
  inference.py          pipeline: profanity -> name capture -> slot confirm -> LLM
  entity_extractor.py   spaCy NER: DATE, TIME, SERVICE, PERSON
  session_manager.py    in-memory session state per call
  sme_action_schema.py  Pydantic v2 discriminated union, 6 action types
  stt.py                Faster-Whisper microphone capture
  tts.py                Piper TTS subprocess wrapper (dual-flag retry for Windows)
  profanity.py          3-strike keyword filter + de-escalation messages
  calendar_store.py     JSON calendar: find_next_slot(), book_slot(), Mon-Fri filter
  metrics_logger.py     SQLite logger -> data/metrics.db

scripts/
  01_merge_adapter.py   merge QLoRA adapter into base model
  02_convert_gguf.py    convert merged model to GGUF + quantise
  03_cpu_server.py      start llama.cpp server on port 8080
  04_cpu_eval.py        run 30-case eval against llama.cpp server

data/
  calendar.json         mock 3-week appointment schedule

evaluation/
  eval_phi3.py          GPU evaluation script - Phi-3 mini
  eval_llama3.py        GPU evaluation script - Llama 3.2 3B
  cpu_results/          CPU eval JSON results and summary

checkpoints/            QLoRA adapter weights + merged models + GGUF files (gitignored)

app.py                  Streamlit UI
backend.py              FastAPI: /turn, /metrics, /metrics/clear, /metrics-dashboard
demo.py                 end-to-end demo (text or voice, mock or real model)
test_pipeline.py        smoke + edge case tests (111 checks, mock mode)
```

## Real audio collection

A voice-collector app is live on Render (private project, CallFlow). It accepts audio recordings from the public and stores them to Supabase. Used to gather real caller speech to supplement the synthetic eval set. If you want to contribute a recording, ask Karan for the link.

After collecting 20+ real samples, run a real-audio eval against the CPU model using `scripts/04_cpu_eval.py` on actual transcriptions.

## Local voice demo (talk to your laptop)

Runs the full pipeline end-to-end: mic input, Faster-Whisper STT, LLM, Piper TTS out of your speakers. No phone needed.

```bash
# Set up Piper TTS first (see Piper TTS setup section below), then:
python demo.py
```

The system greets you, listens via your mic, processes through the pipeline, and speaks back. Say "bye" to end the call.

If mic input fails, check that your default recording device is set correctly in Windows sound settings (sounddevice uses the system default).

For a demo recording: use OBS or Windows Game Bar (Win+G) to capture screen and mic audio simultaneously.

## Twilio phone call demo

Lets anyone call a real phone number and talk to the system. Twilio transcribes the caller's speech, sends it to your local server, and speaks back the response. You need the API server running and a public URL via cloudflared.

**Step 1: Get a free trial phone number**

Go to [console.twilio.com](https://console.twilio.com), then Phone Numbers > Manage > Buy a number. Pick any UK or US number (trial credit covers it).

**Step 2: Expose your local server**

Download cloudflared (no account needed for quick tunnels) from [github.com/cloudflare/cloudflared/releases](https://github.com/cloudflare/cloudflared/releases), then:

```bash
# Terminal 1: start the API
uvicorn backend:app --port 5005

# Terminal 2: open the public tunnel
cloudflared tunnel --url http://localhost:5005
```

cloudflared prints a URL like `https://abc-def-ghi.trycloudflare.com`. Copy it.

**Step 3: Set the Twilio webhook**

In the Twilio console, open your phone number's configuration. Under "Voice and Fax", set "A call comes in" to:
- Webhook: `https://abc-def-ghi.trycloudflare.com/twilio/voice`
- HTTP POST

Save the config.

**Step 4: Call the number**

Call your Twilio number from any phone. The system answers, greets the caller, and handles the full booking conversation. Each call gets its own session (Twilio's CallSid is the session ID, so state persists across turns in the same call).

Twilio free trial can only call back your verified number. Buy credit (~$20) to remove that restriction.

**To use the real fine-tuned model:** start the llama.cpp server first (see CPU inference section), then change `mode="mock"` to `mode="llama3"` in `twilio_handler.py`.

## Piper TTS setup

Create a `piper/` folder in the project root and put these two files in it:

1. Piper binary: download from [github.com/rhasspy/piper/releases](https://github.com/rhasspy/piper/releases) - pick the Windows or Linux build
2. Voice model: download `en_US-lessac-medium.onnx` and its `.json` config from [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)