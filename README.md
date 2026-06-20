# sme-reception-llm

LLM pipeline for a real-time voice assistant handling inbound calls for small businesses.
UWE Bristol MSc Data Science — IGP Group Project.

## What it does

Converts a caller utterance into a structured booking action:

```
mic -> Faster-Whisper STT -> spaCy entity extraction -> fine-tuned LLM -> JSON action -> Piper TTS
```

Supports: book appointment, cancel appointment, check availability, clarify (missing fields), out of scope.

## Models

Two models fine-tuned with QLoRA on a 480-sample synthetic dataset:

| Model | Action Accuracy | Exact Match | Latency p50 |
|---|---|---|---|
| Phi-3 mini 3.8B (vanilla) | 0.4% | 0% | 4410ms |
| Phi-3 mini 3.8B (fine-tuned) | 98.1% | 70.6% | 3556ms |
| Llama 3.2 3B (vanilla) | 0% | 0% | 2791ms |
| Llama 3.2 3B (fine-tuned) | 99.8% | 70.4% | 3703ms |

Adapters trained on Kaggle T4 GPU (free tier). Mock mode runs on CPU with no download.

## Quick start

```
cd files_IGP
pip install fastapi uvicorn pydantic spacy
python -m spacy download en_core_web_sm

# Text demo (no mic, no GPU)
python demo.py --text --no-tts

# Backend server
uvicorn backend:app --reload --port 5005
# Then open http://localhost:5005/docs
```

## Docker (mock backend)

```
cd files_IGP
docker build -t sme-backend .
docker run -p 5005:5005 sme-backend
```

Image size: ~500MB. No GPU needed.

## Project structure

```
files_IGP/
  src/
    inference.py        pipeline core (mock / vanilla / finetuned / ollama)
    entity_extractor.py spaCy NER for date, time, service
    session_manager.py  multi-turn conversation state
    sme_action_schema.py Pydantic models + JSON schema + TTS templates
    stt.py              Faster-Whisper mic recording + transcription
    tts.py              Piper TTS wrapper
  scripts/
    generate_dataset.py synthetic training data (600 samples)
    train_qlora.py      QLoRA fine-tuning (local)
    kaggle_train.ipynb  Phi-3 training notebook (Kaggle)
    kaggle_train_llama.ipynb Llama 3 training notebook (Kaggle)
    kaggle_eval.ipynb   4-condition evaluation notebook (Kaggle)
    evaluate_model.py   local evaluation script
  backend.py            FastAPI server with mock calendar
  demo.py               end-to-end voice demo script
  docs/
    pipeline_flowchart.html      interactive pipeline diagram
    pipeline_figure_report.html  report-style diagram
```

## Related repos

- STT evaluation: [whisper-stt-eval](https://github.com/KNHNF/whisper-stt-eval)

## Stack

Python 3.11, Faster-Whisper, spaCy, Phi-3 mini, Llama 3.2 3B, QLoRA/PEFT, FastAPI, Pydantic v2, Piper TTS, Docker
