# IGP Wiki Structure — Blackboard Pages
# Copy each section below as a separate Blackboard wiki page.
# Each person logs into Blackboard and edits their own section.
# Blackboard tracks who wrote what automatically.

---

## PAGE 1: Project Overview
**Who writes this:** Whole team together (anyone can start, others add)

### What We Built
A real-time voice assistant for small and medium businesses (SMEs).
When a customer calls, the system:
1. Transcribes speech using Faster-Whisper (STT)
2. Extracts entities (date, time, service) using spaCy
3. Sends the utterance to a fine-tuned LLM (Phi-3 mini or Llama 3.2 3B)
4. The LLM returns a structured JSON action (book, cancel, check availability)
5. A spoken confirmation is generated (TTS via Piper)

### Team Members and Roles
- Karan: STT pipeline, LLM integration, QLoRA fine-tuning, evaluation, backend API
- [Name]: Website frontend, UI
- [Name]: Website frontend, integration
- [Name]: [their contribution]
- [Name]: [their contribution]

### Tools and Technologies
Python, Faster-Whisper, spaCy, Phi-3 mini, Llama 3.2 3B, QLoRA, PEFT, FastAPI,
Kaggle T4 GPU, HuggingFace, Ollama, PostgreSQL (planned), Piper TTS

### Repository Links
- LLM + Backend: [your GitHub link]
- STT Pipeline: [GitLab link]

---

## PAGE 2: Literature Review Summary
**Who writes this:** Karan (you collected and summarised the articles)

### Research Approach
Started with Copilot to find references but found hallucinated links and invalid DOIs.
Switched to manual search on arXiv, Elsevier, and ACL Anthology.
Each paper was downloaded as PDF and summarised using a fixed template:
- Which pipeline stage it relates to
- Research gap addressed
- Key findings
- Relevance to this project

### Key Papers and What They Contributed
[Paste your 12+ article summaries here, or link to the Word doc]

### Model Selection Journey
We initially considered Llama 70B, then Llama 8B, then settled on Llama 3.2 3B
and Phi-3 mini 3.8B. The larger models required 40GB+ VRAM and could not be
fine-tuned on available hardware (Kaggle T4 = 16GB). The 3B models fit within
16GB using 4-bit NF4 quantisation (QLoRA).

---

## PAGE 3: Technical Implementation — STT Pipeline
**Who writes this:** Karan

### What It Does
Converts incoming audio to text using Faster-Whisper (a fast CTranslate2
implementation of OpenAI Whisper). The output feeds directly into the LLM pipeline.

### Key Decisions
- Chose Faster-Whisper over base Whisper for speed (3-4x faster on same hardware)
- Used `base` model size for balance of speed and accuracy
- WER evaluation done on LibriSpeech dataset

### Issues Encountered
[Paste from your Issues and Fixes doc — section 1]

---

## PAGE 4: Technical Implementation — LLM Pipeline
**Who writes this:** Karan

### Architecture
```
utterance -> spaCy entity extractor -> system prompt builder -> LLM -> JSON validator -> TTS string
```

### Models Used
- Phi-3 mini 3.8B (Microsoft) — chat template uses <|system|>, <|user|>, <|assistant|>
- Llama 3.2 3B (Meta) — chat template uses <|begin_of_text|>, header tokens

### Fine-tuning Method: QLoRA
- Base models loaded in 4-bit NF4 quantisation (bitsandbytes)
- LoRA adapters injected into attention and MLP projection layers
- Only 0.23% of parameters trained (approx 9 million out of 3.8 billion)
- Trained on Kaggle T4 GPU (free tier, 16GB VRAM)
- Training dataset: 600 synthetic samples generated from templates, 480 used for training

### Training Results
| Model | Epochs | Final Train Loss | Final Eval Loss |
|---|---|---|---|
| Phi-3 mini | 3 | 0.028 | 0.045 |
| Llama 3.2 3B | 3 | 0.028 | 0.034 |

Loss dropped from ~0.5 at step 10 to ~0.03 by step 90 for both models.

### Evaluation Results
| Condition | JSON Valid | Action Accuracy | Exact Match | Latency p50 |
|---|---|---|---|---|
| Vanilla Phi-3 | 90% | 0.4% | 0% | 4410ms |
| Fine-tuned Phi-3 | 100% | 98.1% | 70.6% | 3556ms |
| Vanilla Llama 3 | 95% | 0% | 0% | 2791ms |
| Fine-tuned Llama 3 | 100% | 99.8% | 70.4% | 3703ms |

Fine-tuning with 480 samples improved action accuracy from near-zero to 98-99%.

---

## PAGE 5: Technical Implementation — Backend API
**Who writes this:** Karan

### What It Does
FastAPI server that receives customer utterances and returns spoken confirmations.
The website team connects their frontend to this.

### Endpoints
- POST /turn — main endpoint, takes utterance, returns action + spoken text
- GET /availability — check available slots
- POST /book — book an appointment
- POST /cancel — cancel an appointment
- GET /health — server health check
- GET /docs — Swagger UI (auto-generated)

### How to Run
```
uvicorn backend:app --reload --port 5005
```
Then open http://localhost:5005/docs to see all endpoints.

---

## PAGE 6: Website and Frontend
**Who writes this:** Peter and Goodnews (their section entirely)

[They write what they built, how it connects to the backend, what framework they used, screenshots]

---

## PAGE 7: Project Management
**Who writes this:** Whole team, anyone can contribute

### Timeline
- February/March 2026: Project kickoff, literature review, Trello used for task tracking
- March 2026: Two team meetings (dates), proposal drafted in Notion
- April–May 2026: STT pipeline developed, GitLab repo set up
- June 2026: Switched to Microsoft Teams for communication, LLM pipeline built,
  fine-tuning completed on Kaggle, evaluation done
- July 2026: Website integration, demo preparation, wiki completion

### Tools Used for Collaboration
- Trello: Used for 4-5 weeks for early sprint planning
- Notion: Proposal writing, article summaries, meeting notes
- Microsoft Teams: Daily communication from June onwards (recorded sessions available)
- GitLab: STT pipeline repo
- GitHub: LLM pipeline repo (private)
- UWE OneDrive: Large files (LibriSpeech audio, PDF articles)

### Meeting Notes
**March [date] — Meeting 1**
Attendees: [names]
Discussed: [what was decided]

**March [date] — Meeting 2**
Attendees: [names]
Discussed: [what was decided]

**June onwards:** Communication moved to Microsoft Teams.
Key decisions documented in Teams channel (screenshots below).
[Add Teams screenshots of key technical decisions here]

---

## PAGE 8: Issues and Fixes Log
**Who writes this:** Karan (you have this drafted already)

[Paste your Issues and Fixes doc here — the one you wrote above]

---

## PAGE 9: Individual Reflections
**Who writes this:** Each person writes their own entry — DO NOT write each other's

### Karan's Reflection
[Write this yourself — what you learned, what was hard, what you would do differently]

### [Team member name]'s Reflection
[They write their own]

### [Team member name]'s Reflection
[They write their own]

---

## PAGE 10: Results and Demo Evidence
**Who writes this:** Karan + website team for their screenshots

### Screenshots to Include
- Training loss curve from Kaggle (screenshot from notebook output)
- Evaluation results table (from eval_summary.json)
- Mock demo running in terminal (screenshot of --mock output)
- Backend Swagger UI screenshot (http://localhost:5005/docs)
- Website screenshot showing a booking being made
- Any Teams messages showing key decisions

### What the Demo Shows
The pipeline running in mock mode demonstrates the full flow:
utterance -> entity extraction -> action JSON -> spoken confirmation
at 4-6ms latency per turn.

The fine-tuned model (evaluated on Kaggle) achieves 99.8% action accuracy
compared to 0% for the vanilla model, demonstrating the value of QLoRA fine-tuning
on domain-specific data.
