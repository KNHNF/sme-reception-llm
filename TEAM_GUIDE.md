g# SME Voice Assistant - Team Guide
UWE Bristol MSc Data Science IGP | Group Project

This guide is for everyone on the team. Read it before the viva.
It explains what we built, how it works, and what questions you might get asked.

---

## What We Built

An AI-powered phone assistant for small businesses.
When a customer calls, instead of waiting on hold or pressing keys on a menu,
they speak naturally. The system understands what they want and handles it.

The assistant can:
- Book appointments ("I want a consultation on Monday at 2pm")
- Cancel appointments ("Cancel my appointment on Wednesday")
- Check availability ("Do you have anything free this Thursday?")
- Ask for clarification when something is missing ("What time would you like?")
- Redirect questions it cannot handle ("What are your opening hours?" is out of scope)

---

## How the System Works - Simple Version

```
Customer speaks
     |
     v
Speech-to-Text (Faster-Whisper)  <-- converts audio to text
     |
     v
Entity Extractor (spaCy)  <-- finds dates, times, service type in the text
     |
     v
Language Model (Phi-3 or Llama 3)  <-- understands intent, outputs a structured response
     |
     v
Validator (Pydantic)  <-- checks the response is in the correct format
     |
     v
Spoken Confirmation (Piper TTS)  <-- reads the response back to the customer
```

---

## The Team and What Each Person Did

**Karan**
STT pipeline, spaCy entity extraction, LLM integration, QLoRA fine-tuning on Kaggle,
evaluation (comparing fine-tuned vs vanilla models), FastAPI backend, dataset generation,
literature review coordination.

**Theo**
Literature review cleanup and formatting. [Theo adds his own contributions here]

**[Name]**
[Add your contributions here when you push to GitLab]

**[Name]**
[Add your contributions here when you push to GitLab]

**[Name]**
[Add your contributions here when you push to GitLab]

---

## The Repositories

| Repo | What is in it | Where |
|---|---|---|
| LLM + Backend | Language model, training scripts, evaluation, API | GitHub (private, KNHNF) |
| STT Pipeline | Faster-Whisper setup, WER evaluation, audio processing | GitLab (team repo) |
| Website | Frontend, UI, integration with backend | GitLab (team repo) |

Large files (audio datasets, model weights) are stored on UWE OneDrive, not in Git.

---

## How to Run the Demo

You need Python 3.11 installed. Open PowerShell.

**Step 1 - go to the right folder:**
```
cd E:\Coding\public-projects\ai-reception-sme
```

**Step 2 - run the pipeline (no GPU needed, no download needed):**
```
C:\Users\USER\AppData\Local\Programs\Python\Python311\python.exe src/inference.py --mock
```

You should see the system processing test sentences and returning actions with spoken confirmations.

**Step 3 - start the backend server:**
```
C:\Users\USER\AppData\Local\Programs\Python\Python311\python.exe -m uvicorn backend:app --reload --port 5005
```

Open http://localhost:5005/docs in your browser. You will see a list of all API endpoints
you can test directly from the browser.

**If something breaks:**
- Missing package error: run `pip install [package name]`
- Port already in use: change 5005 to 5006
- spaCy error: run `python -m spacy download en_core_web_sm`

---

## What the Website Team Needs to Know

The backend runs at http://localhost:5005

To send a customer utterance and get a response, send a POST request to /turn:

```json
Request body:
{
  "utterance": "I want to book a consultation for tomorrow at 3pm",
  "session_id": "caller-001"
}

Response:
{
  "action": {"action": "book_appointment", "date": "2026-06-22", "time": "15:00", "service": "consultation"},
  "validated": true,
  "spoken": "I have booked a consultation for Monday, 22 June at 3:00 PM.",
  "latency_ms": 6.2
}
```

The `spoken` field is the text that gets read back to the customer.
The `session_id` keeps track of multi-turn conversations (when the system asks a follow-up question).

All endpoints are documented at http://localhost:5005/docs - you can test them live from the browser.

---

## Literature Review

The team collectively reviewed articles from arXiv, Elsevier, and ACL Anthology.
Each paper was summarised using a fixed template covering:
- Which part of the pipeline it relates to
- Key findings
- Relevance to this project
- Research gap it addresses

Copilot was used initially to find references but many links were hallucinated or broken.
We switched to downloading PDFs manually from trusted academic sources and summarising each one.
Karan coordinated the final file. Theo cleaned up the formatting.

The full literature review is in the Word document in this repo.

---

## Viva Questions You Should Be Able to Answer

These are the kinds of questions tutors ask. Read the answers and understand them
well enough to say them in your own words.

**"What problem does this project solve?"**
Small businesses lose customers when no one answers the phone. Our system handles
appointment calls automatically using AI, without needing a human receptionist.

**"Why not just use ChatGPT or Gemini?"**
Cost, privacy, and latency. A commercial API costs money per call and sends customer
data to a third-party server. A small fine-tuned local model is cheaper, faster,
and keeps data on-site.

**"What is fine-tuning and why did you do it?"**
The base model (Phi-3 or Llama 3) knows how to speak English but has no idea
what our appointment booking system looks like. Fine-tuning teaches it our specific
task by showing it 480 examples of correct input-output pairs.

**"What is QLoRA?"**
A technique to fine-tune large models on limited hardware. Instead of training all
3.8 billion parameters (which needs expensive GPUs), we compress the model to 4-bit
and only train small adapter matrices. Only 0.23% of parameters are trained.

**"What were your results?"**
The vanilla (unmodified) models scored near 0% on action accuracy because they
do not know our JSON schema. After fine-tuning, Phi-3 reached 98.1% and Llama 3
reached 99.8% action accuracy on the test set.

**"What would you do differently?"**
Connect a real phone line (Twilio), integrate Google Calendar for live booking,
add voice activity detection to handle background noise, use constrained decoding
to guarantee valid JSON output every time.

**"What did [your name] specifically contribute?"**
Answer this with your actual contribution. Be specific and honest.

---

## Project Timeline

| Period | What Happened |
|---|---|
| Feb/Mar 2026 | Kickoff, team formed, literature review started, Trello used for task tracking |
| March 2026 | Two team meetings, proposal drafted in Notion, GitLab account issues resolved |
| April/May 2026 | STT pipeline built and tested, WER evaluation, LibriSpeech dataset |
| June 2026 | Moved to Teams, LLM pipeline built, fine-tuning on Kaggle, evaluation done |
| July 2026 | Website integration, wiki, viva preparation |

---

## Known Limitations - Mention These in the Viva

Being honest about limitations shows critical thinking, which markers reward.

- The live demo uses mock mode (rule-based), not the trained model, because the trained model needs a GPU to run
- Training data is synthetic (generated by code), not from real customer calls
- No real phone line - uses microphone input via a web interface
- Text-to-speech (Piper) is integrated but not demonstrated live in this version
- Google Calendar not connected - appointments stored in memory only
- GitLab collaboration had some early issues with branch management and account setup
