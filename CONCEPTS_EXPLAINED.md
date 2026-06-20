# Concepts Explained — No Prior Knowledge Required
SME Voice Assistant IGP | UWE Bristol

Read this if you want to understand what the technical parts of this project
actually mean. Written in plain English.

---

## Large Language Models (LLMs)

An LLM is a program trained on huge amounts of text (books, websites, articles)
to predict what word comes next. By doing this billions of times during training,
it learns grammar, facts, reasoning, and how to have conversations.

Examples: GPT-4 (OpenAI), Gemini (Google), Phi-3 (Microsoft), Llama (Meta).

The models we used:
- **Phi-3 mini** — 3.8 billion parameters, made by Microsoft, small enough to run on limited hardware
- **Llama 3.2 3B** — 3 billion parameters, made by Meta, similar size

A "parameter" is a number the model uses to make decisions. More parameters generally
means smarter but slower and more expensive to run.

---

## Fine-Tuning

A general LLM knows how to speak English but does not know your specific task.
If you ask a vanilla Phi-3 "book me a consultation" it might write a paragraph
of helpful text. But we need it to output a specific JSON object like:
`{"action": "book_appointment", "date": "2026-06-23", "time": "14:00"}`

Fine-tuning fixes this. You give the model hundreds of examples of correct
input-output pairs. It adjusts its internal weights to match the pattern.
After fine-tuning, it knows exactly what format to produce.

We showed it 480 examples. Before training, both models scored near 0% on our task.
After training, both scored 98-99%.

---

## QLoRA — Why We Could Not Just Fine-Tune Normally

Full fine-tuning of a 3.8B model requires updating all 3.8 billion parameters.
This needs roughly 28GB of GPU memory. A free Kaggle T4 GPU has 16GB.

QLoRA solves this in two ways:

**Step 1 — Quantisation:**
Normally each parameter is stored as a 32-bit or 16-bit number.
QLoRA compresses them to 4-bit. This cuts memory usage by roughly 4x.
The base model is frozen — its weights do not change during training.

**Step 2 — LoRA adapters:**
Instead of changing the base model, LoRA adds small extra matrices alongside
the model's existing layers. Only these small matrices are trained.
They represent roughly 0.23% of all parameters — about 9 million out of 3.8 billion.

After training, the adapter weights (about 35MB) are saved separately.
At inference time, you load the base model and snap the adapter on top.

**Who invented it:**
LoRA was published by Microsoft Research in 2021.
QLoRA was published by Tim Dettmers et al. at University of Washington in 2023.

---

## spaCy and Entity Extraction

spaCy is a Python library for natural language processing.
We use it to pull specific pieces of information out of a customer's message
before sending it to the LLM.

Example:
- Input: "I need a follow-up appointment next Tuesday at half ten"
- spaCy finds: DATE = "next Tuesday", TIME = "half ten"
- We convert these: date = "2026-06-24", time = "10:30"
- We pass the resolved date and time to the LLM as extra context

This reduces the LLM's workload. Instead of figuring out what "next Tuesday" means,
the LLM just sees "2026-06-24" and can focus on the booking logic.

---

## FastAPI

FastAPI is a Python web framework for building APIs.
An API (Application Programming Interface) is a way for two programs to talk to each other.

In this project, the website (frontend) sends customer utterances to the FastAPI backend.
The backend processes them through the pipeline and sends back the action and spoken text.

Why FastAPI:
- Fast to write
- Automatically generates documentation at /docs
- Validates request and response formats automatically

---

## Pydantic

Pydantic is a Python library for data validation.
After the LLM outputs JSON, Pydantic checks whether it matches our expected schema.

If the LLM says `{"action": "book_appointement"}` (typo), Pydantic catches it.
If the LLM forgets to include a required field, Pydantic catches it.
This prevents bad data from reaching the booking system.

---

## Speech-to-Text (STT) — Faster-Whisper

Whisper is a speech recognition model made by OpenAI.
Faster-Whisper is a reimplementation that runs 3-4 times faster using CTranslate2.

It takes an audio file or microphone stream and outputs a text transcript.
That transcript is what gets sent to the entity extractor and then the LLM.

**WER (Word Error Rate):** How we measure STT quality.
If the reference is "book a consultation" and Whisper outputs "look a consultation",
WER = 1 wrong word out of 3 = 33%. Lower is better. 0% means perfect.

---

## Training Loss — What Those Numbers Mean

During training, after each batch of examples, we calculate how wrong the model was.
This is called the loss. A high loss means the model is making big mistakes.
A low loss means it is getting the answers right.

We saw:
- Step 10 (very early): loss around 0.5 — model is mostly wrong
- Step 90 (end of training): loss around 0.028 — model is almost always right

The fact that loss drops smoothly from 0.5 to 0.028 confirms the training worked correctly.

---

## Evaluation Metrics — What We Measured

**JSON validity rate:** What percentage of outputs were valid, parseable JSON.
Vanilla models sometimes output English prose instead of JSON. 100% means always JSON.

**Action accuracy:** What percentage of outputs had the correct action type
(book vs cancel vs clarify etc). This is the most important metric.

**Exact match rate:** What percentage of outputs had every single field exactly right.
This is the strictest metric. Even a date format difference counts as wrong.

**Latency:** How long each inference call took in milliseconds.
p50 = median (typical case). p95 = 95th percentile (worst normal case).

---

## Synthetic Dataset

We did not have real customer call transcripts (privacy reasons, no phone line).
Instead we wrote a script (`generate_dataset.py`) that generates realistic
example utterances using templates.

Example template for booking:
"I'd like to book {service} on {weekday} at {time}"

The script fills in random values: service=consultation, weekday=Monday, time=2pm.
It generates 600 examples covering all 5 intent types.

This approach is common in NLP research when real data is unavailable or private.
It is worth mentioning in the viva as a deliberate methodological choice.

---

## Kaggle

Kaggle is a platform for data science and ML. It provides free GPU access.
The free tier gives you access to NVIDIA T4 GPUs (16GB VRAM) for 30 hours per week.

We used Kaggle to run training and evaluation because the T4 GPU is enough
for QLoRA fine-tuning of 3-4B parameter models, and it is free.

Training time per model: approximately 40-50 minutes.
Evaluation of all 4 conditions: approximately 90 minutes.

---

## Mock Mode vs Real Model

**Mock mode** (`--mock`): No AI involved. A simple Python script checks for keywords
like "cancel" or "book" and returns a hardcoded JSON response. Used for demos and testing
the pipeline structure. Latency is 4-6ms.

**Vanilla mode** (`--vanilla`): Loads the actual base model with no fine-tuning.
Gives near-random outputs on our task. Used as the baseline for comparison.

**Fine-tuned mode** (`--adapter`): Loads the base model plus the trained adapter.
Gives 98-99% action accuracy. Requires a GPU to run at reasonable speed.

**Ollama mode** (`--model ollama`): Calls a locally running Ollama server.
Uses the vanilla model via Ollama API. Convenient but gives poor task accuracy.
