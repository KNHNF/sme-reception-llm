# Concepts Explained Part 2 — More Detail, No Prior Knowledge Required
SME Voice Assistant IGP | UWE Bristol

This continues from CONCEPTS_EXPLAINED.md. Read that one first.

---

## Tokens — What the Model Actually Reads

An LLM does not read letters or words. It reads tokens.
A token is a chunk of text — usually 3-4 characters long.

Examples of how text is split into tokens:
- "appointment" = ["app", "oint", "ment"] = 3 tokens
- "2026-06-23" = ["2026", "-", "06", "-", "23"] = 5 tokens
- "I want to book" = ["I", " want", " to", " book"] = 4 tokens

Why this matters:
- Models have a context limit measured in tokens, not words (Phi-3 mini = 4,096 tokens)
- Generating more tokens = slower output
- We cap max_new_tokens=40 per response because our JSON outputs are short

The word "tokens" also refers to authentication tokens (like HF_TOKEN for Hugging Face).
These are completely different things — just coincidental naming.

---

## Chat Templates — Why the Prompt Format Matters

LLMs trained as chat assistants expect a specific format to know who is speaking.
If you just send raw text, the model does not know where the system instructions end
and the user message begins.

Each model family uses a different format.

Phi-3 mini format:
```
<|system|>
[system instructions here]<|end|>
<|user|>
[user message here]<|end|>
<|assistant|>
[model writes here]
```

Llama 3.2 format:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

[system instructions here]<|eot_id|>
<|start_header_id|>user<|end_header_id|>

[user message here]<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

[model writes here]
```

If you use the wrong format for a model, accuracy drops dramatically because
the model gets confused about which role is speaking.

This is why we have separate `fmt_phi3()` and `fmt_llama3()` functions in the code.

---

## EOS Token — How the Model Knows When to Stop

EOS stands for End of Sequence. It is a special token that tells the model to stop generating.

Without an EOS token, the model would keep writing forever (or until max_new_tokens is reached).

Each model family has a different EOS token:
- Phi-3: `<|end|>`
- Llama 3: `<|eot_id|>` (end of turn)

We pass this token ID to `model.generate()` so generation stops as soon as the model
outputs it. This is what keeps latency low — we stop immediately after the JSON ends.

---

## Epochs and Batch Size

**Epoch:** One full pass through the entire training dataset.

If you have 480 training examples and train for 3 epochs, the model sees each
example 3 times (1,440 total training steps with batch size 1).

More epochs generally means better accuracy, up to a point — too many epochs causes
overfitting (the model memorises the training examples instead of learning the pattern).

**Batch size:** How many examples are processed at once before updating the model weights.

We used batch_size=1, meaning the model sees one example at a time.
This is slow but uses the least memory — important when running on a 16GB GPU.

**Gradient accumulation:** A trick to simulate a larger batch size without extra memory.
With grad_accum=16 and batch_size=1, the model processes 16 examples in sequence
and only updates weights after all 16 are done. The result is similar to batch_size=16
but memory usage stays low.

---

## Loss Masking — Training on Outputs Only

During fine-tuning, we do not want the model to learn from its own input.
We only want it to learn from the expected output (the JSON).

Loss masking hides the prompt tokens from the loss calculation.
Only the assistant's JSON tokens contribute to the loss.

If we did not mask the prompt:
- The model would spend most of its learning effort on the system prompt, which never changes
- It would learn to copy fixed patterns rather than understand the mapping from input to output

The DataCollatorForCompletionOnlyLM class handles this automatically.
You tell it the response template (e.g. `<|assistant|>\n`) and it masks everything before it.

---

## What "Parameters" Are in More Detail

A neural network is a large set of numbers (parameters/weights) arranged in layers.
During training, these numbers are adjusted so the network makes fewer mistakes.

For a 3.8 billion parameter model:
- There are 3,800,000,000 individual numbers
- They are stored as matrices (tables of numbers)
- At inference time, the input (tokens) flows through each layer multiplied by these matrices
- The output of the final layer is a probability distribution over all possible next tokens
- The model picks the highest probability token and adds it to the output

With QLoRA, we freeze all 3.8B original parameters and only train the small LoRA matrices
(about 9 million numbers). The original knowledge stays intact; the adapter changes
how the model responds in specific situations.

---

## LoRA in More Detail — What "Rank" Means

LoRA injects small matrices into the model's attention layers.
The key idea is "low rank" -- the adapter matrices are small.

If a model layer has a weight matrix of size 4096 x 4096 (about 16M parameters),
a LoRA adapter with rank 16 adds two small matrices: 4096x16 and 16x4096.
That is only 2 x 65,536 = 131,072 parameters instead of 16 million.

Rank controls the adapter size:
- rank=8: very small, trains fast, less capacity
- rank=16: good balance (what we used)
- rank=64: larger, more capacity, slower

Alpha controls how strongly the adapter output is scaled before being added to the base output.
We used alpha=32, which is 2x the rank -- a common default.

---

## NF4 Quantisation — What 4-bit Actually Means

Standard model weights are stored as 16-bit floating point numbers (bfloat16).
4-bit NF4 (Normal Float 4) compresses each weight into just 4 bits.

4 bits can represent 16 distinct values (2^4 = 16).
NF4 spaces those 16 values unevenly, following a normal distribution,
because model weights tend to cluster near zero.

Result: memory drops from roughly 2 bytes per weight to 0.5 bytes per weight.
A 3.8B model that would need 7.6GB in bfloat16 now fits in about 2GB.

The base model weights stay at 4-bit during training.
The LoRA adapter weights are trained in bfloat16 (16-bit) for precision.
This combination is what makes QLoRA work on consumer-grade GPUs.

---

## JSON Schema — What It Is and Why We Use One

A JSON schema is a formal definition of what valid JSON must look like.

Example -- our book_appointment schema says:
- "action" field must be the string "book_appointment"
- "date" field must match pattern YYYY-MM-DD
- "time" field must match pattern HH:MM
- "service" field must be one of: general, consultation, follow_up
- All four fields are required

Without a schema, the LLM might output "June 23rd" as a date instead of "2026-06-23",
or "half ten" as a time instead of "10:30", or omit a required field entirely.

Pydantic reads the schema and checks every output before it reaches the booking system.
If the output fails validation, the system asks the caller to repeat.

We also export the schema (via `get_json_schema()`) so it can be passed to a constrained
decoding library that physically prevents the model from generating invalid tokens.

---

## REST API and HTTP Methods

REST (Representational State Transfer) is the standard way for web applications to communicate.

HTTP methods:
- GET: retrieve something (no side effects)
- POST: send data, trigger an action

Our backend endpoints:
- POST /turn: send a customer utterance, get an action + spoken response
- GET /health: check if the server is running
- GET /docs: open the Swagger documentation page

When the website team sends a POST request to /turn, they include a JSON body
with the customer's utterance and a session ID. The server returns a JSON response
with the action and the spoken confirmation.

This is the same pattern used by every major web API (Twitter, Stripe, Google Maps, etc.).

---

## RAG vs Fine-Tuning — What is the Difference?

Both are ways to make a general LLM more useful for a specific task.

**RAG (Retrieval Augmented Generation):**
- You store knowledge in a database (e.g. your appointment schedule)
- At inference time, you search the database for relevant information
- You inject that information into the prompt before sending it to the model
- The model is not changed — it stays vanilla
- Good for: knowledge bases, FAQs, large amounts of changing information

**Fine-Tuning:**
- You show the model hundreds of examples of correct input-output pairs
- The model's weights are adjusted to match the pattern
- The model itself changes
- Good for: teaching the model a specific output format or behaviour

We used fine-tuning because our problem is about output format (JSON schema),
not about retrieving external knowledge. The model needed to learn the task structure,
not look up facts.

For a real production system, you would combine both:
fine-tune for format, RAG for pulling real appointment slots from a calendar.

---

## Session Management — Multi-Turn Conversations

A single customer call may require multiple exchanges:

Turn 1 — Customer: "I want to book a consultation"
Turn 1 — System: "What date would you like?"
Turn 2 — Customer: "Next Thursday"
Turn 2 — System: "And what time?"
Turn 3 — Customer: "Half two in the afternoon"
Turn 3 — System: "Booked for Thursday 25 June at 14:30"

To handle this, we need to remember what was said before.
The session manager keeps a partial_context dictionary per session_id.

When the model returns action=clarify (missing fields), the session stores:
- What fields are already known (e.g. service=consultation)
- What fields are still missing (e.g. date, time)

On the next turn, this context is injected into the prompt so the model sees:
"[Already known: {service: consultation}] [Still missing: date, time] Next Thursday"

Without session management, the caller would have to repeat everything on every turn.

---

## VAD — Voice Activity Detection

VAD (Voice Activity Detection) is a filter that detects when a person is actually speaking
versus when the line is silent or has background noise.

Without VAD, the STT system tries to transcribe everything including:
- Silence between sentences
- "Umm", "erm", breath sounds
- Background music or TV

With VAD, the system only transcribes when a human voice is detected.

Faster-Whisper supports VAD via the silero-vad library.
We decided not to implement this for the current version due to time constraints.
It would be a clear next step for a production system handling real phone calls.

---

## TTS — Text-to-Speech

TTS converts the system's text response into spoken audio so the caller hears a voice.

We planned to use Piper TTS, an open-source offline TTS system from the Home Assistant project.
Piper runs locally, requires no internet connection, and sounds natural.

The `spoken` field in every pipeline response is the text ready to be passed to Piper.
Piper would convert it to a WAV audio file or audio stream, which would be played to the caller.

TTS integration is listed as a known limitation in the current version -- the text is generated
correctly but Piper is not connected to the live call path.

---

## bitsandbytes — What It Actually Does

bitsandbytes is a Python library that adds 4-bit and 8-bit quantisation to PyTorch.

Under the hood, it replaces standard PyTorch Linear layers with custom layers
that store weights in 4-bit NF4 format and dequantise them on-the-fly during computation.

The compute still happens in bfloat16 (for accuracy), but the weights sit in memory as 4-bit.
The dequantisation step adds a small overhead but the memory saving is worth it.

Version matters: we pin bitsandbytes==0.45.5 on Kaggle because:
- 0.43.1 tries to import `triton.ops`, which does not exist in Kaggle's Triton installation
- 0.45.5 removed that dependency and works on Kaggle without errors

---

## PEFT — What the Library Does

PEFT (Parameter-Efficient Fine-Tuning) is a Hugging Face library that provides
different adapter methods including LoRA, IA3, and prompt tuning.

We use two PEFT functions:
1. `get_peft_model()` during training: wraps the base model and adds LoRA layers
2. `PeftModel.from_pretrained()` during inference: loads a trained adapter on top of a base model

The adapter saved after training is just the LoRA matrices (not the full model).
This is why our Phi-3 adapter is only 35MB while the base model is 7.6GB.
You always need the base model to load the adapter -- the adapter cannot run alone.

---

## Constrained Decoding — What It Is and Why We Skipped It

Constrained decoding forces the model to only generate tokens that match a valid JSON schema.

At each step, instead of picking the single highest-probability token,
the decoding algorithm checks which tokens would still result in valid JSON,
and only allows those.

This guarantees 100% valid JSON output, every time, with no post-processing needed.

Libraries that do this: outlines, lm-format-enforcer.

Why we did not implement it:
- It adds a dependency and integration complexity
- Our fine-tuned models already output 100% valid JSON in evaluation
- It is a sensible next step for a production deployment where reliability must be absolute

The schema code in `sme_action_schema.py` already exports the right JSON schema
for these libraries, so adding it later would be straightforward.

---

## Alpaca Format — What Our Training Data Looks Like

The training data follows the Alpaca format, named after a Stanford research project.
Each training example has three fields:

```json
{
  "instruction": "Appointment assistant. Output one JSON object only...",
  "input": "I want to book a consultation on Monday at 2pm",
  "output": "{\"action\": \"book_appointment\", \"date\": \"2026-06-23\", \"time\": \"14:00\", \"service\": \"consultation\"}"
}
```

- `instruction`: the system prompt (same for every example)
- `input`: the customer utterance (varies for each example)
- `output`: the correct JSON response (what the model must learn to produce)

The SFTTrainer (from trl) takes these fields, builds the chat-formatted prompt,
and applies loss masking so only the output tokens are trained on.

---

## Why 480 Samples Is Enough

480 is a small dataset by ML standards. Most production models are trained on millions of examples.

We can get away with 480 because:
1. The task is narrow (5 action types, fixed JSON schema)
2. The base model already knows English and JSON formatting
3. We are only teaching a new response format, not new knowledge
4. QLoRA only trains 9 million parameters (not 3.8 billion)

The training loss dropping from 0.5 to 0.028 in 90 steps confirms the model learned
the task well from a small number of examples. This is called few-shot fine-tuning.

---

## Inference vs Training

**Training:** Adjusting model weights using examples and a loss function. Slow, done once.

**Inference:** Using the trained model to process new inputs. Fast, done every call.

During training, gradients are computed and weights are updated.
This requires storing the full computation graph in memory, which is why training uses more VRAM.

During inference, gradients are not computed (no weight updates needed).
This is why we call `model.eval()` and use `torch.no_grad()` before generating outputs.
These two lines alone reduce VRAM usage significantly during inference.

---

## What Kaggle Is Actually Doing Under the Hood

When you run a committed notebook on Kaggle:
1. Kaggle spins up a virtual machine (a server in Google's data centre)
2. The VM has an NVIDIA T4 GPU attached (16GB VRAM)
3. Your notebook cells run top to bottom on that VM
4. Outputs are saved to `/kaggle/working/` on the VM's local disk
5. Kaggle's platform copies those outputs to their storage so you can download them

The VM is not your laptop. It runs independently of your browser.
That is why closing the browser does not stop a committed run.

When the run finishes, the VM is deleted. The outputs persist in Kaggle's storage.
You download the outputs through the Kaggle website.
