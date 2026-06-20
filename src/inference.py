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
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path when running as script
sys.path.insert(0, str(Path(__file__).parent))

from entity_extractor import extract, to_prompt_context
from sme_action_schema import ActionOutput, render_confirmation

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

    if model_family == "llama3":
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # Default: Phi-3 format
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_content}<|end|>\n"
        f"<|assistant|>\n"
    )


def parse_llm_output(text: str) -> Optional[dict]:
    """Extract JSON from LLM output, tolerating minor formatting noise."""
    text = text.strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


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
                 ollama_url: str = "http://localhost:11434"):

        self.mode         = mode
        self.model_family = model_family
        self.model        = None
        self.tokenizer    = None
        self.ollama_model = ollama_model
        self.ollama_url   = ollama_url

        model_id = MODEL_IDS.get(model_family, MODEL_IDS["phi3"])

        if mode == "mock":
            print("Pipeline running in MOCK mode -- no GPU needed.")
            return

        if mode == "ollama":
            print(f"Pipeline using Ollama: {ollama_model} at {ollama_url}")
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
            }
        """
        t0 = time.perf_counter()

        entities = extract(utterance)
        prompt   = build_prompt(utterance, entities, partial_context, self.model_family)

        if self.mode == "mock":
            raw_text = self._mock_output(utterance, entities)
        elif self.mode == "ollama":
            raw_text = self._ollama_generate(prompt)
        else:
            raw_text = self._hf_generate(prompt)

        t1 = time.perf_counter()

        parsed    = parse_llm_output(raw_text)
        validated = validate_action(parsed)
        spoken    = render_confirmation(validated) if validated else "I could not process that. Could you repeat?"

        return {
            "raw_output":  raw_text,
            "action":      parsed,
            "validated":   validated is not None,
            "spoken":      spoken,
            "latency_ms":  round((t1 - t0) * 1000, 2),
            "entities":    entities,
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

    def _mock_output(self, utterance: str, entities: dict) -> str:
        """
        Returns plausible JSON without a real model.
        Used for testing the pipeline structure before training is done.
        """
        u = utterance.lower()
        date = entities.get("date_resolved") or "2026-06-25"
        time = entities.get("time_resolved") or "10:00"
        svc  = entities.get("service") or "general"

        if any(w in u for w in ["cancel", "cancellation"]):
            return json.dumps({"action": "cancel_appointment", "date": date, "time": time})
        if any(w in u for w in ["available", "availability", "free", "open"]):
            return json.dumps({"action": "check_availability", "date": date, "service": svc})
        if any(w in u for w in ["book", "schedule", "appointment", "come in"]):
            if not entities.get("time_resolved"):
                return json.dumps({"action": "clarify", "missing_fields": ["time"]})
            return json.dumps({"action": "book_appointment", "date": date, "time": time, "service": svc})
        return json.dumps({"action": "out_of_scope"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mock",         action="store_true")
    p.add_argument("--vanilla",      action="store_true")
    p.add_argument("--adapter",      default=None)
    p.add_argument("--model",        default="phi3", choices=["phi3", "llama3", "ollama"])
    p.add_argument("--ollama_model", default="llama3.2")
    p.add_argument("--ollama_url",   default="http://localhost:11434")
    args = p.parse_args()

    if args.model == "ollama":
        pipeline = Pipeline(mode="ollama", model_family="llama3",
                            ollama_model=args.ollama_model, ollama_url=args.ollama_url)
    elif args.mock or (not args.vanilla and not args.adapter):
        pipeline = Pipeline(mode="mock", model_family=args.model)
    elif args.vanilla:
        pipeline = Pipeline(mode="vanilla", model_family=args.model)
    else:
        pipeline = Pipeline(mode="finetuned", model_family=args.model, adapter_path=args.adapter)

    test_utterances = [
        "I want to book a consultation for tomorrow at 3pm.",
        "Do you have any slots for a general appointment on Monday?",
        "I need to cancel my appointment on Wednesday at 10am.",
        "Book me in for a follow-up.",
        "What are your opening hours?",
    ]

    for utt in test_utterances:
        print(f"\nInput: {utt}")
        result = pipeline.run(utt)
        print(f"Action:    {result['action']}")
        print(f"Valid:     {result['validated']}")
        print(f"Spoken:    {result['spoken']}")
        print(f"Latency:   {result['latency_ms']} ms")
