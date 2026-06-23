"""
End-to-end voice demo
mic -> Faster-Whisper STT -> pipeline -> Piper TTS

This is the script to run for the viva demo and screen recording.

Usage:
    python demo.py                        voice input, mock pipeline, TTS output
    python demo.py --text                 type instead of speaking (no mic needed)
    python demo.py --no-tts               voice input but text-only output
    python demo.py --text --no-tts        fully text-based, no audio hardware needed
    python demo.py --model tiny           use Whisper tiny (faster, default)
    python demo.py --model small          use Whisper small (more accurate)

Requirements:
    pip install faster-whisper sounddevice soundfile
    (Piper optional -- see src/tts.py for setup)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import Pipeline
from src.tts import TTS


BANNER = """
=============================================================
  SME Voice Assistant -- Live Demo
  UWE Bristol MSc Data Science IGP
=============================================================
  Pipeline: mic -> Whisper STT -> spaCy -> LLM -> Piper TTS
  Mode:     MOCK (rule-based, no GPU needed)
  Commands: 'quit' or Ctrl+C to exit
=============================================================
"""

TEST_UTTERANCES = [
    "I'd like to book a consultation for next Monday at 2pm",
    "Do you have any slots available on Thursday for a general appointment?",
    "I need to cancel my appointment on Wednesday at 10am",
    "Book me in for a follow-up",
    "What are your opening hours?",
]


def run_turn(pipeline: Pipeline, tts, utterance: str,
             session_id: str, recorder=None) -> bool:
    """Run one pipeline turn. Returns True if the call should end."""
    print(f"\n  Input:   {utterance!r}")

    result = pipeline.run(utterance, session_id=session_id)

    name_tag = f"  Caller:  {result['caller_name']}\n" if result.get("caller_name") else ""
    print(f"{name_tag}  Action:  {result['action']}")
    print(f"  Spoken:  {result['spoken']}")
    print(f"  Latency: {result['latency_ms']}ms")

    audio = tts.speak(result["spoken"])
    if recorder and audio is not None:
        recorder.save(audio, result["spoken"])

    return result.get("end_call", False)


def voice_loop(pipeline: Pipeline, tts, whisper_model: str, recorder=None) -> None:
    from src.stt import STT
    stt = STT(model_size=whisper_model)
    session_id = "demo-voice"

    print(BANNER)
    print("  Speak after the prompt. Say 'quit' to exit.\n")

    while True:
        try:
            input("  [Press Enter to speak]")
        except (KeyboardInterrupt, EOFError):
            break

        utterance = stt.listen(duration=5.0, prompt=True)
        if not utterance:
            print("  [nothing heard, try again]")
            continue
        if "quit" in utterance.lower():
            break

        if run_turn(pipeline, tts, utterance, session_id, recorder):
            print("\n  [Call ended by caller]")
            break

    print("\n[Demo ended]")


def text_loop(pipeline: Pipeline, tts, recorder=None) -> None:
    session_id = "demo-text"

    print(BANNER)
    print("  Type a customer utterance and press Enter.\n")
    print("  Demo utterances to try:")
    for i, u in enumerate(TEST_UTTERANCES, 1):
        print(f"    {i}. {u}")
    print()

    while True:
        try:
            raw = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        if raw.isdigit() and 1 <= int(raw) <= len(TEST_UTTERANCES):
            utterance = TEST_UTTERANCES[int(raw) - 1]
            print(f"  [using preset: {utterance!r}]")
        else:
            utterance = raw

        if run_turn(pipeline, tts, utterance, session_id, recorder):
            print("\n  [Call ended by caller]")
            break

    print("\n[Demo ended]")


class _SilentTTS:
    def speak(self, text: str):
        return None


class _Recorder:
    """Saves each TTS turn to a timestamped WAV in recordings/."""
    def __init__(self):
        import datetime
        self.out_dir = Path(__file__).parent / "recordings"
        self.out_dir.mkdir(exist_ok=True)
        self.session_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turn = 0

    def save(self, audio_data, spoken_text: str):
        import soundfile as sf
        self.turn += 1
        fname = self.out_dir / f"{self.session_ts}_turn{self.turn:02d}.wav"
        try:
            sf.write(str(fname), audio_data["data"], audio_data["samplerate"])
            print(f"  [Recorded -> {fname.name}]")
        except Exception as e:
            print(f"  [Record failed: {e}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text",    action="store_true", help="type input instead of speaking")
    p.add_argument("--no-tts",  action="store_true", help="skip audio output")
    p.add_argument("--record",  action="store_true", help="save TTS audio to recordings/")
    p.add_argument("--model",   default="tiny",      help="whisper model size: tiny or small")
    args = p.parse_args()

    pipeline = Pipeline(mode="mock")
    tts      = TTS() if not args.no_tts else _SilentTTS()
    recorder = _Recorder() if args.record else None

    if args.text:
        text_loop(pipeline, tts, recorder=recorder)
    else:
        voice_loop(pipeline, tts, whisper_model=args.model, recorder=recorder)
