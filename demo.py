"""
End-to-end voice demo
mic -> Faster-Whisper STT -> pipeline -> Piper TTS

This is the script to run for the viva demo and screen recording.

Voice mode auto-stops when you go quiet, so it feels like a real call: press
Enter to start talking, and recording ends after a short pause of silence.

Usage:
    python demo.py                        voice input (auto-stop), mock pipeline, TTS output
    python demo.py --seconds 5            fixed 5s record window instead of auto-stop
    python demo.py --silence 1.5          end a turn after 1.5s of quiet (default 1.2)
    python demo.py --text                 type instead of speaking (no mic needed)
    python demo.py --no-tts               voice input but text-only output
    python demo.py --text --no-tts        fully text-based, no audio hardware needed
    python demo.py --model tiny           use Whisper tiny (faster, default)
    python demo.py --model small          use Whisper small (more accurate)
    python demo.py --llm cpu              real model -- prompts you to choose which one,
                                            then starts its llama.cpp server automatically
                                            (no separate terminal needed)
    python demo.py --llm cpu --family llama3   skip the prompt, use Llama 3.2 directly
    python demo.py --text --llm cpu       type input, real model, no mic needed
    python demo.py --llm cpu --bargein    voice mode, caller can interrupt mid-reply
                                            (headset recommended, see src/bargein.py)

For --llm cpu, the server now starts automatically (see src/model_server.py).
Only start scripts/03_cpu_server.py yourself if you want a specific quant, or
want the server to keep running across multiple demo.py runs.

Requirements:
    pip install faster-whisper sounddevice soundfile
    (Piper optional -- see src/tts.py for setup)
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import Pipeline
from src.tts import TTS
from src.bargein import speak_with_bargein

def banner(mode_label: str) -> str:
    return f"""
=============================================================
  SME Voice Assistant -- Live Demo
  UWE Bristol MSc Data Science IGP
=============================================================
  Pipeline: mic -> Whisper STT -> spaCy -> LLM -> Piper TTS
  Mode:     {mode_label}
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

def _speak_result(tts, result: dict, recorder=None, stt=None,
                  bargein_threshold: float = None):
    """Speak a pipeline result (greeting or turn reply), handling barge-in
    if enabled. Returns the caller's utterance if they interrupted
    mid-reply, else None. Shared by run_turn and the call-opening greeting.
    """
    bargein_utterance = None
    if bargein_threshold is not None and stt is not None:
        interrupted, caught = speak_with_bargein(tts, stt, result["spoken"], bargein_threshold)
        if interrupted:
            print("  [caller interrupted]")
            if caught:
                bargein_utterance = caught
    else:
        audio = tts.speak(result["spoken"])
        if recorder and audio is not None:
            recorder.save(audio, result["spoken"])
    return bargein_utterance

def run_turn(pipeline: Pipeline, tts, utterance: str, session_id: str,
             recorder=None, stt=None, bargein_threshold: float = None):
    """Run one pipeline turn.

    Returns (end_call, bargein_utterance). bargein_utterance is not None
    when the caller talked over the reply and the barge-in listener
    already captured what they said - the caller loop should feed that
    straight back in as the next turn instead of prompting again.
    """
    print(f"\n  Input:   {utterance!r}")

    result = pipeline.run(utterance, session_id=session_id)

    name_tag = f"  Caller:  {result['caller_name']}\n" if result.get("caller_name") else ""
    print(f"{name_tag}  Action:  {result['action']}")
    print(f"  Spoken:  {result['spoken']}")
    print(f"  Latency: {result['latency_ms']}ms")

    bargein_utterance = _speak_result(tts, result, recorder, stt, bargein_threshold)
    return result.get("end_call", False), bargein_utterance

def voice_loop(pipeline: Pipeline, tts, whisper_model: str, recorder=None,
               mode_label: str = "MOCK", silence: float = 1.2,
               max_seconds: float = 15.0, fixed_seconds: float = 0.0,
               bargein: bool = False) -> None:
    from src.stt import STT
    session_id = "demo-voice"

    # Load Whisper on a background thread instead of blocking here - it can
    # take several seconds, and doing it before anything is said means the
    # operator sees nothing happen for that whole stretch. Loads in
    # parallel with the greeting being spoken below.
    stt_holder = {}

    def _load_stt():
        stt_holder["stt"] = STT(model_size=whisper_model)

    stt_thread = threading.Thread(target=_load_stt, daemon=True)
    stt_thread.start()

    if bargein and not getattr(tts, "available", False):
        print("  [bargein] TTS unavailable (--no-tts or Piper missing) -- barge-in disabled\n")
        bargein = False

    print(banner(mode_label))
    if fixed_seconds > 0:
        print(f"  Press Enter, then speak for up to {fixed_seconds:.0f}s. Say 'quit' to exit.\n")
    else:
        print("  Press Enter to speak. Recording stops when you go quiet. Say 'quit' to exit.\n")

    bargein_threshold = None
    if bargein:
        # Barge-in needs stt loaded and calibrated before playback starts,
        # so this path can't avoid waiting for the background load.
        stt_thread.join()
        stt = stt_holder["stt"]
        print("  [bargein] calibrating room noise, stay quiet a moment...")
        bargein_threshold = stt.calibrate_ambient()
        print(f"  [bargein] on (threshold={bargein_threshold:.4f}). Reliable only with a "
              "headset or low speaker volume near the mic -- see src/bargein.py.\n")

    # Assistant speaks first, like a real receptionist answering - the
    # caller shouldn't have to say something before hearing anything back.
    greeting = pipeline.greet(session_id=session_id)
    print(f"\n  Assistant: {greeting['spoken']}")
    pending_utterance = _speak_result(tts, greeting, recorder,
                                      stt_holder.get("stt"), bargein_threshold)

    stt_thread.join()  # make sure Whisper is ready before we try to listen
    stt = stt_holder["stt"]

    while True:
        if pending_utterance:
            utterance = pending_utterance
            pending_utterance = None
        else:
            try:
                input("  [Press Enter to speak]")
            except (KeyboardInterrupt, EOFError):
                break

            try:
                if fixed_seconds > 0:
                    utterance = stt.listen(duration=fixed_seconds, prompt=True)
                else:
                    utterance = stt.listen_vad(silence_duration=silence, max_duration=max_seconds)
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"  [turn error, continuing: {type(e).__name__}: {e}]")
                continue

            if not utterance:
                print("  [nothing heard, try again]")
                continue
            if "quit" in utterance.lower():
                break

        try:
            ended, pending_utterance = run_turn(pipeline, tts, utterance, session_id, recorder,
                                                stt=stt, bargein_threshold=bargein_threshold)
            if ended:
                print("\n  [Call ended by caller]")
                break
        except Exception as e:
            print(f"  [turn error, continuing: {type(e).__name__}: {e}]")
            continue

    print("\n[Demo ended]")

def text_loop(pipeline: Pipeline, tts, recorder=None,
              mode_label: str = "MOCK") -> None:
    session_id = "demo-text"

    print(banner(mode_label))
    print("  Type a customer utterance and press Enter.\n")
    print("  Demo utterances to try:")
    for i, u in enumerate(TEST_UTTERANCES, 1):
        print(f"    {i}. {u}")
    print()

    greeting = pipeline.greet(session_id=session_id)
    print(f"  Assistant: {greeting['spoken']}")
    _speak_result(tts, greeting, recorder)

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

        ended, _ = run_turn(pipeline, tts, utterance, session_id, recorder)
        if ended:
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
    p.add_argument("--llm",     default="mock", choices=["mock", "cpu", "ollama"],
                   help="LLM backend: mock (rule-based), cpu (real fine-tuned model via "
                        "llama.cpp server), or ollama")
    p.add_argument("--family",  default=None,
                   choices=["phi3", "llama3", "llama1b", "qwen0.5b", "qwen1.5b", "smol360"],
                   help="model family when --llm cpu. If omitted, you'll be prompted to "
                        "choose from the models that have a GGUF built, and the server "
                        "will be started automatically -- no more running "
                        "scripts/03_cpu_server.py by hand in a second terminal.")
    p.add_argument("--llm-port", type=int, default=8080,
                   help="llama.cpp server port for --llm cpu")
    p.add_argument("--silence", type=float, default=1.2,
                   help="seconds of quiet that end a turn (auto-stop voice mode)")
    p.add_argument("--max-seconds", type=float, default=15.0,
                   help="hard cap on a single spoken turn")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="use a fixed record window of this many seconds instead of auto-stop")
    p.add_argument("--bargein", action="store_true",
                   help="let the caller interrupt the assistant mid-reply (voice mode only). "
                        "Energy-based, no echo cancellation -- reliable only with a headset, "
                        "or low speaker volume near the mic. See src/bargein.py for the caveat.")
    args = p.parse_args()

    server_proc = None
    if args.llm == "cpu":
        from src.model_server import prompt_for_family, ensure_server
        family = args.family or prompt_for_family()
        try:
            server_proc = ensure_server(family, port=args.llm_port)
        except RuntimeError as e:
            print(f"\n[model] {e}")
            raise SystemExit(1)
        pipeline   = Pipeline(mode="cpu", model_family=family,
                              cpu_url=f"http://127.0.0.1:{args.llm_port}")
        mode_label = f"REAL MODEL ({family} via llama.cpp CPU server)"
    elif args.llm == "ollama":
        pipeline   = Pipeline(mode="ollama")
        mode_label = "OLLAMA (local)"
    else:
        pipeline   = Pipeline(mode="mock")
        mode_label = "MOCK (rule-based, no GPU needed)"

    tts      = TTS() if not args.no_tts else _SilentTTS()
    recorder = _Recorder() if args.record else None

    try:
        if args.text:
            text_loop(pipeline, tts, recorder=recorder, mode_label=mode_label)
        else:
            voice_loop(pipeline, tts, whisper_model=args.model, recorder=recorder,
                       mode_label=mode_label, silence=args.silence,
                       max_seconds=args.max_seconds, fixed_seconds=args.seconds,
                       bargein=args.bargein)
    finally:
        # Only stop a server this run started - if ensure_server() reused an
        # already-running one (returns None), leave it alone.
        if server_proc is not None:
            print("[model] stopping the llama.cpp server...")
            server_proc.terminate()
