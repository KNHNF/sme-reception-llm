"""
Text-to-Speech module using Piper TTS.

Setup (one time):
    1. Download piper_windows_amd64.zip from:
       https://github.com/rhasspy/piper/releases/latest
       Extract piper.exe to: piper/piper.exe  (repo root)

    2. Download voice model from:
       https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium
       Place en_US-lessac-medium.onnx and .onnx.json in: piper/  (repo root)

Standalone diagnostic (run before demo.py to confirm Piper works):
    python src/tts.py

Fallback:
    If piper.exe is not found, text is printed to console only.
"""

import os
import subprocess
import tempfile
from pathlib import Path

_HERE       = Path(__file__).resolve().parent.parent
PIPER_EXE   = _HERE / "piper" / "piper.exe"
PIPER_MODEL = _HERE / "piper" / "en_US-ryan-high.onnx"

class TTS:
    def __init__(self):
        self.available = PIPER_EXE.exists() and PIPER_MODEL.exists()
        if self.available:
            print(f"[TTS] Piper ready: {PIPER_MODEL.name}")
        else:
            print("[TTS] Piper not found -- text-only mode")
            print(f"      Expected piper.exe : {PIPER_EXE}")
            print(f"      Expected model     : {PIPER_MODEL}")

    def speak(self, text: str) -> None:
        """Convert text to speech and play. Falls back to print if Piper unavailable."""
        print(f"[TTS] Speaking: {text!r}")
        if not self.available:
            print(f"[TTS] (audio disabled) -> {text}")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            self._run_piper(text, wav_path)
            self._play(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _run_piper(self, text: str, output_path: str) -> None:
        """
        Run Piper subprocess. Tries --output-file first (Piper >= 1.0),
        then --output_file (older builds). Reports full details on failure.
        """
        result = None
        last_cmd = None

        for flag in ("--output-file", "--output_file"):
            cmd = [str(PIPER_EXE), "--model", str(PIPER_MODEL), flag, output_path]
            last_cmd = cmd
            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
            )
            if result.returncode == 0:
                return  # success

            # Bad flag? Try the other variant
            hint = (result.stderr + result.stdout).lower()
            if any(w in hint for w in ("unrecognized", "unknown", "usage", "invalid")):
                continue
            break  # Real failure -- stop

        stderr = result.stderr.strip() if result else ""
        stdout = result.stdout.strip() if result else ""
        raise RuntimeError(
            f"Piper failed (exit {result.returncode if result else '?'}).\n"
            f"  stderr : {stderr or '(empty)'}\n"
            f"  stdout : {stdout or '(empty)'}\n"
            f"  cmd    : {' '.join(last_cmd or [])}\n"
            "  Likely causes:\n"
            "    1. Missing VC++ runtime -- https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
            "    2. Wrong architecture (need piper_windows_amd64.zip)\n"
            "    3. Corrupt model -- re-download .onnx and .onnx.json"
        )

    def speak_interruptible(self, text: str, stop_event) -> bool:
        """Like speak(), but playback can be cut short mid-sentence.

        Returns True if playback was interrupted (barge-in), False if it
        played to completion. `stop_event` is a threading.Event that some
        other thread (a mic-monitoring loop, see STT.detect_speech_onset)
        sets the instant it hears the caller start talking.

        This uses sounddevice instead of winsound.PlaySound. winsound
        blocks with no interrupt hook, so once playback started there was
        no way to stop it early - barge-in needs playback that can be
        cancelled from another thread.
        """
        print(f"[TTS] Speaking (interruptible): {text!r}")
        if not self.available:
            print(f"[TTS] (audio disabled) -> {text}")
            return False

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            self._run_piper(text, wav_path)
            return self._play_interruptible(wav_path, stop_event)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _play_interruptible(self, wav_path: str, stop_event) -> bool:
        """Play a WAV via sounddevice, polling stop_event every ~30ms.
        Returns True if stopped early, False if it finished naturally.

        Requires: pip install sounddevice soundfile
        """
        import sounddevice as sd
        import soundfile as sf

        data, samplerate = sf.read(wav_path, dtype="float32")
        sd.play(data, samplerate)
        try:
            while sd.get_stream() is not None and sd.get_stream().active:
                if stop_event is not None and stop_event.is_set():
                    sd.stop()
                    return True
                sd.sleep(30)
        except Exception:
            sd.stop()
            raise
        return False

    def to_wav_bytes(self, text: str) -> bytes | None:
        """Generate WAV bytes from text without playing. Used by Streamlit UI."""
        if not self.available:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            self._run_piper(text, wav_path)
            with open(wav_path, "rb") as f:
                return f.read()
        except Exception:
            return None
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _play(self, wav_path: str) -> None:
        """Blocking playback via sounddevice - same backend as
        speak_interruptible()'s playback, the STT beep, and mic capture
        everywhere else in this app. Previously used winsound.PlaySound on
        Windows, which is a separate audio API from the sounddevice-based
        mic recording that follows immediately after - mixing the two was
        the likely cause of intermittent lost beeps / clipped first words
        (see stt.py's _beep). Using one backend throughout removes that.
        """
        import sounddevice as sd
        import soundfile as sf
        data, samplerate = sf.read(wav_path, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()

if __name__ == "__main__":
    # Standalone diagnostic -- mirrors demo.py but TTS only.
    # Run this first to confirm Piper works before the full voice demo.
    print("=== Piper TTS diagnostic ===")
    print(f"  piper.exe  : {PIPER_EXE}  {'[FOUND]' if PIPER_EXE.exists() else '[MISSING]'}")
    print(f"  voice model: {PIPER_MODEL}  {'[FOUND]' if PIPER_MODEL.exists() else '[MISSING]'}")

    if not PIPER_EXE.exists():
        print()
        print("piper.exe is missing.")
        print("Download piper_windows_amd64.zip from:")
        print("  https://github.com/rhasspy/piper/releases/latest")
        print("Extract piper.exe into the piper/ folder at the repo root.")
        raise SystemExit(1)

    if not PIPER_MODEL.exists():
        print()
        print("Voice model missing.")
        print("Download both files from:")
        print("  https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium")
        print("Place en_US-lessac-medium.onnx and .onnx.json into the piper/ folder.")
        raise SystemExit(1)

    tts = TTS()

    phrases = [
        "Piper TTS is working correctly.",
        "Thank you for calling City Medical Practice.",
        "I have booked a consultation for Monday at 10 a m.",
        "Your appointment has been cancelled. Is there anything else I can help you with?",
    ]

    print()
    for phrase in phrases:
        try:
            tts.speak(phrase)
            print("  [OK]")
        except RuntimeError as exc:
            print("  [FAIL]")
            print(f"  {exc}")
            raise SystemExit(1)
        print()

    print("All tests passed. Piper is working.")
    print("You can now run:  python demo.py")
