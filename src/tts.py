"""
Text-to-Speech module
Converts spoken confirmation text to audio using Piper TTS.

Setup (one time):
    1. Download piper.exe from:
       https://github.com/rhasspy/piper/releases/latest
       Get: piper_windows_amd64.zip
       Extract piper.exe to: piper/piper.exe  (in repo root)

    2. Download a voice model:
       https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium
       Download both files:
         en_US-lessac-medium.onnx
         en_US-lessac-medium.onnx.json
       Place both in: piper/  (in repo root)

    3. Install winsound (built into Python on Windows, no pip needed)
       or optionally: pip install playsound==1.2.2

Usage:
    from src.tts import TTS
    tts = TTS()
    tts.speak("I have booked a consultation for Monday at 2pm.")

Fallback:
    If piper.exe is not found, text is printed to console only.
    This lets the demo run without TTS during testing.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

_HERE        = Path(__file__).resolve().parent.parent
PIPER_EXE    = _HERE / "piper" / "piper.exe"
PIPER_MODEL  = _HERE / "piper" / "en_US-lessac-medium.onnx"


class TTS:
    def __init__(self):
        self.available = PIPER_EXE.exists() and PIPER_MODEL.exists()
        if self.available:
            print(f"[TTS] Piper ready: {PIPER_MODEL.name}")
        else:
            print("[TTS] Piper not found -- text-only mode")
            print(f"      Expected piper.exe at: {PIPER_EXE}")
            print(f"      Expected model at:     {PIPER_MODEL}")

    def speak(self, text: str) -> None:
        """
        Convert text to speech and play it.
        Falls back to printing if Piper is not installed.
        """
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
        cmd = [
            str(PIPER_EXE),
            "--model", str(PIPER_MODEL),
            "--output_file", output_path,
        ]
        result = subprocess.run(
            cmd,
            input=text,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Piper failed: {result.stderr.strip()}")

    def _play(self, wav_path: str) -> None:
        import sys
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        else:
            try:
                import playsound
                playsound.playsound(wav_path)
            except ImportError:
                os.system(f"aplay {wav_path} 2>/dev/null || afplay {wav_path} 2>/dev/null")


if __name__ == "__main__":
    tts = TTS()
    tts.speak("Hello. I can help with appointment booking, cancellations, and availability.")
    tts.speak("I have booked a consultation for Monday the 23rd of June at 2pm.")
    tts.speak("Your appointment has been cancelled. Is there anything else I can help you with?")
