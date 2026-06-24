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
PIPER_MODEL = _HERE / "piper" / "en_US-lessac-medium.onnx"

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
        import sys
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        else:
            os.system("aplay " + wav_path + " 2>/dev/null || afplay " + wav_path + " 2>/dev/null")

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
