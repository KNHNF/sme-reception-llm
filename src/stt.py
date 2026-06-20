"""
Speech-to-Text module
Records from the microphone and returns a transcript string.

Uses faster-whisper (same models as whisper-stt-eval repo).
Models expected at: E:/Coding/models/faster-whisper

Usage:
    from src.stt import STT
    stt = STT(model_size="tiny")
    text = stt.listen(duration=5)

    or standalone:
    python src/stt.py
"""

import os
import tempfile
import time
from pathlib import Path

LOCAL_MODEL_DIR = "E:/Coding/models/faster-whisper"
DEFAULT_MODEL   = "tiny"
SAMPLE_RATE     = 16000
CHANNELS        = 1


class STT:
    def __init__(self, model_size: str = DEFAULT_MODEL):
        from faster_whisper import WhisperModel

        model_path = str(Path(LOCAL_MODEL_DIR) / model_size)
        if Path(model_path).exists():
            print(f"[STT] Loading model from {model_path}")
            self.model = WhisperModel(model_path, device="cpu", compute_type="int8")
        else:
            print(f"[STT] Local model not found at {model_path}, downloading {model_size}...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

        self.model_size = model_size
        print(f"[STT] Ready ({model_size})")

    def listen(self, duration: float = 5.0, prompt: bool = True) -> str:
        """
        Record from the default microphone for `duration` seconds,
        then transcribe and return the text.

        Requires: pip install sounddevice soundfile
        """
        import sounddevice as sd
        import soundfile as sf

        if prompt:
            print(f"\n[STT] Recording for {duration}s... speak now")

        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            tmp_path = tmp.name

        try:
            text = self._transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

        return text

    def transcribe_file(self, path: str) -> str:
        """Transcribe an existing audio file and return the text."""
        return self._transcribe(path)

    def _transcribe(self, path: str) -> str:
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(
            path,
            beam_size=1,
            temperature=0.0,
            vad_filter=True,
            language="en",
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        latency = (time.perf_counter() - t0) * 1000
        print(f"[STT] Transcript ({latency:.0f}ms): {text!r}")
        return text


if __name__ == "__main__":
    stt = STT(model_size="tiny")
    while True:
        input("\nPress Enter to record 5 seconds...")
        result = stt.listen(duration=5.0)
        print(f"You said: {result}")
