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

    def listen_vad(self, silence_duration: float = 1.2, max_duration: float = 15.0,
                   start_timeout: float = 8.0, calibrate: float = 0.4,
                   prompt: bool = True) -> str:
        """Record until the speaker goes quiet, then transcribe.

        Energy-based voice activity detection. It measures the ambient noise
        floor for `calibrate` seconds, waits up to `start_timeout` for speech
        to begin, captures until `silence_duration` seconds of trailing
        silence, and caps a turn at `max_duration`. Returns "" if nothing was
        heard. No fixed window, so the caller speaks naturally.

        Requires: pip install sounddevice soundfile numpy
        """
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        block = int(0.03 * SAMPLE_RATE)   # 30 ms frames
        step = block / SAMPLE_RATE

        def rms(x) -> float:
            return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0

        buf: list = []
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32") as stream:
            ambient = []
            for _ in range(max(1, int(calibrate / step))):
                data, _ = stream.read(block)
                ambient.append(rms(data[:, 0]))
            base = sorted(ambient)[len(ambient) // 2] if ambient else 0.0
            # Trigger a few times above the noise floor, with a floor of its own
            # so a silent mic does not set a near-zero threshold.
            threshold = max(base * 3.5, 0.012)

            if prompt:
                print("[STT] Listening, speak now (stops when you go quiet)")

            started = False
            waited = 0.0
            elapsed = 0.0
            silent = 0.0
            while True:
                data, _ = stream.read(block)
                samples = data[:, 0]
                level = rms(samples)
                if not started:
                    waited += step
                    if level > threshold:
                        started = True
                        buf.append(samples.copy())
                    elif waited >= start_timeout:
                        return ""
                    continue
                buf.append(samples.copy())
                elapsed += step
                if level > threshold:
                    silent = 0.0
                else:
                    silent += step
                    if silent >= silence_duration:
                        break
                if elapsed >= max_duration:
                    break

        if not buf:
            return ""
        audio = np.concatenate(buf)
        if len(audio) < int(0.3 * SAMPLE_RATE):
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            tmp_path = tmp.name
        try:
            return self._transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

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
