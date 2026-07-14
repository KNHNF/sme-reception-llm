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
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

LOCAL_MODEL_DIR = "E:/Coding/models/faster-whisper"
DEFAULT_MODEL   = "tiny"
SAMPLE_RATE     = 16000
CHANNELS        = 1


def _beep(frequency: float = 880.0, duration: float = 0.12, volume: float = 0.3,
          on_fail=None) -> bool:
    """Short cue tone marking the exact moment real listening starts.

    Without this, callers guess when the mic is ready and often start
    talking during calibration/stream-open, losing the first word or two.

    Generates the tone and plays it through sounddevice (same backend as
    the mic capture and TTS playback everywhere else in this app), instead
    of winsound.Beep. This call site itself no longer overlaps with an open
    mic InputStream (see listen_vad, which now calibrates and closes its
    stream before calling this) - the earlier hypothesis (winsound vs
    sounddevice cross-API contention) turned out to still be missing beeps
    after the switch, so the real conflict was more likely concurrent
    input+output stream use, not the API mismatch itself.

    Still wrapped in try/except: a missing cue is much less disruptive to
    a live call than raising and killing the turn. Reported by print()
    AND via the optional `on_fail` callback - a GUI app like call_ui.py
    has no visible console most callers will actually be watching, so a
    bare print() here can silently vanish. Pass e.g. a transcript-log
    function as on_fail to see beep failures in the app itself.
    Returns True if the tone actually played, False if it failed.
    """
    try:
        import numpy as np
        import sounddevice as sd
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        tone = (volume * np.sin(2 * np.pi * frequency * t)).astype("float32")
        sd.play(tone, SAMPLE_RATE)
        sd.wait()
        return True
    except Exception as e:
        msg = f"[STT] beep failed: {type(e).__name__}: {e}"
        print(msg)
        if on_fail is not None:
            try:
                on_fail(msg)
            except Exception:
                pass
        return False

def calibrate_ambient(duration: float = 0.5) -> float:
    """Measure the room noise floor once, in silence, and return a
    threshold to reuse across a whole call.

    Module-level (not a method) so it can be called before the STT model
    itself has finished loading - it only touches sounddevice/numpy, never
    self.model - letting a caller (e.g. call_ui.py) calibrate in true
    silence right at call start, before kicking off the slow Whisper load
    in the background and before the greeting speaks, instead of having to
    wait for both first.

    Requires: pip install sounddevice numpy
    """
    import numpy as np
    import sounddevice as sd

    block = int(0.03 * SAMPLE_RATE)
    step = block / SAMPLE_RATE

    def rms(x) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32") as stream:
        ambient = []
        for _ in range(max(1, int(duration / step))):
            data, _ = stream.read(block)
            ambient.append(rms(data[:, 0]))
    base = sorted(ambient)[len(ambient) // 2] if ambient else 0.0
    return max(base * 3.5, 0.012)


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
                   prompt: bool = True, beep: bool = True, stop_event=None,
                   threshold: float = None, beep_log=None) -> str:
        """Record until the speaker goes quiet, then transcribe.

        Energy-based voice activity detection. It measures the ambient noise
        floor for `calibrate` seconds, waits up to `start_timeout` for speech
        to begin, captures until `silence_duration` seconds of trailing
        silence, and caps a turn at `max_duration`. Returns "" if nothing was
        heard. No fixed window, so the caller speaks naturally.

        `beep`: play a short tone right when real listening starts (after
        calibration, before the wait-for-speech loop), so the caller knows
        exactly when to talk instead of guessing and losing the first word.

        `stop_event`: an optional threading.Event checked every ~30ms frame.
        When set, returns "" immediately instead of running to completion -
        without this, a caller-facing "End Call" button can take up to
        `max_duration` seconds to actually take effect, since nothing
        previously interrupted a blocking read loop mid-recording.

        `threshold`: skip the ambient-noise calibration and use this value
        directly. ALWAYS pass this in a multi-turn call (call_ui.py /
        demo.py) - calibrate once with calibrate_ambient() before the
        greeting speaks, in genuine silence, and reuse that one value for
        every turn. Leaving this None makes listen_vad recalibrate itself
        fresh on every single turn, right after TTS playback just finished -
        that recalibration window can catch trailing room echo or the audio
        device still settling, which inflates the computed threshold well
        above the true room noise floor. That mismatch is the most likely
        cause of a caller having to raise their voice to be heard at all:
        the threshold silently crept up on some earlier turn and stayed
        high for the rest of the call. Passing a single pre-calibrated
        threshold removes that per-turn recalibration entirely.

        `beep_log`: optional callable(str) invoked if the beep tone fails
        to play. print() alone is easy to miss in a GUI app with no
        visible console - pass e.g. call_ui.py's transcript logger here so
        beep failures show up in the app itself instead of vanishing.

        Requires: pip install sounddevice soundfile numpy
        """
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        block = int(0.03 * SAMPLE_RATE)   # 30 ms frames
        step = block / SAMPLE_RATE

        def rms(x) -> float:
            return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0

        def stopped() -> bool:
            return stop_event is not None and stop_event.is_set()

        # Settle delay before opening the mic. Right after TTS playback
        # finishes, the audio device can still be mid-transition (Windows
        # especially) - widened from 0.15s since that wasn't always enough
        # headroom on real hardware and a rushed re-open is a plausible
        # contributor to both the inflated-threshold and missing-beep
        # reports.
        time.sleep(0.35)

        if threshold is None:
            # Calibrate on its own short-lived stream, fully closed before
            # the beep plays below. Previously this ran inside the SAME
            # InputStream used for the whole listen (beep included, mid-
            # stream) - trying to play the beep out while the mic stream
            # was still open at the same time is the actual likely cause of
            # the beep sometimes not playing at all (concurrent input+output
            # contention), not just a winsound-vs-sounddevice mismatch.
            # Closing this stream before playing anything removes that
            # overlap entirely.
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                dtype="float32") as cal_stream:
                ambient = []
                for _ in range(max(1, int(calibrate / step))):
                    if stopped():
                        return ""
                    data, _ = cal_stream.read(block)
                    ambient.append(rms(data[:, 0]))
            base = sorted(ambient)[len(ambient) // 2] if ambient else 0.0
            # Trigger a few times above the noise floor, with a floor of
            # its own so a silent mic does not set a near-zero threshold.
            threshold = max(base * 3.5, 0.012)

        if stopped():
            return ""

        buf: list = []
        started = False
        waited = 0.0
        elapsed = 0.0
        silent = 0.0
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32") as stream:
            # Beep AFTER the listening stream is already open, not before.
            # PortAudio buffers input continuously once the stream starts,
            # whether or not stream.read() has been called yet, so anything
            # the caller says during/right after the beep is sitting in that
            # buffer waiting to be read below - it isn't lost. The previous
            # order (beep, then open a fresh stream) had a real dead gap
            # between the beep ending and the mic actually capturing, and
            # callers reliably start talking right on the beep, which is
            # exactly the word or two that gap was swallowing.
            if beep:
                _beep(on_fail=beep_log)
            if prompt:
                print("[STT] Listening, speak now (stops when you go quiet)")
            # Keeps the last ~150ms of pre-speech audio so a short reply
            # ("yes", "no", "that's correct") doesn't lose its attack
            # transient. VAD only starts buffering on the exact frame that
            # first crosses threshold, which clips the very start of the
            # word - for a long sentence that's negligible, but for a
            # one-word answer it's a large enough fraction of the whole clip
            # to be the difference between Whisper hearing "yes" and hearing
            # something else entirely (this is the actual cause of short
            # replies like "yes"/"to book" coming back as unrelated words -
            # not a threshold or timing bug, just too little acoustic
            # context at the very start of a very short utterance).
            pre_roll: deque = deque(maxlen=5)
            while True:
                if stopped():
                    return ""
                data, _ = stream.read(block)
                samples = data[:, 0]
                level = rms(samples)
                if not started:
                    waited += step
                    if level > threshold:
                        started = True
                        buf.extend(pre_roll)
                        buf.append(samples.copy())
                    else:
                        pre_roll.append(samples.copy())
                        if waited >= start_timeout:
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

        if stopped():
            return ""

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

    def calibrate_ambient(self, duration: float = 0.5) -> float:
        """Measure the room noise floor once, in silence, and return a
        threshold to reuse across a whole call.

        Call this once at the start of a call, before the assistant speaks,
        and pass the result to every listen_vad() call afterwards via its
        `threshold` argument - never let listen_vad() recalibrate itself
        mid-call (see that method's docstring for why that goes wrong).
        Do NOT call it while TTS is playing: if the mic can hear the
        speakers at all (no headset, no echo cancellation), it would
        calibrate on the assistant's own voice instead of room noise, and
        every barge-in / threshold check downstream would be wrong.

        Thin wrapper around the module-level calibrate_ambient() below -
        kept as a method for existing call sites (self.stt.calibrate_ambient()).
        Call the module-level version directly if you need ambient noise
        measured before the STT model itself has finished loading (this
        doesn't touch self.model at all, only sounddevice/numpy).
        """
        return calibrate_ambient(duration)

    def detect_speech_onset(self, threshold: float, min_frames: int = 3,
                            stop_event=None) -> bool:
        """Watch the mic and return True the moment sustained speech starts.

        This is the barge-in listener: run it on a background thread while
        TTS is playing. It does not transcribe anything, just detects that
        the caller has started talking, cheaply enough to poll continuously
        for the whole duration of playback.

        `threshold` must come from calibrate_ambient(), measured in silence
        before playback started. Do not calibrate live here: while TTS is
        playing, "ambient" would include the assistant's own voice bleeding
        into the mic (speaker/mic echo), which on a laptop with no headset
        means this will false-trigger on the assistant's own speech. This
        module has no acoustic echo cancellation. Barge-in only works
        reliably with a headset, or a mic with enough physical separation
        from the speakers that TTS playback stays below `threshold`.

        `min_frames`: consecutive 30ms frames required above threshold
        before triggering, so a single click, breath, or spike of speaker
        bleed doesn't count as a barge-in.

        `stop_event`: checked every frame; if set externally (playback
        already finished naturally, or the call ended), returns False.

        Requires: pip install sounddevice numpy
        """
        import numpy as np
        import sounddevice as sd

        block = int(0.03 * SAMPLE_RATE)

        def rms(x) -> float:
            return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0

        consecutive = 0
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32") as stream:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return False
                data, _ = stream.read(block)
                level = rms(data[:, 0])
                if level > threshold:
                    consecutive += 1
                    if consecutive >= min_frames:
                        return True
                else:
                    consecutive = 0

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
            # Zero-training domain bias: Whisper reads this as a hint of
            # plausible vocabulary, which matters most on exactly the short,
            # acoustically ambiguous clips ("yes", "to book") that have been
            # coming back as unrelated words. Costs nothing, no fine-tuning,
            # no extra latency worth mentioning - just biases the decoder's
            # priors toward what a caller here actually says.
            initial_prompt=(
                "Appointment booking call. General appointment, consultation, "
                "follow-up, booking, cancellation, availability. Yes, no, "
                "that's correct, Monday, Tuesday, Wednesday, Thursday, Friday, "
                "morning, afternoon."
            ),
            # Each call here is one short, independent utterance in its own
            # file - there is no real multi-segment context to condition on
            # within a single turn, and condition_on_previous_text defaults
            # to True in faster-whisper, which mainly helps long continuous
            # dictation, not isolated short replies. Off is the safer default
            # here and is a known mitigation for repetition/hallucination
            # artifacts on short or quiet clips.
            condition_on_previous_text=False,
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
