"""
STT control-flow tests via a monkeypatched sounddevice/numpy, no mic or
faster-whisper model required. Covers calibrate_ambient(), listen_vad(),
detect_speech_onset(), and a regression guard for the beep/InputStream
concurrency bug fixed in src/stt.py (beep must never play while an
InputStream is open - that was the actual cause of the intermittently
missing beep on real calls).

Run: python test_stt_vad.py
Exit 0 = all passed, exit 1 = failures.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def check(label, ok, detail=""):
    print(f"  {PASS if ok else FAIL} {label}")
    if not ok and detail:
        print(f"       {detail}")
    results.append(ok)


# --- Build a fake sounddevice + numpy environment ---------------------

import numpy as np  # numpy is a real, lightweight dependency - fine to use directly

events = []          # records "input_open" / "input_close" / "play" in order
input_open_count = [0]


class FakeInputStream:
    def __init__(self, samplerate=16000, channels=1, dtype="float32"):
        self.samplerate = samplerate
        self._frame = np.zeros(int(0.03 * samplerate), dtype="float32")
        self.loud = False

    def __enter__(self):
        events.append("input_open")
        input_open_count[0] += 1
        return self

    def __exit__(self, *a):
        events.append("input_close")
        input_open_count[0] -= 1
        return False

    def read(self, block):
        level = 0.5 if self.loud else 0.0
        data = np.full((block, 1), level, dtype="float32")
        return data, None


def fake_play(tone, samplerate):
    events.append("play")
    if input_open_count[0] > 0:
        events.append("play_while_input_open!")  # the exact bug this guards against


def fake_wait():
    pass


fake_sd = types.ModuleType("sounddevice")
fake_sd.InputStream = FakeInputStream
fake_sd.play = fake_play
fake_sd.wait = fake_wait
fake_sd.rec = lambda *a, **k: np.zeros((1, 1), dtype="float32")
sys.modules["sounddevice"] = fake_sd

fake_sf = types.ModuleType("soundfile")
fake_sf.write = lambda path, audio, sr: Path(path).touch()
fake_sf.read = lambda path, dtype="float32": (np.zeros(16000, dtype="float32"), 16000)
sys.modules["soundfile"] = fake_sf

import time as _time_mod
_orig_sleep = _time_mod.sleep
_time_mod.sleep = lambda s: None  # skip real delays in tests

from src.stt import STT, _beep  # noqa: E402


def make_stt():
    # Bypass __init__ (it loads a real faster_whisper model) and stub
    # transcription so listen_vad()'s control flow can be tested without
    # a real model.
    obj = STT.__new__(STT)
    obj.model_size = "tiny"

    class _Seg:
        text = "hello"

    class _FakeModel:
        def transcribe(self, path, **kwargs):
            return [_Seg()], None

    obj.model = _FakeModel()
    return obj


# 1. calibrate_ambient returns a sane threshold with a floor

print("=== calibrate_ambient ===")
stt = make_stt()
events.clear()
threshold = stt.calibrate_ambient(duration=0.1)
check("returns a positive threshold", threshold > 0, str(threshold))
check("respects the 0.012 floor on a silent mic", threshold >= 0.012, str(threshold))
check("opens exactly one InputStream", events.count("input_open") == 1, str(events))
check("closes it again (no leaked stream)", events.count("input_close") == 1, str(events))


# 2. listen_vad: stop_event set immediately returns "" without hanging

print("\n=== listen_vad: stop_event honoured ===")
import threading
stop_ev = threading.Event()
stop_ev.set()
events.clear()
result = stt.listen_vad(stop_event=stop_ev, threshold=0.5)
check("returns empty string when already stopped", result == "")


# 3. listen_vad: never exceeds threshold -> times out to "" (fast timeout)

print("\n=== listen_vad: silence times out cleanly ===")
events.clear()
result = stt.listen_vad(start_timeout=0.06, calibrate=0.03, threshold=0.5, prompt=False)
check("returns empty string on silence (nobody spoke)", result == "")
check("did not crash or hang", True)


# 4. Regression guard: beep never plays while an InputStream is open
#    (the actual root cause of the intermittently missing beep bug)

print("\n=== regression: beep and InputStream are never concurrent ===")
events.clear()
stt.listen_vad(calibrate=0.03, start_timeout=0.03, beep=True, prompt=False)
check("beep played at some point", "play" in events, str(events))
check("beep never played while an InputStream was open",
      "play_while_input_open!" not in events, str(events))
# Also confirm the calibration stream fully closes before the beep, and a
# fresh stream opens after - not one long-lived stream spanning the beep.
play_idx = events.index("play")
before = events[:play_idx]
after = events[play_idx + 1:]
check("calibration stream closed before the beep played",
      before.count("input_open") == before.count("input_close"), str(before))
check("a new stream opens after the beep (not reusing one held open across it)",
      "input_open" in after, str(after))


# 5. detect_speech_onset triggers after min_frames consecutive loud frames

print("\n=== detect_speech_onset ===")


class LoudAfterNFrames(FakeInputStream):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def read(self, block):
        self.calls += 1
        self.loud = self.calls > 2  # quiet for 2 frames, then loud
        return super().read(block)


fake_sd.InputStream = LoudAfterNFrames
triggered = stt.detect_speech_onset(threshold=0.1, min_frames=3)
check("detects onset once sustained speech starts", triggered is True)

fake_sd.InputStream = FakeInputStream  # restore for anything after


# 6. _beep itself never raises even if sounddevice misbehaves

print("\n=== _beep never raises, even on backend failure ===")


def broken_play(*a, **k):
    raise RuntimeError("device busy")


_orig_play = fake_sd.play
fake_sd.play = broken_play
try:
    _beep()
    ok = True
except Exception:
    ok = False
check("swallows a backend failure instead of crashing the turn", ok)
fake_sd.play = _orig_play


total = len(results)
passed = sum(results)
print(f"\n{passed}/{total} passed")
_time_mod.sleep = _orig_sleep
sys.exit(0 if passed == total else 1)
