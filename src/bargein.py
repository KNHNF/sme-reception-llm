"""
Barge-in coordinator: lets the caller interrupt the assistant mid-sentence.

Ties together two pieces added for this:
  - TTS.speak_interruptible()   playback that can be cancelled mid-clip
  - STT.detect_speech_onset()   cheap mic-energy check, no transcription

While the assistant is speaking, a background thread watches the mic for
sustained energy above a pre-calibrated threshold. The moment it fires,
playback is cut and the caller's already-in-progress utterance is captured
with the normal STT.listen_vad() the rest of the way.

IMPORTANT CAVEAT (read before demoing): this is energy-based, not acoustic
echo cancellation. If the mic can hear the speakers - any laptop demo
without a headset - it will pick up the assistant's own TTS output as
"speech" too. calibrate_ambient() is measured in silence before the
assistant starts talking specifically to avoid this, but if playback
volume is loud relative to the mic's distance from the speakers, false
barge-ins are still possible. For a reliable demo: use a headset, or keep
speaker volume low and sit close to the mic. This is a real limitation,
not a bug to "just fix" - proper AEC is its own project.
"""

import threading


def speak_with_bargein(tts, stt, text: str, threshold: float,
                        call_stop_event=None):
    """Speak `text`, listening for the caller to talk over it.

    Returns (interrupted, caller_utterance):
      - (False, None)      played to completion, no barge-in
      - (True, "")          caller triggered a barge-in but nothing
                             transcribable followed (false trigger, cough,
                             etc.) - caller code should treat like silence
      - (True, "text...")   caller barged in and said something; this is
                             their full utterance, ready to run through the
                             pipeline same as a normal turn

    `threshold` must come from STT.calibrate_ambient(), measured once in
    silence at the start of the call - see module docstring for why.
    `call_stop_event` is the existing "End Call" stop event; passed through
    so a caller hanging up mid-barge-in still exits promptly.
    """
    if not tts.available:
        tts.speak(text)
        return False, None

    barge_event = threading.Event()
    monitor_stop = threading.Event()

    def monitor():
        heard = stt.detect_speech_onset(threshold=threshold, stop_event=monitor_stop)
        if heard:
            barge_event.set()

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    interrupted = tts.speak_interruptible(text, stop_event=barge_event)

    monitor_stop.set()
    monitor_thread.join(timeout=1.0)

    if not interrupted:
        return False, None

    # Caller is already mid-utterance. Capture the rest of it now, using
    # the same pre-calibrated threshold (skip re-calibration - see
    # STT.listen_vad's `threshold` param docstring for why).
    utterance = stt.listen_vad(
        silence_duration=1.0,
        max_duration=15.0,
        start_timeout=2.0,
        beep=False,
        stop_event=call_stop_event,
        threshold=threshold,
    )
    return True, utterance
