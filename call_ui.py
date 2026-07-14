"""
Offline call demo, GUI wrapper around the existing tested pipeline.

Reuses exactly what demo.py already uses and already works: Pipeline (real
fine-tuned model via llama.cpp CPU server), STT.listen_vad (continuous mic
listening, auto-stops on silence, same as a real call), TTS.speak (Piper,
offline). This file adds no new STT/TTS/model logic, only a visual shell and
threading so the window stays responsive during blocking calls.

No prerequisite step needed - if no server is already running at --cpu-url,
this prompts you to choose a model (from what has a GGUF built in
checkpoints/gguf/) and starts scripts/03_cpu_server.py automatically. See
src/model_server.py. Pass --family to skip the prompt.

Usage:
    python call_ui.py
    python call_ui.py --family llama3 --whisper small

Start Call listens continuously through the microphone, same as demo.py's
voice mode: speak, pause, it responds, no clicking per turn. Test Call runs
a scripted sequence through the same pipeline via typed text instead of the
mic, a reliable fallback take if live speech recognition misbehaves during
recording.
"""

import argparse
import calendar as _calmod
import json
import os
import queue
import sys
import threading
import tkinter as tk
import urllib.parse
import urllib.request
from datetime import date as _date_cls, datetime as _datetime_cls
from pathlib import Path
from tkinter import font as tkfont

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import Pipeline
from src.tts import TTS
from src.bargein import speak_with_bargein

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    # CPU display just shows a "not installed" hint instead of crashing the
    # whole app over an optional monitoring feature - pip install psutil.
    _HAS_PSUTIL = False

CALENDAR_PATH = Path(__file__).parent / "data" / "calendar.json"

TEST_UTTERANCES = [
    "I'd like to book a consultation for next Monday at 2pm",
    "Do you have any slots available on Thursday for a general appointment?",
    "I need to cancel my appointment on Wednesday at 10am",
    "Book me in for a follow-up",
    "What are your opening hours?",
    "That's all, goodbye",
]

STATUS_COLOUR = {
    "Ready": "#9aa0a6", "Listening": "#1f9d55", "Thinking": "#d97706",
    "Speaking": "#1f77b4", "Call ended": "#666666", "Error": "#c62828",
}

# (background, foreground) per aggregate day/slot status.
CAL_COLOURS = {
    "free":    ("#d7f5dc", "#1f9d55"),
    "partial": ("#fff3cd", "#b8860b"),
    "full":    ("#fbdada", "#c62828"),
    "none":    ("#f0f0f0", "#aaaaaa"),
}

SLOT_TIMES = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
              "14:00", "14:30", "15:00", "15:30", "16:00", "16:30"]
SERVICES = [("general", "General"), ("consultation", "Consultation"), ("follow_up", "Follow-up")]


def check_cpu_server(cpu_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{cpu_url}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


class CalendarPanel(tk.Frame):
    """Outlook-style month view: colour-coded days (free / partly booked /
    fully booked / no slots), click a day to zoom into its per-time,
    per-service detail. Reads data/calendar.json fresh on every render, so
    calling refresh() after a turn shows new bookings immediately - no
    diffing, no stale state.
    """

    def __init__(self, parent, calendar_path: Path, bg: str = "white"):
        super().__init__(parent, bg=bg)
        self.calendar_path = calendar_path
        self.bg = bg
        today = _date_cls.today()
        self.view_year = today.year
        self.view_month = today.month
        self.selected_date = None  # None = month view, else "YYYY-MM-DD"

        self._bold = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self._small = tkfont.Font(family="Segoe UI", size=8)

        header = tk.Frame(self, bg=bg)
        header.pack(fill="x")
        tk.Button(header, text="<", relief="flat", bg="#eeeeee", width=2,
                  command=self._prev_month).pack(side="left")
        tk.Button(header, text="Today", relief="flat", bg="#eeeeee",
                  command=self._go_to_today).pack(side="left", padx=(4, 0))
        self.month_label = tk.Label(header, text="", font=self._bold, bg=bg, fg="#222222")
        self.month_label.pack(side="left", expand=True)
        tk.Button(header, text=">", relief="flat", bg="#eeeeee", width=2,
                  command=self._next_month).pack(side="right")

        legend = tk.Frame(self, bg=bg, pady=4)
        legend.pack(fill="x")
        for key, label in [("free", "Free"), ("partial", "Partly booked"),
                            ("full", "Fully booked"), ("none", "No slots")]:
            bg_c, _ = CAL_COLOURS[key]
            swatch = tk.Frame(legend, bg=bg_c, width=10, height=10,
                               highlightthickness=1, highlightbackground="#cccccc")
            swatch.pack(side="left", padx=(0, 2))
            swatch.pack_propagate(False)
            tk.Label(legend, text=label, font=self._small, bg=bg, fg="#666666").pack(
                side="left", padx=(0, 8))

        self.body = tk.Frame(self, bg=bg)
        self.body.pack(fill="both", expand=True)

        self.render()

    def _prev_month(self):
        self.selected_date = None
        self.view_month -= 1
        if self.view_month < 1:
            self.view_month = 12
            self.view_year -= 1
        self.render()

    def _next_month(self):
        self.selected_date = None
        self.view_month += 1
        if self.view_month > 12:
            self.view_month = 1
            self.view_year += 1
        self.render()

    def _go_to_today(self):
        """Jump straight back to the current month, no matter how far the
        prev/next buttons have wandered. Without this, getting back to
        "now" after a few clicks of < or > means counting months by hand,
        the exact friction Karan flagged."""
        today = _date_cls.today()
        self.view_year = today.year
        self.view_month = today.month
        self.selected_date = None
        self.render()

    def _load_slots(self) -> list:
        try:
            data = json.loads(self.calendar_path.read_text())
        except Exception:
            return []
        return data.get("slots", [])

    def refresh(self):
        """Call after every turn so bookings show up live, no manual diffing."""
        self.render()

    def render(self):
        for w in self.body.winfo_children():
            w.destroy()
        if self.selected_date is None:
            self._render_month()
        else:
            self._render_day()

    @staticmethod
    def _status_for_slots(slots_for_key: list) -> str:
        """Free / partial / full, matching what a caller could actually book.

        A slot's "available" flag only tracks whether it's been booked, it
        never accounts for the time of day already passing. Without the same
        filter calendar_store.find_slots() applies, today's date keeps
        showing green/amber long after its last remaining slot time has gone
        (e.g. still "free" at 10pm even though every slot for today was this
        morning), which contradicts what check_availability actually offers.
        """
        if not slots_for_key:
            return "none"
        now = _datetime_cls.now()
        today_str = now.strftime("%Y-%m-%d")
        now_time = now.strftime("%H:%M")
        bookable = [
            s for s in slots_for_key
            if s["date"] > today_str or (s["date"] == today_str and s["time"] > now_time)
        ]
        if not bookable:
            return "full"  # today, but every slot's time has already passed
        free = sum(1 for s in bookable if s["available"])
        if free == len(bookable):
            return "free"
        if free == 0:
            return "full"
        return "partial"

    def _render_month(self):
        self.month_label.configure(text=f"{_calmod.month_name[self.view_month]} {self.view_year}")

        slots = self._load_slots()
        by_date = {}
        for s in slots:
            by_date.setdefault(s["date"], []).append(s)

        today_str = _date_cls.today().isoformat()

        hdr = tk.Frame(self.body, bg=self.bg)
        hdr.pack(fill="x")
        for wd in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            tk.Label(hdr, text=wd, font=self._small, bg=self.bg, fg="#999999",
                     width=4).pack(side="left", expand=True)

        cal = _calmod.Calendar(firstweekday=0)
        grid = tk.Frame(self.body, bg=self.bg)
        grid.pack(fill="both", expand=True)

        for week in cal.monthdatescalendar(self.view_year, self.view_month):
            row = tk.Frame(grid, bg=self.bg)
            row.pack(fill="x", expand=True)
            for d in week:
                in_month = d.month == self.view_month
                date_str = d.isoformat()
                day_slots = by_date.get(date_str, [])
                status = self._status_for_slots(day_slots) if in_month else "none"
                if in_month:
                    bg_c, fg_c = CAL_COLOURS[status]
                else:
                    bg_c, fg_c = self.bg, "#dddddd"
                border = "#1f77b4" if date_str == today_str else "#e0e0e0"
                cell = tk.Label(row, text=str(d.day), font=self._small, bg=bg_c, fg=fg_c,
                                 width=4, height=2, relief="flat",
                                 highlightthickness=1, highlightbackground=border)
                cell.pack(side="left", expand=True, fill="both", padx=1, pady=1)
                if in_month and day_slots:
                    cell.bind("<Button-1>", lambda e, ds=date_str: self._select_day(ds))
                    cell.configure(cursor="hand2")

    def _select_day(self, date_str: str):
        self.selected_date = date_str
        self.render()

    def _back_to_month(self):
        self.selected_date = None
        self.render()

    def _render_day(self):
        d = _datetime_cls.strptime(self.selected_date, "%Y-%m-%d").date()
        self.month_label.configure(text=d.strftime("%A %d %B %Y"))

        tk.Button(self.body, text="< Back to month", relief="flat", bg="#eeeeee",
                  anchor="w", command=self._back_to_month).pack(fill="x", pady=(0, 6))

        slots = self._load_slots()
        day_slots = [s for s in slots if s["date"] == self.selected_date]
        by_time = {}
        for s in day_slots:
            by_time.setdefault(s["time"], {})[s["service"]] = s["available"]

        # Same past-time filter as _status_for_slots: today's slot times that
        # have already gone are not really bookable, even if "available" is
        # still True in the file, so shown badges shouldn't claim they're free.
        now = _datetime_cls.now()
        is_today = self.selected_date == now.strftime("%Y-%m-%d")
        now_time = now.strftime("%H:%M")

        if not day_slots:
            tk.Label(self.body, text="No slots this day (weekend, or outside the schedule).",
                     font=self._small, bg=self.bg, fg="#999999").pack(anchor="w", pady=8)
            return

        for t in SLOT_TIMES:
            services = by_time.get(t)
            if not services:
                continue
            row = tk.Frame(self.body, bg=self.bg, pady=2)
            row.pack(fill="x")
            tk.Label(row, text=t, font=self._bold, bg=self.bg, fg="#222222",
                     width=6, anchor="w").pack(side="left")
            for svc_key, svc_label in SERVICES:
                avail = services.get(svc_key)
                if avail is None:
                    continue
                if is_today and t <= now_time:
                    avail = False  # time has passed, no longer actually bookable
                bg_c, fg_c = CAL_COLOURS["free" if avail else "full"]
                tk.Label(row, text=svc_label, font=self._small, bg=bg_c, fg=fg_c,
                         padx=6, pady=2).pack(side="left", padx=3)


class CallApp(tk.Tk):
    def __init__(self, family: str, whisper_model: str, cpu_url: str, bargein: bool = False,
                 beep: bool = False):
        super().__init__()
        self.title("SME Voice Assistant, offline demo")
        self.geometry("880x560")
        self.configure(bg="white")

        self.family = family
        self.whisper_model = whisper_model
        self.cpu_url = cpu_url
        self.bargein = bargein
        self.beep = beep  # off by default: greeting no longer promises a beep,
                          # see GREETING in src/inference.py. The stream-open
                          # race that used to swallow the first word is fixed
                          # independently of whether this tone plays.
        self._bargein_threshold = None
        self.pipeline = None
        self.tts = None
        self.stt = None

        self._q = queue.Queue()
        self._stop_event = threading.Event()
        self._call_thread = None

        self._build_widgets()
        self.after(100, self._poll_queue)
        self.after(50, self._startup_check)

        if _HAS_PSUTIL:
            self._proc = psutil.Process()
            # First call to cpu_percent() always returns a meaningless 0.0 /
            # garbage value (it measures the interval since the *previous*
            # call) - prime both counters now so the first real reading a
            # second later is accurate instead of showing 0%.
            psutil.cpu_percent(interval=None)
            self._proc.cpu_percent(interval=None)
            self.after(1000, self._update_cpu)

    def _build_widgets(self):
        bold = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        normal = tkfont.Font(family="Segoe UI", size=10)
        mono = tkfont.Font(family="Consolas", size=9)

        top = tk.Frame(self, bg="white", pady=10, padx=14)
        top.pack(fill="x")
        tk.Label(top, text="SME Voice Assistant", font=("Segoe UI", 16, "bold"),
                 bg="white", fg="#222222").pack(side="left")
        bargein_tag = "barge-in on" if self.bargein else "barge-in off"
        self.model_label = tk.Label(top, text=f"{self.family}  |  Whisper {self.whisper_model}  |  offline, CPU only  |  {bargein_tag}",
                                    font=normal, bg="white", fg="#666666")
        self.model_label.pack(side="left", padx=16)

        status_row = tk.Frame(self, bg="white", padx=14, pady=4)
        status_row.pack(fill="x")
        self.status_dot = tk.Canvas(status_row, width=16, height=16, bg="white", highlightthickness=0)
        self.status_dot.pack(side="left")
        self.dot = self.status_dot.create_oval(2, 2, 14, 14, fill=STATUS_COLOUR["Ready"], outline="")
        self.status_label = tk.Label(status_row, text="Ready", font=bold, bg="white", fg="#222222")
        self.status_label.pack(side="left", padx=8)

        # System CPU is the whole machine (covers the llama.cpp server,
        # which runs as a separate process); "this app" covers just this
        # Python process (mainly Whisper STT, which does run in-process).
        # Useful for the viva to show inference actually loading the CPU.
        cpu_text = "CPU: -- " if _HAS_PSUTIL else "CPU: psutil not installed"
        self.cpu_label = tk.Label(status_row, text=cpu_text, font=normal, bg="white", fg="#999999")
        self.cpu_label.pack(side="right")

        mid = tk.Frame(self, bg="white", padx=14)
        mid.pack(fill="both", expand=True)

        left = tk.Frame(mid, bg="white")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(left, text="Call transcript", font=bold, bg="white", anchor="w").pack(fill="x")
        self.transcript = tk.Text(left, font=mono, wrap="word", bg="#fafafa", relief="flat", padx=8, pady=8)
        self.transcript.pack(fill="both", expand=True)
        self.transcript.configure(state="disabled")

        right = tk.Frame(mid, bg="white", width=330)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)
        tk.Label(right, text="Calendar", font=bold, bg="white", anchor="w").pack(fill="x")
        self.calendar_panel = CalendarPanel(right, CALENDAR_PATH, bg="white")
        self.calendar_panel.pack(fill="both", expand=True, pady=(4, 0))

        btn_row = tk.Frame(self, bg="white", pady=12, padx=14)
        btn_row.pack(fill="x")
        self.call_btn = tk.Button(btn_row, text="Start Call", font=bold, bg="#1f9d55", fg="white",
                                  activebackground="#17803f", relief="flat", padx=18, pady=8,
                                  command=self._toggle_call)
        self.call_btn.pack(side="left")
        self.test_btn = tk.Button(btn_row, text="Test Call (scripted, no mic)", font=normal,
                                  bg="#eeeeee", fg="#222222", relief="flat", padx=14, pady=8,
                                  command=self._start_test_call)
        self.test_btn.pack(side="left", padx=10)

    def _startup_check(self):
        ok = check_cpu_server(self.cpu_url)
        if ok:
            self._log_transcript(f"[server] llama.cpp reachable at {self.cpu_url}\n")
        else:
            self._set_status("Error")
            self._log_transcript(
                f"[server] NOT reachable at {self.cpu_url}. Start it first:\n"
                f"  python scripts/03_cpu_server.py --model {self.family} --quant Q4_K_M\n")
            self.call_btn.configure(state="disabled")
            self.test_btn.configure(state="disabled")

    def _ensure_pipeline(self):
        if self.pipeline is None:
            self.pipeline = Pipeline(mode="cpu", model_family=self.family, cpu_url=self.cpu_url)
        if self.tts is None:
            self.tts = TTS()

    def _ensure_stt(self):
        if self.stt is None:
            from src.stt import STT
            self.stt = STT(model_size=self.whisper_model)

    def _set_status(self, label: str):
        self._q.put(("status", label))

    def _log_transcript(self, text: str):
        self._q.put(("transcript", text))

    def _update_cpu(self):
        try:
            sys_cpu = psutil.cpu_percent(interval=None)
            proc_cpu = self._proc.cpu_percent(interval=None) / (os.cpu_count() or 1)
            self.cpu_label.configure(text=f"CPU: {sys_cpu:.0f}% system, {proc_cpu:.0f}% this app")
        except Exception:
            pass
        self.after(1000, self._update_cpu)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "status":
                    self.status_label.configure(text=payload)
                    self.status_dot.itemconfig(self.dot, fill=STATUS_COLOUR.get(payload, "#9aa0a6"))
                elif kind == "transcript":
                    self.transcript.configure(state="normal")
                    self.transcript.insert("end", payload)
                    self.transcript.see("end")
                    self.transcript.configure(state="disabled")
                elif kind == "calendar_refresh":
                    # Runs on the main thread (this is the Tk event loop's own
                    # after() callback), so it's safe to touch the widget here
                    # even though the booking happened on a background thread.
                    self.calendar_panel.refresh()
                elif kind == "call_ended":
                    self.call_btn.configure(text="Start Call", bg="#1f9d55")
                    self.test_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _speak_result(self, result: dict):
        """Speak a pipeline result (greeting or turn reply), handling
        barge-in if enabled. Returns the caller's utterance if they
        interrupted mid-reply, else None. Shared by _run_turn and the
        call-opening greeting so both go through the same barge-in path.
        """
        self._set_status("Speaking")
        self._log_transcript(f"  spoken: {result['spoken']}\n")

        bargein_utterance = None
        if self.bargein and self._bargein_threshold is not None:
            interrupted, caught = speak_with_bargein(
                self.tts, self.stt, result["spoken"], self._bargein_threshold,
                call_stop_event=self._stop_event)
            if interrupted:
                self._log_transcript("  [caller interrupted]\n")
                if caught:
                    bargein_utterance = caught
        else:
            self.tts.speak(result["spoken"])

        self._q.put(("calendar_refresh", None))
        return bargein_utterance

    def _run_turn(self, utterance: str, session_id: str):
        """Run one turn. Returns (end_call, bargein_utterance).

        bargein_utterance is not None when the caller talked over the
        reply and the barge-in listener already captured what they said.
        The outer loop should feed that straight in as the next turn's
        input instead of calling listen_vad again - the caller is already
        mid-sentence, so waiting for a fresh prompt would just cut them
        off a second time.
        """
        self._set_status("Thinking")
        self._log_transcript(f"\n> {utterance}\n")
        result = self.pipeline.run(utterance, session_id=session_id)
        action = result.get("action")
        self._log_transcript(f"  action: {action}\n  latency: {result.get('latency_ms')}ms\n")
        bargein_utterance = self._speak_result(result)
        return bool(result.get("end_call")), bargein_utterance

    def _toggle_call(self):
        if self._call_thread and self._call_thread.is_alive():
            self._stop_event.set()
            self.call_btn.configure(text="Start Call", bg="#1f9d55")
            self._set_status("Call ended")
            return
        self._stop_event.clear()
        self.call_btn.configure(text="End Call", bg="#c62828")
        self.test_btn.configure(state="disabled")
        self._call_thread = threading.Thread(target=self._voice_call_worker, daemon=True)
        self._call_thread.start()

    def _voice_call_worker(self):
        try:
            self._ensure_pipeline()
            self._set_status("Listening")
            session_id = "call-ui-voice"
            self._log_transcript("\n[call] answering...\n")

            # Calibrate ambient noise FIRST, before anything else - no TTS
            # has played yet and no STT model load is running, so this is
            # the quietest and most reliable moment to measure the room's
            # true noise floor. Uses the module-level calibrate_ambient()
            # (sounddevice/numpy only), NOT self.stt.calibrate_ambient(),
            # specifically so it doesn't have to wait for the slow Whisper
            # model load below first.
            #
            # This value is then reused for every listen_vad() call for the
            # rest of the call. Previously, calibration only happened when
            # --bargein was on, AND even then the main listening loop below
            # never passed the calibrated value through to listen_vad() - so
            # every turn silently recalibrated itself instead, right after
            # TTS playback just finished. That recalibration window can
            # catch trailing room echo or the audio device still settling,
            # inflating the threshold well above the true room noise floor -
            # the most likely explanation for needing to raise your voice to
            # be heard at all, since the threshold could silently creep up
            # on any turn and stay high for the rest of the call.
            self._log_transcript("[call] calibrating room noise (stay quiet a moment)...\n")
            from src.stt import calibrate_ambient as _calibrate_ambient
            self._bargein_threshold = _calibrate_ambient()

            # Load Whisper on a background thread instead of blocking here -
            # loading tiny/small can take several seconds, and doing it
            # before anything is said means the caller hears total silence
            # for that whole stretch (the "nothing happens for ~10s" report).
            # It loads in parallel with the greeting being spoken below, so
            # by the time that finishes it's usually already done.
            stt_thread = threading.Thread(target=self._ensure_stt, daemon=True)
            stt_thread.start()

            greeting = self.pipeline.greet(session_id=session_id)
            pending_utterance = self._speak_result(greeting)
            stt_thread.join()  # make sure Whisper is ready before we try to listen
            while not self._stop_event.is_set():
                if pending_utterance:
                    utterance = pending_utterance
                    pending_utterance = None
                else:
                    self._set_status("Listening")
                    utterance = self.stt.listen_vad(silence_duration=1.2, max_duration=15.0,
                                                    stop_event=self._stop_event,
                                                    threshold=self._bargein_threshold,
                                                    beep=self.beep,
                                                    beep_log=self._log_transcript)
                if self._stop_event.is_set():
                    break
                if not utterance:
                    # listen_vad() returns "" for a few different reasons
                    # (nobody spoke before start_timeout, a stray noise too
                    # short to be real speech, or a transcription that came
                    # back blank) and previously this loop just silently
                    # re-listened with no visible sign anything happened.
                    # That's indistinguishable from the app being broken.
                    # This is not a fix for missed speech itself (still
                    # worth trying a lower calibration multiplier in
                    # STT.listen_vad if it keeps happening), just honesty
                    # about what occurred.
                    self._log_transcript("  (didn't catch that, listening again...)\n")
                    continue
                ended, pending_utterance = self._run_turn(utterance, session_id)
                if ended:
                    break
        except Exception as exc:
            self._log_transcript(f"\n[error] {type(exc).__name__}: {exc}\n")
        finally:
            self._set_status("Call ended")
            self._q.put(("call_ended", None))

    def _start_test_call(self):
        self.call_btn.configure(state="disabled")
        self.test_btn.configure(state="disabled")
        threading.Thread(target=self._test_call_worker, daemon=True).start()

    def _test_call_worker(self):
        try:
            self._ensure_pipeline()
            session_id = "call-ui-test"
            greeting = self.pipeline.greet(session_id=session_id)
            self._speak_result(greeting)
            for utterance in TEST_UTTERANCES:
                ended, _ = self._run_turn(utterance, session_id)
                if ended:
                    break
        except Exception as exc:
            self._log_transcript(f"\n[error] {type(exc).__name__}: {exc}\n")
        finally:
            self._set_status("Call ended")
            self.call_btn.configure(state="normal")
            self.test_btn.configure(state="normal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None,
                    choices=["phi3", "llama3", "llama1b", "qwen0.5b", "qwen1.5b", "smol360"],
                    help="If omitted, you'll be prompted to choose from the models that "
                         "have a GGUF built, and the server will be started automatically.")
    ap.add_argument("--whisper", default="tiny", choices=["tiny", "small"])
    ap.add_argument("--cpu-url", default="http://127.0.0.1:8080")
    ap.add_argument("--bargein", action="store_true",
                    help="let the caller interrupt the assistant mid-reply. "
                         "Energy-based, no echo cancellation - reliable only "
                         "with a headset, or low speaker volume near the mic. "
                         "See src/bargein.py for the caveat.")
    ap.add_argument("--beep", action="store_true",
                    help="play a short tone right when listening starts each turn. "
                         "Off by default: the greeting no longer says 'after the "
                         "beep' (read as voicemail-style, not a live receptionist), "
                         "and the fix for the missing-first-word bug was the "
                         "stream-open ordering in STT.listen_vad, not the tone "
                         "itself, so turning this off doesn't reintroduce it.")
    args = ap.parse_args()

    family = args.family
    server_proc = None
    _parsed = urllib.parse.urlparse(args.cpu_url)
    if _parsed.hostname in ("127.0.0.1", "localhost"):
        from src.model_server import prompt_for_family, ensure_server
        family = family or prompt_for_family()
        try:
            server_proc = ensure_server(family, port=_parsed.port or 8080)
        except RuntimeError as e:
            print(f"\n[model] {e}")
            raise SystemExit(1)
    elif family is None:
        # --cpu-url points somewhere else (e.g. a server already running on
        # another machine) - can't auto-launch a remote server, and there's
        # no local GGUF list to prompt from either. Fall back to the old
        # default rather than crashing.
        family = "qwen0.5b"

    try:
        app = CallApp(family=family, whisper_model=args.whisper, cpu_url=args.cpu_url,
                      bargein=args.bargein, beep=args.beep)
        app.mainloop()
    finally:
        if server_proc is not None:
            print("[model] stopping the llama.cpp server...")
            server_proc.terminate()
