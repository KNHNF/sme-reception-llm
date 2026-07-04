"""
Step 8: Real-audio evaluation.

Runs the downloaded real recordings through the deployed path: Faster-Whisper STT
then the fine-tuned LLM, and scores the predicted action against the human label
(action_type in labels.csv) and the transcription against the reference transcript.

Why it calls the model directly instead of Pipeline.run(): the live pipeline is
conversational (turn 0 greets and asks for the caller's name), so a single clip
fed to a fresh session returns the name prompt, not an action. The synthetic evals
(06_aligned_eval.py) scored by prompting the model directly. This does the same via
build_prompt(), so real-audio numbers are comparable to the synthetic ones.

Prerequisites:
  1. python scripts/fetch_real_audio.py   (clips + labels.csv under evaluation/real_audio/)
  2. For --mode cpu: start the server first, e.g.
     python scripts/03_cpu_server.py --model llama3 --quant Q4_K_M
  3. pip install faster-whisper

Usage:
  python scripts/08_real_audio_eval.py                       # cpu llama3, whisper tiny
  python scripts/08_real_audio_eval.py --whisper small
  python scripts/08_real_audio_eval.py --mode mock           # dry run, no model server

Outputs:
  evaluation/real_audio/real_audio_detail.json      per clip, gitignored (has transcripts)
  evaluation/real_audio_results/real_audio_summary.json   aggregate only, safe to commit
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

AUDIO_DIR = ROOT / "evaluation" / "real_audio"
LABELS = AUDIO_DIR / "labels.csv"
DETAIL_OUT = AUDIO_DIR / "real_audio_detail.json"          # gitignored
SUMMARY_DIR = ROOT / "evaluation" / "real_audio_results"   # committable
SUPPORTED = {"book_appointment", "check_availability",
             "cancel_appointment", "clarify", "out_of_scope"}


def normalise(text: str) -> list[str]:
    keep = []
    for ch in text.lower():
        keep.append(ch if ch.isalnum() or ch.isspace() else " ")
    return "".join(keep).split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalise(ref), normalise(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # Levenshtein over words
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cost = 0 if rw == hw else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(r)


def load_labels() -> list[dict]:
    if not LABELS.exists():
        print(f"ERROR: {LABELS} not found. Run scripts/fetch_real_audio.py first.")
        sys.exit(1)
    with LABELS.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_audio(row: dict) -> Path | None:
    speaker = row.get("speaker_id", "")
    fname = row.get("filename", "")
    cand = AUDIO_DIR / speaker / fname
    if cand.exists():
        return cand
    matches = list(AUDIO_DIR.rglob(fname)) if fname else []
    return matches[0] if matches else None


def load_whisper(size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


STT_DOMAIN_PROMPT = (
    "Appointment booking call for a clinic. Terms: book, cancel, reschedule, "
    "appointment, consultation, follow-up, availability, "
    "Monday Tuesday Wednesday Thursday Friday, morning, afternoon."
)


def transcribe(model, path: Path) -> str:
    segments, _ = model.transcribe(
        str(path), language="en", vad_filter=True,
        initial_prompt=STT_DOMAIN_PROMPT,
    )
    return " ".join(s.text for s in segments).strip()


def predict_action(text: str, mode: str, family: str, cpu_url: str) -> tuple[dict | None, str]:
    """Return (parsed_json_or_None, raw_text). Mirrors the pipeline LLM step."""
    from src.inference import build_prompt, parse_llm_output
    try:
        from src.entity_extractor import extract
        entities = extract(text)
    except Exception:
        entities = {}
    prompt = build_prompt(text, entities, None, family)

    if mode == "mock":
        from src.inference import Pipeline
        raw = Pipeline(mode="mock", model_family=family)._mock_output(text, entities, None)
        return parse_llm_output(raw), raw

    import urllib.request
    stop = ["<|eot_id|>"] if family == "llama3" else ["<|end|>"]
    payload = json.dumps({"prompt": prompt, "n_predict": 40,
                          "temperature": 0, "stop": stop, "stream": False}).encode()
    req = urllib.request.Request(f"{cpu_url}/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = (json.loads(resp.read()).get("content") or "").strip()
    return parse_llm_output(raw), raw


def check_cpu(cpu_url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{cpu_url}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu", "mock"], default="cpu")
    ap.add_argument("--family", choices=["llama3", "phi3"], default="llama3")
    ap.add_argument("--whisper", default="tiny", help="Faster-Whisper size (tiny matches deployment)")
    ap.add_argument("--cpu-url", default="http://127.0.0.1:8080")
    args = ap.parse_args()

    if args.mode == "cpu" and not check_cpu(args.cpu_url):
        print(f"ERROR: no llama.cpp server at {args.cpu_url}.")
        print("Start it: python scripts/03_cpu_server.py --model llama3 --quant Q4_K_M")
        sys.exit(1)

    rows = load_labels()
    print(f"Loaded {len(rows)} labelled clips. Loading Whisper '{args.whisper}'...")
    model = load_whisper(args.whisper)

    results = []
    for i, row in enumerate(rows, 1):
        audio = find_audio(row)
        expected = (row.get("action_type") or "").strip()
        ref = row.get("transcript", "")
        if audio is None:
            print(f"{i:>3}  MISSING audio for {row.get('filename')}")
            results.append({"filename": row.get("filename"), "error": "audio_missing"})
            continue
        try:
            hyp = transcribe(model, audio)
        except Exception as exc:
            print(f"{i:>3}  STT FAILED {audio.name}: {exc}")
            results.append({"filename": row.get("filename"), "error": f"stt:{exc}"})
            continue

        parsed, raw = predict_action(hyp, args.mode, args.family, args.cpu_url)
        predicted = (parsed or {}).get("action")
        json_valid = parsed is not None
        strict_ok = predicted == expected
        # Scope-aware: if the labelled action is not something the system supports,
        # answering out_of_scope or clarify counts as correct scope awareness.
        if expected in SUPPORTED:
            scope_ok = strict_ok
        else:
            scope_ok = predicted in {"out_of_scope", "clarify"}
        clip_wer = wer(ref, hyp)
        results.append({
            "filename": row.get("filename"), "speaker_id": row.get("speaker_id"),
            "difficulty": row.get("difficulty"), "expected": expected,
            "predicted": predicted, "json_valid": json_valid,
            "strict_correct": strict_ok, "scope_correct": scope_ok,
            "wer": round(clip_wer, 3), "reference": ref, "hypothesis": hyp,
        })
        mark = "ok " if strict_ok else "MISS"
        print(f"{i:>3}  {mark}  exp={expected:<18} got={str(predicted):<18} wer={clip_wer:.2f}")

    scored = [r for r in results if "error" not in r]
    n = len(scored)
    if not n:
        print("No clips scored.")
        return
    strict = sum(r["strict_correct"] for r in scored) / n
    scope = sum(r["scope_correct"] for r in scored) / n
    json_v = sum(r["json_valid"] for r in scored) / n
    mean_wer = sum(r["wer"] for r in scored) / n

    by_diff = {}
    for r in scored:
        d = r["difficulty"] or "unknown"
        by_diff.setdefault(d, []).append(r["strict_correct"])
    diff_acc = {d: round(sum(v) / len(v) * 100, 1) for d, v in by_diff.items()}

    summary = {
        "harness": "real_audio", "timestamp": datetime.now().isoformat(),
        "mode": args.mode, "family": args.family, "whisper": args.whisper,
        "n_clips": n, "n_errors": len(results) - n,
        "n_speakers": len({r["speaker_id"] for r in scored}),
        "action_accuracy_strict": round(strict * 100, 1),
        "action_accuracy_scope_aware": round(scope * 100, 1),
        "json_valid": round(json_v * 100, 1),
        "mean_wer": round(mean_wer * 100, 1),
        "accuracy_by_difficulty": diff_acc,
    }

    print("\n=== Real-audio summary ===")
    print(f"  Clips scored     : {n}  ({summary['n_errors']} errors, "
          f"{summary['n_speakers']} speakers)")
    print(f"  Action accuracy  : {summary['action_accuracy_strict']}% strict, "
          f"{summary['action_accuracy_scope_aware']}% scope-aware")
    print(f"  JSON valid       : {summary['json_valid']}%")
    print(f"  Mean WER         : {summary['mean_wer']}%")
    print(f"  By difficulty    : {diff_acc}")

    DETAIL_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "real_audio_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDetail (gitignored): {DETAIL_OUT.relative_to(ROOT)}")
    print(f"Summary (committable): {(SUMMARY_DIR / 'real_audio_summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
