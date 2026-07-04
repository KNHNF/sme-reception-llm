"""
Fetch real-audio recordings from Supabase Storage for the real-audio evaluation.

Storage layout (set by the voice-collector project):
  bucket "recordings"
    <speaker_id>/<filename>.webm|.mp4     the audio clip
    <speaker_id>/<filename>.json          metadata, including the ground-truth
                                          action_type and transcript

This downloads every clip and its metadata into evaluation/real_audio/, then
builds labels.csv (one row per clip) so the scoring script has the ground truth.

The recordings are personal data. evaluation/real_audio/ is gitignored and must
never be committed. Only aggregate scores (evaluation/real_audio_results/) are
safe to commit.

Credentials come from the environment, never hard-coded:
  SUPABASE_URL   defaults to the known project URL
  SUPABASE_KEY   service_role key, required (do not paste it into this file)

Usage:
  set SUPABASE_KEY=...              (PowerShell: $env:SUPABASE_KEY="...")
  python scripts/fetch_real_audio.py

Requires: pip install supabase
"""

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "evaluation" / "real_audio"
BUCKET = "recordings"

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://zxigjenkovfuuzbckuyh.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

AUDIO_EXTS = (".webm", ".mp4", ".wav", ".m4a", ".ogg")


def get_client():
    if not SUPABASE_KEY:
        print("ERROR: SUPABASE_KEY not set. Export the service_role key first:")
        print('  PowerShell:  $env:SUPABASE_KEY="<key>"')
        print("  Do NOT paste the key into this file.")
        sys.exit(1)
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: supabase package missing. Run: pip install supabase")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def list_all(client, prefix: str = "") -> list[str]:
    """Recursively list every object path in the bucket."""
    paths = []
    items = client.storage.from_(BUCKET).list(prefix, {"limit": 1000})
    for it in items:
        name = it.get("name")
        if not name:
            continue
        full = f"{prefix}/{name}" if prefix else name
        # Folders come back with id == None; recurse into them.
        if it.get("id") is None:
            paths.extend(list_all(client, full))
        else:
            paths.append(full)
    return paths


def main():
    client = get_client()
    print(f"Listing bucket '{BUCKET}' at {SUPABASE_URL} ...")
    paths = list_all(client)
    if not paths:
        print("No objects found. Check the bucket name and key.")
        return

    audio_paths = [p for p in paths if p.lower().endswith(AUDIO_EXTS)]
    json_paths = [p for p in paths if p.lower().endswith(".json")]
    print(f"Found {len(audio_paths)} audio clips and {len(json_paths)} metadata files.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for p in paths:
        dest = OUT_DIR / p
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = client.storage.from_(BUCKET).download(p)
        except Exception as exc:
            print(f"  skip {p}: {exc}")
            continue
        dest.write_bytes(data)
        downloaded += 1
    print(f"Downloaded {downloaded} new files into {OUT_DIR.relative_to(ROOT)}")

    # Build labels.csv from the metadata JSON files. Ground truth for scoring is
    # action_type; transcript is the reference for WER. IP is deliberately excluded.
    rows = []
    for jp in json_paths:
        local = OUT_DIR / jp
        if not local.exists():
            continue
        try:
            meta = json.loads(local.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "speaker_id":   meta.get("speaker_id", jp.split("/")[0]),
            "filename":     meta.get("filename", ""),
            "scenario_id":  meta.get("scenario_id", ""),
            "action_type":  meta.get("action_type", ""),
            "difficulty":   meta.get("difficulty", ""),
            "take_number":  meta.get("take_number", ""),
            "transcript":   meta.get("transcript", ""),
            "age_range":    meta.get("age_range", ""),
            "first_language": meta.get("first_language", ""),
            "duration_seconds": meta.get("duration_seconds", ""),
        })

    if rows:
        labels_path = OUT_DIR / "labels.csv"
        fields = list(rows[0].keys())
        with labels_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {labels_path.relative_to(ROOT)}")

        speakers = sorted({r["speaker_id"] for r in rows})
        actions = {}
        for r in rows:
            actions[r["action_type"]] = actions.get(r["action_type"], 0) + 1
        print(f"\nSpeakers ({len(speakers)}): {', '.join(speakers)}")
        print("Action distribution:")
        for a, n in sorted(actions.items(), key=lambda x: -x[1]):
            print(f"  {a or '(none)':<20} {n}")
    else:
        print("No metadata JSON found, labels.csv not written. "
              "Clips still downloaded, but you will need labels to score them.")


if __name__ == "__main__":
    main()
