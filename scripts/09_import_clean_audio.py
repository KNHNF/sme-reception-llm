"""
Clean and import the collected voice recordings into evaluation/real_audio/.

Problem this solves: some speaker folders in the voice-collector download
contain more than one real person (a shared phone, a reused link), so folder
name is not a reliable speaker identity. This tool splits folders into
sessions (gaps over 20 minutes between recordings start a new session),
proposes one speaker ID per session, and rebuilds the eval layout only after
a human confirms the mapping.

Two steps:

  1. python scripts/09_import_clean_audio.py --propose
     Writes mapping_proposal.csv next to the recordings. One row per clip:
     session, proposed speaker ID, keep/drop, reason, quality_flag.
     REVIEW AND EDIT THIS FILE BY HAND (only the assigned_speaker and action
     columns), then

  2. python scripts/09_import_clean_audio.py --apply
     Rebuilds evaluation/real_audio/: per-speaker folders, renamed clips,
     labels.csv and metadata.csv regenerated from the sidecar JSONs.
     Use --clean to delete existing SP* folders in the output first
     (recommended: single source of truth).

The recordings are personal data: evaluation/real_audio/ is gitignored and
must never be committed. Only aggregate scores are committed.

After a successful --apply, re-run the eval per model:
  python scripts/03_cpu_server.py --model <family> --quant Q4_K_M
  python scripts/08_real_audio_eval.py --family <family>
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_SRC = Path(r"E:\Coding\private-projects\CallFlow\voice-collector\backend\downloaded_recordings")
OUT_DIR = ROOT / "evaluation" / "real_audio"
GAP_MINUTES = 20

# Proposed identity per (folder, session index), from the 2026-07-10 audit:
# demographics + IP + timestamp clusters. Sessions not listed keep their
# folder ID. Karan confirms or corrects in the CSV before --apply.
PROPOSAL = {
    ("SP001", 1): ("SP001", "keep", "30 Jun 20:25, matches original SP001 demographics in old metadata"),
    ("SP001", 2): ("SP001", "keep", "30 Jun 21:07, the original SP001 session (25-34, Farsi)"),
    ("SP001", 3): ("SP005", "keep", "6 Jul 05:30, different person (18-24 Male Arabic, different IP)"),
    ("SP001", 4): ("DROP", "drop", "7 Jul 14:52, single clip, unknown identity (campus IP, mid class session)"),
    ("SP001", 5): ("SP006", "keep", "8 Jul, same person as the 9 Jul session (55-64 Female Farsi)"),
    ("SP001", 6): ("SP006", "keep", "9 Jul, 55-64 Female Farsi, distinct from original SP001"),
    ("SP003", 1): ("SP003", "keep", "7 Jul 14:03 session, first user of the shared phone"),
    ("SP003", 2): ("SP007", "keep", "7 Jul 15:12 session, classmate on the same phone (different campus IP)"),
}

LABEL_FIELDS = ["speaker_id", "filename", "scenario_id", "action_type", "difficulty",
                "take_number", "transcript", "age_range", "first_language", "duration_seconds"]
META_FIELDS = ["speaker_id", "scenario_id", "action_type", "difficulty", "take_number",
               "filename", "transcript", "transcript_word_count", "age_range", "first_language",
               "device_type", "environment", "timestamp", "file_size_bytes", "ip", "quality_flag"]


def parse_ts(name: str) -> datetime:
    return datetime.strptime(name.rsplit("_", 1)[-1].replace(".webm", ""), "%Y%m%dT%H%M%S")


def scan(src: Path) -> list[dict]:
    clips = []
    for spdir in sorted(p for p in src.iterdir() if p.is_dir()):
        folder_clips = sorted(spdir.glob("*.webm"), key=lambda p: parse_ts(p.stem))
        session, prev = 0, None
        for w in folder_clips:
            t = parse_ts(w.stem)
            if prev is None or (t - prev).total_seconds() > GAP_MINUTES * 60:
                session += 1
            prev = t
            j = w.with_suffix(".json")
            meta = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
            clips.append({"folder": spdir.name, "session": session, "path": w,
                          "has_json": j.exists(), "meta": meta})
    return clips


def propose(src: Path, out_csv: Path) -> None:
    clips = scan(src)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["folder", "session", "filename", "assigned_speaker", "action",
                     "quality_flag", "has_json", "reason"])
        for c in clips:
            target, action, reason = PROPOSAL.get((c["folder"], c["session"]),
                                                  (c["folder"], "keep", "single-session folder, identity unchanged"))
            flag = c["meta"].get("quality_flag", "no-json")
            wr.writerow([c["folder"], c["session"], c["path"].name, target, action,
                         flag, c["has_json"], reason])
    n = len(clips)
    print(f"Wrote {out_csv} ({n} clips). Edit assigned_speaker/action, then run --apply.")
    print("Flags to review by listening: any row with quality_flag 'review' or 'mismatch'.")


def apply(src: Path, mapping_csv: Path, out: Path, clean: bool, fallback_meta: Path | None) -> None:
    fallback = {}
    if fallback_meta and fallback_meta.exists():
        with fallback_meta.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fallback[row["filename"]] = row

    rows = list(csv.DictReader(mapping_csv.open(encoding="utf-8")))
    if clean:
        for d in out.glob("SP*"):
            if d.is_dir():
                shutil.rmtree(d)
    labels, metas, skipped = [], [], []
    for r in rows:
        if r["action"].strip().lower() != "keep" or r["assigned_speaker"] == "DROP":
            continue
        sp = r["assigned_speaker"].strip()
        src_file = src / r["folder"] / r["filename"]
        if not src_file.exists():
            skipped.append((r["filename"], "source file missing"))
            continue
        new_name = sp + r["filename"][len(r["folder"]):] if r["filename"].startswith(r["folder"]) else f"{sp}_{r['filename']}"
        dest_dir = out / sp
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_dir / new_name)
        j = src_file.with_suffix(".json")
        meta = json.loads(j.read_text(encoding="utf-8")) if j.exists() else dict(fallback.get(r["filename"], {}))
        if not meta:
            skipped.append((r["filename"], "no sidecar json and no fallback metadata row; copied audio, no label"))
            continue
        meta["speaker_id"] = sp
        meta["filename"] = new_name
        labels.append({k: meta.get(k, "") for k in LABEL_FIELDS})
        metas.append({k: meta.get(k, "") for k in META_FIELDS})

    for name, fields, data in (("labels.csv", LABEL_FIELDS, labels), ("metadata.csv", META_FIELDS, metas)):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(data)
    speakers = sorted({l["speaker_id"] for l in labels})
    print(f"Imported {len(labels)} labelled clips across {len(speakers)} speakers: {', '.join(speakers)}")
    for fn, why in skipped:
        print(f"  SKIPPED {fn}: {why}")
    print(f"Rebuilt {out / 'labels.csv'} and metadata.csv. Never commit this folder.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--mapping", type=Path, default=None, help="defaults to <src>/mapping_proposal.csv")
    ap.add_argument("--propose", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--clean", action="store_true", help="delete existing SP* folders in --out before import")
    ap.add_argument("--fallback-metadata", type=Path, default=OUT_DIR / "metadata.csv",
                    help="old metadata.csv used to label clips that have no sidecar json")
    args = ap.parse_args()
    mapping = args.mapping or (args.src / "mapping_proposal.csv")
    if args.propose == args.apply:
        ap.error("pass exactly one of --propose or --apply")
    if args.propose:
        propose(args.src, mapping)
    else:
        if not mapping.exists():
            sys.exit(f"mapping file not found: {mapping} (run --propose first)")
        apply(args.src, mapping, args.out, args.clean, args.fallback_metadata)


if __name__ == "__main__":
    main()
