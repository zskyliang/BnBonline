#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os
from typing import Dict, Iterable


def canonical_key(row: Dict) -> str:
    state = row.get("state") or {}
    next_state = row.get("next_state") or None
    payload = {
        "state_map": state.get("state_map"),
        "state_vector": state.get("state_vector"),
        "action": int(row.get("action", 0)),
        "action_mask": row.get("action_mask"),
        "done": bool(row.get("done", False)),
        "pre_death": bool(row.get("pre_death", False)),
        "outcome_tag": str(row.get("outcome_tag", "ongoing")),
        "next_state_map": (next_state or {}).get("state_map"),
        "next_state_vector": (next_state or {}).get("state_vector"),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def iter_rows(paths: Iterable[str]):
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield p, json.loads(line)
                except Exception:
                    continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=200000)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"no files matched: {args.input_glob}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    seen = set()
    wrote = 0
    scanned = 0
    dup = 0
    by_file = {}

    with open(args.output, "w", encoding="utf-8") as out:
        for src, row in iter_rows(paths):
            scanned += 1
            key = canonical_key(row)
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            wrote += 1
            by_file[src] = by_file.get(src, 0) + 1
            if wrote >= args.limit:
                break

    report = {
        "input_files": paths,
        "scanned_rows": scanned,
        "written_rows": wrote,
        "duplicate_rows": dup,
        "unique_rate": wrote / max(1, scanned),
        "output": os.path.abspath(args.output),
        "per_file_kept": by_file,
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
