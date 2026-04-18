#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List


REQUIRED_KEYS = ["state", "action", "reward", "done", "next_state"]


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_contract(rows: List[dict]) -> Dict[str, int]:
    missing = {k: 0 for k in REQUIRED_KEYS}
    for item in rows:
        for key in REQUIRED_KEYS:
            if key not in item:
                missing[key] += 1
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="CQL scaffold for Offline RL on dodge dataset.")
    parser.add_argument("--dataset", default="output/ml/datasets/dodge_bc_v1.jsonl")
    parser.add_argument("--out-report", default="output/ml/reports/cql_scaffold_report.json")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    if not rows:
        raise RuntimeError("dataset is empty")
    missing = validate_contract(rows)
    report = {
        "dataset_path": args.dataset,
        "rows": len(rows),
        "required_keys": REQUIRED_KEYS,
        "missing_counts": missing,
        "ready_for_cql": all(v == 0 for v in missing.values()),
        "note": "CQL training pipeline is intentionally scaffold-only in V1. "
                "Next iteration should add replay buffer + Q-network + conservative loss.",
    }
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[CQL-SCAFFOLD]", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
