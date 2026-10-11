#!/usr/bin/env python3
"""Aggregate pilot/campaign raw JSON into summary + failure tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def analyze(raw_path: Path, out_dir: Path) -> None:
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"
    failures_path = out_dir / "failures.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["framework", "status", "primary_score", "wall_time_s"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "framework": row.get("framework"),
                    "status": row.get("status"),
                    "primary_score": row.get("primary_score"),
                    "wall_time_s": row.get("wall_time_s"),
                }
            )

    with failures_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["framework", "failure_reason"])
        writer.writeheader()
        for row in rows:
            if row.get("status") in {"failed", "skipped"} and row.get("failure_reason"):
                writer.writerow(
                    {
                        "framework": row.get("framework"),
                        "failure_reason": row.get("failure_reason"),
                    }
                )

    print(f"Wrote {summary_path} and {failures_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "v1" / "pilot-raw.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "v1",
    )
    args = parser.parse_args()
    if not args.raw.is_file():
        print(f"Missing raw results: {args.raw}")
        return 1
    analyze(args.raw, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
