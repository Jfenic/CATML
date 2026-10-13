#!/usr/bin/env python3
"""Aggregate pilot/Study A raw JSON into summary, aggregate, and failure tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _pstdev(xs: list[float]) -> float | None:
    if len(xs) < 1:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def analyze(raw_path: Path, out_dir: Path) -> None:
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"
    failures_path = out_dir / "failures.csv"
    aggregate_path = out_dir / "aggregate.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset_id",
                "task",
                "metric",
                "framework",
                "seed",
                "status",
                "primary_score",
                "wall_time_s",
                "peak_rss_mb",
                "cpu_user_s",
                "framework_version",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset_id": row.get("dataset_id"),
                    "task": row.get("task"),
                    "metric": row.get("metric"),
                    "framework": row.get("framework"),
                    "seed": row.get("seed"),
                    "status": row.get("status"),
                    "primary_score": row.get("primary_score"),
                    "wall_time_s": row.get("wall_time_s"),
                    "peak_rss_mb": row.get("peak_rss_mb"),
                    "cpu_user_s": row.get("cpu_user_s"),
                    "framework_version": row.get("framework_version"),
                }
            )

    with failures_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset_id", "framework", "seed", "status", "failure_reason"],
        )
        writer.writeheader()
        for row in rows:
            if row.get("status") in {"failed", "skipped"} and row.get("failure_reason"):
                writer.writerow(
                    {
                        "dataset_id": row.get("dataset_id"),
                        "framework": row.get("framework"),
                        "seed": row.get("seed"),
                        "status": row.get("status"),
                        "failure_reason": row.get("failure_reason"),
                    }
                )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok" or row.get("primary_score") is None:
            continue
        key = (
            row.get("dataset_id"),
            row.get("task"),
            row.get("metric"),
            row.get("framework"),
        )
        groups[key].append(row)

    with aggregate_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset_id",
                "task",
                "metric",
                "framework",
                "n",
                "mean_score",
                "std_score",
                "mean_wall_s",
                "mean_peak_rss_mb",
                "failure_rate_note",
            ],
        )
        writer.writeheader()
        for key in sorted(groups.keys(), key=lambda k: (str(k[0]), str(k[3]))):
            ds, task, metric, fw = key
            chunk = groups[key]
            scores = [float(r["primary_score"]) for r in chunk]
            walls = [float(r.get("wall_time_s") or 0.0) for r in chunk]
            rss = [float(r["peak_rss_mb"]) for r in chunk if r.get("peak_rss_mb") is not None]
            total_attempts = sum(
                1
                for r in rows
                if r.get("dataset_id") == ds and r.get("framework") == fw
            )
            n_failed = sum(
                1
                for r in rows
                if r.get("dataset_id") == ds
                and r.get("framework") == fw
                and r.get("status") == "failed"
            )
            writer.writerow(
                {
                    "dataset_id": ds,
                    "task": task,
                    "metric": metric,
                    "framework": fw,
                    "n": len(scores),
                    "mean_score": _mean(scores),
                    "std_score": _pstdev(scores),
                    "mean_wall_s": _mean(walls),
                    "mean_peak_rss_mb": _mean(rss),
                    "failure_rate_note": f"{n_failed}/{total_attempts} failed",
                }
            )

    print(f"Wrote {summary_path}, {aggregate_path}, and {failures_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=["pilot", "A", "auto"],
        default="auto",
        help="Which raw file to aggregate (auto prefers study-a-raw.json if present)",
    )
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "v1",
    )
    args = parser.parse_args()
    if args.raw is not None:
        raw = args.raw
    elif args.study == "A":
        raw = args.out / "study-a-raw.json"
    elif args.study == "pilot":
        raw = args.out / "pilot-raw.json"
    else:
        candidate = args.out / "study-a-raw.json"
        raw = candidate if candidate.is_file() else args.out / "pilot-raw.json"
    if not raw.is_file():
        print(f"Missing raw results: {raw}")
        return 1
    analyze(raw, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
