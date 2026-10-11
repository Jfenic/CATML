#!/usr/bin/env python3
"""CATML Benchmark v1 entrypoint (pilot / dry-run)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _pilot(dry_run: bool, out_dir: Path) -> dict:
    from sklearn.datasets import load_breast_cancer

    from benchmarks.frameworks.catml_adapter import run_catml_fit
    from benchmarks.frameworks.sklearn_adapter import run_sklearn_baseline

    df = load_breast_cancer(as_frame=True).frame
    rows = []
    if dry_run:
        rows.append(
            {
                "framework": "dry_run",
                "status": "skipped",
                "primary_score": None,
                "wall_time_s": 0.0,
                "note": "protocol check only",
            }
        )
    else:
        sk = run_sklearn_baseline(df, "target")
        rows.append(
            {
                "framework": "sklearn_baseline",
                "status": sk.status,
                "primary_score": sk.primary_score,
                "wall_time_s": sk.wall_time_s,
                "failure_reason": sk.failure_reason,
            }
        )
        cat = run_catml_fit(df, "target", workspace_dir=out_dir / "ws_catml")
        rows.append(
            {
                "framework": "catml",
                "status": cat.status,
                "primary_score": cat.primary_score,
                "wall_time_s": cat.wall_time_s,
                "failure_reason": cat.failure_reason,
                "model_id": cat.model_id,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "catml-benchmark-v1",
        "study": "pilot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    (out_dir / "pilot-raw.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="CATML Benchmark v1 runner")
    parser.add_argument("--study", choices=["pilot", "C"], default="pilot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "v1",
    )
    args = parser.parse_args()

    if args.study == "C":
        print("Study C: run `pytest tests/test_benchmark_study_c.py`")
        return 0

    payload = _pilot(dry_run=args.dry_run, out_dir=args.out)
    print(json.dumps(payload, indent=2))
    failed = [r for r in payload["rows"] if r.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
