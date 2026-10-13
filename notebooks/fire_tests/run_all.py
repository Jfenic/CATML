#!/usr/bin/env python3
"""Run all fire tests and write an aggregate summary JSON."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SCRIPTS = [
    "01_titanic_like.py",
    "02_house_prices_like.py",
    "03_credit_risk.py",
    "04_multiclass.py",
    "05_high_cardinality.py",
    "06_ordered_time_split.py",
    "07_hpo_short.py",
    "08_export_predict.py",
    "09_explore_smoke.py",
    "10_ensemble_compare.py",
]


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    py = sys.executable
    for script in SCRIPTS:
        path = HERE / script
        print(f"=== {script} ===", flush=True)
        proc = subprocess.run(
            [py, str(path)],
            cwd=str(HERE),
            capture_output=True,
            text=True,
        )
        test_id = path.stem
        result_path = RESULTS / f"{test_id}.json"
        entry: dict = {
            "test_id": test_id,
            "exit_code": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-500:],
            "stderr_tail": (proc.stderr or "")[-500:],
        }
        if result_path.is_file():
            entry["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            entry["result"] = {
                "test_id": test_id,
                "status": "failed",
                "detail": "no result json written",
                "traceback": proc.stderr,
            }
        rows.append(entry)
        status = entry["result"].get("status")
        print(f"→ {status} (exit {proc.returncode})", flush=True)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_tests": len(rows),
        "n_ok": sum(1 for r in rows if r.get("result", {}).get("status") == "ok"),
        "n_failed": sum(1 for r in rows if r.get("result", {}).get("status") != "ok"),
        "rows": rows,
    }
    out = RESULTS / "fire_tests_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out} — ok={summary['n_ok']} failed={summary['n_failed']}")
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
