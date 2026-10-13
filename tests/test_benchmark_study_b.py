"""Study B smoke: dry-run scaffold + one-dataset train-only Explore path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run import _study_b_campaign, _study_b_scaffold


def test_study_b_dry_run_scaffold(tmp_path: Path):
    payload = _study_b_scaffold(dry_run=True, out_dir=tmp_path)
    assert payload["study"] == "B"
    assert "catml_explore" in payload["configurations"]
    assert (tmp_path / "study-b-scaffold.json").exists()


def test_study_b_smoke_breast_cancer(tmp_path: Path):
    payload = _study_b_campaign(
        dry_run=False,
        out_dir=tmp_path,
        time_budget_s=30.0,
        seeds=[42],
        dataset_filter=["sklearn_breast_cancer"],
    )
    assert payload["study"] == "B"
    assert payload["holdout_policy"] == "never_seen_during_search"
    by_fw = {r["framework"]: r for r in payload["rows"]}
    assert by_fw["sklearn_baseline"]["status"] == "ok"
    assert by_fw["catml_base"]["status"] == "ok"
    assert by_fw["catml_explore"]["status"] == "ok"
    assert by_fw["catml_agent"]["status"] == "skipped"
    explore_extra = by_fw["catml_explore"].get("extra") or {}
    assert explore_extra.get("findings_count", 0) >= 0
    assert explore_extra.get("study_id")
    raw = json.loads((tmp_path / "study-b-raw.json").read_text())
    assert raw["comparison_complete"] is True
