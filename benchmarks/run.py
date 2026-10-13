#!/usr/bin/env python3
"""CATML Benchmark v1 entrypoint — shared external holdout (pilot / Study A / Study B scaffold)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

ANNOUNCED_FRAMEWORKS = ("sklearn_baseline", "catml", "flaml", "autogluon")


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return round(kb / (1024 * 1024), 2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("sklearn", "pandas", "numpy", "flaml", "catml"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            versions[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[name] = None
    try:
        import autogluon

        versions["autogluon"] = getattr(autogluon, "__version__", "installed")
    except ImportError:
        versions["autogluon"] = None
    return versions


def _environment() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "os": platform.system(),
        "cpu_model": _cpu_model(),
        "ram_gb": _ram_gb(),
        "catml_git_commit": _git_commit(),
        "dependency_versions": _dependency_versions(),
    }


def _load_protocol() -> dict:
    path = ROOT / "benchmarks" / "protocol-v1.yaml"
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        # Minimal fallback without PyYAML
        text = path.read_text(encoding="utf-8")
        status = "frozen" if "status: frozen" in text else "draft"
        version = "1.0.0"
        for line in text.splitlines():
            if line.strip().startswith("version:"):
                version = line.split(":", 1)[1].strip().strip('"')
        return {"status": status, "version": version}


def _append_from_adapter(
    rows: list[dict],
    *,
    ds,
    seed: int,
    framework: str,
    status: str,
    y_pred,
    y_proba,
    wall_time_s: float,
    peak_rss_mb: float | None = None,
    cpu_user_s: float | None = None,
    cpu_system_s: float | None = None,
    failure_reason: str | None = None,
    model_id: str | None = None,
    note: str | None = None,
    holdout_test,
) -> None:
    from benchmarks.frameworks.eval import framework_version, result_dict, score_external

    primary = None
    secondary: dict = {}
    if status == "ok":
        scores = score_external(
            task=ds.task,
            metric=ds.metric,
            y_true=holdout_test[ds.target],
            y_pred=y_pred,
            y_proba=y_proba,
        )
        primary = scores.primary
        secondary = scores.secondary
    rows.append(
        result_dict(
            dataset_id=ds.dataset_id,
            task=ds.task,
            metric=ds.metric,
            framework=framework,
            seed=seed,
            status=status,
            primary_score=primary,
            wall_time_s=wall_time_s,
            peak_rss_mb=peak_rss_mb,
            cpu_user_s=cpu_user_s,
            cpu_system_s=cpu_system_s,
            secondary=secondary,
            failure_reason=failure_reason,
            model_id=model_id,
            note=note,
            framework_version=framework_version(framework),
        )
    )


def _run_frameworks_on_holdout(
    *,
    ds,
    holdout,
    seed: int,
    out_dir: Path,
    time_budget_s: float,
    frameworks: tuple[str, ...],
    rows: list[dict],
) -> None:
    from benchmarks.frameworks.autogluon_adapter import run_autogluon
    from benchmarks.frameworks.catml_adapter import run_catml_fit
    from benchmarks.frameworks.flaml_adapter import run_flaml
    from benchmarks.frameworks.sklearn_adapter import run_sklearn_baseline

    for fw in frameworks:
        if fw == "sklearn_baseline":
            result = run_sklearn_baseline(
                holdout.train, holdout.test, ds.target, task=ds.task
            )
            _append_from_adapter(
                rows,
                ds=ds,
                seed=seed,
                framework=fw,
                status=result.status,
                y_pred=result.y_pred,
                y_proba=result.y_proba,
                wall_time_s=result.wall_time_s,
                peak_rss_mb=result.peak_rss_mb,
                cpu_user_s=result.cpu_user_s,
                cpu_system_s=result.cpu_system_s,
                failure_reason=result.failure_reason,
                holdout_test=holdout.test,
            )
        elif fw == "catml":
            cat = run_catml_fit(
                holdout.train,
                holdout.test,
                ds.target,
                metric=ds.metric,
                workspace_dir=out_dir / "ws_catml" / ds.dataset_id / f"seed{seed}",
                time_budget=time_budget_s,
                random_state=seed,
            )
            _append_from_adapter(
                rows,
                ds=ds,
                seed=seed,
                framework=fw,
                status=cat.status,
                y_pred=cat.y_pred,
                y_proba=cat.y_proba,
                wall_time_s=cat.wall_time_s,
                peak_rss_mb=cat.peak_rss_mb,
                cpu_user_s=cat.cpu_user_s,
                cpu_system_s=cat.cpu_system_s,
                failure_reason=cat.failure_reason,
                model_id=cat.model_id,
                holdout_test=holdout.test,
            )
        elif fw == "flaml":
            fl = run_flaml(
                holdout.train,
                holdout.test,
                ds.target,
                task=ds.task,
                metric=ds.metric,
                time_budget=time_budget_s,
                random_state=seed,
            )
            _append_from_adapter(
                rows,
                ds=ds,
                seed=seed,
                framework=fw,
                status=fl.status,
                y_pred=fl.y_pred,
                y_proba=fl.y_proba,
                wall_time_s=fl.wall_time_s,
                peak_rss_mb=fl.peak_rss_mb,
                cpu_user_s=fl.cpu_user_s,
                cpu_system_s=fl.cpu_system_s,
                failure_reason=fl.failure_reason,
                holdout_test=holdout.test,
            )
        elif fw == "autogluon":
            ag = run_autogluon(
                holdout.train,
                holdout.test,
                ds.target,
                task=ds.task,
                metric=ds.metric,
                time_budget=time_budget_s,
                work_dir=out_dir / "ws_autogluon" / ds.dataset_id / f"seed{seed}",
                random_state=seed,
            )
            _append_from_adapter(
                rows,
                ds=ds,
                seed=seed,
                framework=fw,
                status=ag.status,
                y_pred=ag.y_pred,
                y_proba=ag.y_proba,
                wall_time_s=ag.wall_time_s,
                peak_rss_mb=ag.peak_rss_mb,
                cpu_user_s=ag.cpu_user_s,
                cpu_system_s=ag.cpu_system_s,
                failure_reason=ag.failure_reason,
                holdout_test=holdout.test,
            )


def _write_artifacts(out_dir: Path, payload: dict, raw_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / raw_name
    raw_text = json.dumps(payload, indent=2)
    raw_path.write_text(raw_text, encoding="utf-8")

    env_path = out_dir / "environment.json"
    env_path.write_text(json.dumps(payload.get("environment", {}), indent=2), encoding="utf-8")

    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    checksums = out_dir / "checksums.sha256"
    checksums.write_text(
        f"{digest}  {raw_name}\n"
        f"{hashlib.sha256(env_path.read_bytes()).hexdigest()}  environment.json\n",
        encoding="utf-8",
    )
    return raw_path


def _campaign(
    *,
    study: str,
    dry_run: bool,
    out_dir: Path,
    time_budget_s: float,
    seeds: list[int],
    frameworks: tuple[str, ...] = ANNOUNCED_FRAMEWORKS,
) -> dict:
    from benchmarks.frameworks.datasets import load_pilot_datasets
    from benchmarks.frameworks.eval import make_holdout, result_dict

    protocol = _load_protocol()
    datasets = load_pilot_datasets(study=study if study in {"pilot", "A"} else "A")
    rows: list[dict] = []

    if dry_run:
        for ds in datasets:
            for seed in seeds:
                for fw in frameworks:
                    rows.append(
                        result_dict(
                            dataset_id=ds.dataset_id,
                            task=ds.task,
                            metric=ds.metric,
                            framework=fw,
                            seed=seed,
                            status="skipped",
                            primary_score=None,
                            wall_time_s=0.0,
                            note="protocol check only",
                        )
                    )
    else:
        for ds in datasets:
            for seed in seeds:
                holdout = make_holdout(ds.frame, ds.target, task=ds.task, seed=seed)
                _run_frameworks_on_holdout(
                    ds=ds,
                    holdout=holdout,
                    seed=seed,
                    out_dir=out_dir,
                    time_budget_s=time_budget_s,
                    frameworks=frameworks,
                    rows=rows,
                )

    skipped = [r for r in rows if r.get("status") == "skipped" and not dry_run]
    comparison_complete = not any(
        r.get("framework") in frameworks and r.get("status") == "skipped" for r in rows
    ) or dry_run

    payload = {
        "protocol": "catml-benchmark-v1",
        "protocol_version": protocol.get("version"),
        "protocol_status": protocol.get("status"),
        "study": study,
        "evaluation": "shared_external_holdout",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "time_budget_s": time_budget_s,
        "seeds": seeds,
        "environment": _environment(),
        "datasets": [d.dataset_id for d in datasets],
        "frameworks": list(frameworks),
        "comparison_complete": comparison_complete,
        "n_skipped": len(skipped),
        "rows": rows,
        "notes": [
            "primary_score is always from the shared external holdout evaluator",
            "skipped frameworks are recorded but do not count as completed comparisons",
            "publish competitive claims only when comparison_complete is true",
        ],
    }
    raw_name = "pilot-raw.json" if study == "pilot" else f"study-{study.lower()}-raw.json"
    _write_artifacts(out_dir, payload, raw_name)
    return payload


def _study_b_scaffold(dry_run: bool, out_dir: Path) -> dict:
    """Scaffold-only payload (kept for --dry-run)."""
    from benchmarks.frameworks.study_b import STUDY_B_CONFIGS

    rows = [
        {
            "config": c,
            "status": "scaffolded",
            "note": "hidden holdout; Explore/agent only on train folds; not executed yet",
        }
        for c in STUDY_B_CONFIGS
    ]
    payload = {
        "protocol": "catml-benchmark-v1",
        "study": "B",
        "evaluation": "hidden_holdout_until_final",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configurations": list(STUDY_B_CONFIGS),
        "holdout_policy": "never_seen_during_search",
        "dry_run": dry_run,
        "rows": rows,
        "notes": [
            "Study B dry-run scaffold",
            "Wire Ollama/Qwen later for catml_agent if measuring LLM lift",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "study-b-scaffold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _study_b_campaign(
    *,
    dry_run: bool,
    out_dir: Path,
    time_budget_s: float,
    seeds: list[int],
    dataset_filter: list[str] | None,
) -> dict:
    """Execute Study B: shared holdout; Explore only sees train."""
    from benchmarks.frameworks.datasets import load_pilot_datasets
    from benchmarks.frameworks.eval import make_holdout, result_dict
    from benchmarks.frameworks.study_b import STUDY_B_CONFIGS, run_study_b_cell

    if dry_run:
        return _study_b_scaffold(dry_run=True, out_dir=out_dir)

    datasets = load_pilot_datasets(study="A")
    if dataset_filter:
        wanted = set(dataset_filter)
        datasets = [d for d in datasets if d.dataset_id in wanted]
    if not datasets:
        raise SystemExit("No datasets selected for Study B")

    rows: list[dict] = []
    for ds in datasets:
        for seed in seeds:
            holdout = make_holdout(ds.frame, ds.target, task=ds.task, seed=seed)
            for config in STUDY_B_CONFIGS:
                cell = run_study_b_cell(
                    config=config,
                    holdout=holdout,
                    task=ds.task,
                    metric=ds.metric,
                    workspace_dir=out_dir / "ws_study_b" / ds.dataset_id / f"seed{seed}" / config,
                    time_budget=time_budget_s,
                    random_state=seed,
                )
                rows.append(
                    result_dict(
                        dataset_id=ds.dataset_id,
                        task=ds.task,
                        metric=ds.metric,
                        framework=config,
                        seed=seed,
                        status=cell.status,
                        primary_score=cell.primary_score,
                        wall_time_s=cell.wall_time_s,
                        peak_rss_mb=cell.peak_rss_mb,
                        secondary=(cell.extra or {}).get("secondary"),
                        failure_reason=cell.failure_reason,
                        model_id=(cell.extra or {}).get("model_id"),
                        note=(cell.extra or {}).get("note"),
                        framework_version="study-b",
                    )
                )
                # Attach explore metadata without breaking result_dict schema
                if cell.extra:
                    rows[-1]["extra"] = {
                        k: v for k, v in cell.extra.items() if k != "secondary"
                    }

    skipped = [r for r in rows if r.get("status") == "skipped"]
    failed = [r for r in rows if r.get("status") == "failed"]
    ok = [r for r in rows if r.get("status") == "ok"]
    # comparison_complete when every non-skipped cell is ok
    comparison_complete = len(failed) == 0 and len(ok) > 0

    payload = {
        "protocol": "catml-benchmark-v1",
        "study": "B",
        "evaluation": "hidden_holdout_until_final",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "time_budget_s": time_budget_s,
        "seeds": seeds,
        "environment": _environment(),
        "datasets": [d.dataset_id for d in datasets],
        "configurations": list(STUDY_B_CONFIGS),
        "holdout_policy": "never_seen_during_search",
        "comparison_complete": comparison_complete,
        "n_skipped": len(skipped),
        "n_failed": len(failed),
        "rows": rows,
        "notes": [
            "Explore runs only on train CSV registered in an isolated workspace",
            "catml_agent is skipped until FakeLLM/Ollama lift measurement is wired",
            "primary_score is always from the shared external holdout evaluator",
        ],
    }
    _write_artifacts(out_dir, payload, "study-b-raw.json")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="CATML Benchmark v1 runner")
    parser.add_argument("--study", choices=["pilot", "A", "B", "C"], default="pilot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--time-budget-minutes",
        type=float,
        default=5.0,
        help="Wall-clock budget per framework×dataset×seed (default: 5)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds (protocol default 42,43,44)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset_id filter (Study A/B)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "v1",
    )
    args = parser.parse_args()

    if args.study == "C":
        print("Study C: run `pytest tests/test_benchmark_study_c.py`")
        return 0

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    dataset_filter = [s.strip() for s in args.datasets.split(",") if s.strip()] or None

    if args.study == "B":
        payload = _study_b_campaign(
            dry_run=args.dry_run,
            out_dir=args.out,
            time_budget_s=float(args.time_budget_minutes) * 60.0,
            seeds=seeds,
            dataset_filter=dataset_filter,
        )
        print(json.dumps(payload, indent=2))
        failed = [r for r in payload["rows"] if r.get("status") == "failed"]
        return 1 if failed else 0

    payload = _campaign(
        study=args.study,
        dry_run=args.dry_run,
        out_dir=args.out,
        time_budget_s=float(args.time_budget_minutes) * 60.0,
        seeds=seeds,
    )
    print(json.dumps(payload, indent=2))
    failed = [r for r in payload["rows"] if r.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
