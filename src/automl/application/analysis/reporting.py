"""Deterministic reporting generator for CATML Explore studies."""
from __future__ import annotations

from typing import Any


def generate_study_markdown_report(
    study: dict[str, Any],
    findings: list[dict[str, Any]],
    visualizations: list[dict[str, Any]] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    evidence_links: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a reproducible, technical Markdown report from study DTOs.

    Includes executive summary, statistical findings, visualizations index,
    actionable ML hypotheses, and empirical evidence links.
    """
    visualizations = visualizations or []
    hypotheses = hypotheses or []
    evidence_links = evidence_links or []

    study_id = study.get("id", "unknown")
    name = study.get("name", "Exploratory Study")
    data_source = study.get("data_source") or {}
    dataset_id = data_source.get("dataset_id") or study.get("dataset_id", "unknown")
    dataset_path = data_source.get("path") or study.get("dataset_path", "-")
    target_col = study.get("target_column") or "None (Unsupervised)"
    status = study.get("status", "COMPLETED")
    created_at = study.get("created_at", "-")

    # Metrics aggregation
    total_findings = len(findings)
    sig_findings = []
    findings_by_type: dict[str, int] = {}

    for f in findings:
        ftype = str(f.get("finding_type", "other")).upper()
        findings_by_type[ftype] = findings_by_type.get(ftype, 0) + 1

        p_raw = f.get("p_value")
        metrics = f.get("metrics") or {}
        p_fdr = metrics.get("p_value_fdr")
        if (p_raw is not None and p_raw <= 0.05) or (p_fdr is not None and p_fdr <= 0.05):
            sig_findings.append(f)

    md = [
        f"# Technical Exploration Report · CATML Explore",
        "",
        f"**Study Name:** {name} (`{study_id}`)",
        f"**Dataset:** `{dataset_id}` (Path: `{dataset_path}`)",
        f"**Target Column:** `{target_col}`",
        f"**Status:** `{status}`",
        f"**Created At:** {created_at}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total Quantitative Findings:** {total_findings}",
        f"- **Statistically Significant Findings (p < 0.05 or FDR p < 0.05):** {len(sig_findings)}",
        f"- **Declarative Visualizations Generated:** {len(visualizations)}",
        f"- **Actionable ML Hypotheses Formulated:** {len(hypotheses)}",
        f"- **Empirically Verified Evidence Links:** {len(evidence_links)}",
        "",
        "### Findings Breakdown by Category",
        "",
        "| Category / Type | Count | Share |",
        "|---|---:|---:|",
    ]

    for ftype, count in sorted(findings_by_type.items(), key=lambda x: -x[1]):
        share = (count / total_findings * 100.0) if total_findings > 0 else 0.0
        md.append(f"| `{ftype}` | {count} | {share:.1f}% |")

    md.extend([
        "",
        "---",
        "",
        "## 2. Statistical Findings Catalog",
        "",
    ])

    if not findings:
        md.append("_No statistical findings recorded for this study._")
    else:
        for idx, f in enumerate(findings, 1):
            fid = f.get("id", f"finding_{idx}")
            ftype = str(f.get("finding_type", "finding")).upper()
            summary = f.get("summary", "")
            col = f.get("column_name") or "-"
            sec_col = f.get("secondary_column")
            cols_str = f"`{col}`" if not sec_col else f"`{col}` / `{sec_col}`"
            method = f.get("method_name", "-")
            p_val = f.get("p_value")
            metrics = f.get("metrics") or {}
            p_fdr = metrics.get("p_value_fdr")
            eff_size = f.get("effect_size")
            limitations = f.get("limitations") or []

            p_str = f"{p_val:.4e}" if p_val is not None else "N/A"
            fdr_str = f" (FDR p={p_fdr:.4e})" if p_fdr is not None else ""
            eff_str = f"{eff_size:.4f}" if eff_size is not None else "N/A"

            md.extend([
                f"### {idx}. [{ftype}] {summary}",
                f"- **Finding ID:** `{fid}`",
                f"- **Features Involved:** {cols_str}",
                f"- **Statistical Method:** `{method}`",
                f"- **Inference:** p-value = `{p_str}`{fdr_str} | Effect Size = `{eff_str}`",
            ])

            if limitations:
                md.append(f"- **Caveats / Assumptions Violations:** {', '.join(limitations)}")

            md.append("")

    md.extend([
        "---",
        "",
        "## 3. Visualizations Index",
        "",
    ])

    if not visualizations:
        md.append("_No declarative visualizations attached to this study._")
    else:
        md.extend([
            "| ID | Type | Title | Key Series / Dimensions |",
            "|---|---|---|---|",
        ])
        for v in visualizations:
            vid = v.get("id", "-")
            vtype = v.get("chart_type", "chart")
            vtitle = v.get("title", "-")
            series_keys = list((v.get("data_series") or {}).keys())
            series_str = ", ".join(series_keys[:4]) if series_keys else "-"
            md.append(f"| `{vid}` | `{vtype}` | {vtitle} | {series_str} |")

    md.extend([
        "",
        "---",
        "",
        "## 4. Actionable ML Hypotheses (Propose ≠ Accept)",
        "",
    ])

    if not hypotheses:
        md.append("_No machine learning hypotheses proposed yet._")
    else:
        for idx, h in enumerate(hypotheses, 1):
            hid = h.get("id", f"hyp_{idx}")
            desc = h.get("description", "")
            action = h.get("proposed_action", "-")
            finding_ref = h.get("finding_id", "-")
            hstatus = h.get("status", "proposed").upper()
            delta = h.get("experiment_delta") or {}

            md.extend([
                f"### H{idx}. [{hstatus}] {desc}",
                f"- **Hypothesis ID:** `{hid}`",
                f"- **Derived from Finding:** `{finding_ref}`",
                f"- **Proposed ML Action:** `{action}`",
                f"- **Experiment Delta Configuration:** `{delta}`",
                "",
            ])

    md.extend([
        "---",
        "",
        "## 5. Empirical Verification & Evidence Links",
        "",
    ])

    if not evidence_links:
        md.append("_No empirical verification links recorded for this study yet._")
    else:
        md.extend([
            "| Link ID | Hypothesis ID | Experiment ID | Baseline | Candidate | Metric | Decision | Notes |",
            "|---|---|---|---:|---:|---|---|---|",
        ])
        for el in evidence_links:
            eid = el.get("id", "-")
            hid = el.get("hypothesis_id", "-")
            exp_id = el.get("experiment_id", "-")
            b_score = el.get("baseline_score", 0.0)
            c_score = el.get("candidate_score", 0.0)
            metric_name = el.get("metric", "-")
            decision_str = "**ACCEPTED**" if el.get("accepted") else "_REJECTED_"
            notes_str = el.get("notes", "-").replace("\n", " ")
            md.append(
                f"| `{eid}` | `{hid}` | `{exp_id}` | {b_score:.4f} | {c_score:.4f} | `{metric_name}` | {decision_str} | {notes_str} |"
            )

    md.extend([
        "",
        "---",
        "",
        "_Report automatically generated by CATML Explore (Deterministic Analysis Engine)._",
        "",
    ])

    return "\n".join(md)
