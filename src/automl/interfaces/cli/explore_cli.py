"""CLI commands for CATML Explore studies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from automl.application.analysis.commands import (
    CreateStudyCommand,
    RunAnalysisCommand,
    VerifyHypothesisCommand,
)
from automl.application.analysis.queries import (
    GetStudyQuery,
    ListEvidenceLinksQuery,
    ListFindingsQuery,
    ListHypothesesQuery,
    ListStudiesQuery,
    ListVisualizationsQuery,
)
from automl.application.analysis.reporting import generate_study_markdown_report
from automl.application.bootstrap import build_application


def run_explore_cli(args: argparse.Namespace) -> int:
    """Dispatches `catml explore` subcommands via CommandBus and QueryBus."""
    ws_dir = getattr(args, "workspace", None)
    ws, cmd, qry = build_application(root_dir=ws_dir)

    action = getattr(args, "explore_action", None)

    if action == "create":
        dataset_id = args.dataset_id
        name = getattr(args, "name", None) or f"Study {dataset_id}"
        target = getattr(args, "target", None)
        try:
            study_id = cmd.dispatch(
                CreateStudyCommand(
                    workspace_id=ws.id,
                    dataset_id=dataset_id,
                    name=name,
                    target_column=target,
                )
            )
            if getattr(args, "json", False):
                print(json.dumps({"status": "success", "study_id": study_id}))
            else:
                print(f"Created study: {study_id} (dataset: {dataset_id})")
            return 0
        except Exception as exc:
            print(f"Error creating study: {exc}", file=sys.stderr)
            return 1

    elif action == "run":
        study_id = args.study_id
        try:
            run_id = cmd.dispatch(RunAnalysisCommand(study_id=study_id))
            study = qry.dispatch(GetStudyQuery(study_id=study_id))
            findings = qry.dispatch(ListFindingsQuery(study_id=study_id, run_id=run_id))
            if getattr(args, "json", False):
                print(json.dumps({"status": "success", "run_id": run_id, "findings_count": len(findings), "study": study}, indent=2))
            else:
                print(f"Executed study {study_id} (Run ID: {run_id})")
                print(f"  Findings detected: {len(findings)}")
                for f in findings[:5]:
                    print(f"  - [{f['finding_type'].upper()}] {f['summary']}")
                if len(findings) > 5:
                    print(f"  ... and {len(findings) - 5} more findings.")
            return 0
        except Exception as exc:
            print(f"Error running study: {exc}", file=sys.stderr)
            return 1

    elif action == "list":
        dataset_id = getattr(args, "dataset", None)
        studies = qry.dispatch(ListStudiesQuery(workspace_id=ws.id, dataset_id=dataset_id))
        if getattr(args, "json", False):
            print(json.dumps(studies, indent=2))
        else:
            if not studies:
                print("No exploratory studies registered in this workspace.")
            else:
                print(f"Exploratory Studies ({len(studies)} registered):")
                print("-" * 65)
                for s in studies:
                    target_str = f"Target: {s.get('target_column') or 'None (Unsupervised)'}"
                    print(f"  {s['id']}  [{s['status']}]  {s['name'][:25]:25s}  {target_str}")
        return 0

    elif action == "show":
        study_id = args.study_id
        study = qry.dispatch(GetStudyQuery(study_id=study_id))
        if not study:
            print(f"Error: Study '{study_id}' not found.", file=sys.stderr)
            return 1
        findings = qry.dispatch(ListFindingsQuery(study_id=study_id))
        viz = qry.dispatch(ListVisualizationsQuery(study_id=study_id))
        hyps = qry.dispatch(ListHypothesesQuery(study_id=study_id))
        evidence = qry.dispatch(ListEvidenceLinksQuery(study_id=study_id))
        payload = {
            "study": study,
            "findings": findings,
            "visualizations": viz,
            "hypotheses": hyps,
            "evidence_links": evidence,
        }
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2))
        else:
            print(f"Study: {study['name']} ({study['id']})")
            print("-" * 60)
            print(f"  Dataset ID:       {study['data_source']['dataset_id']}")
            print(f"  Path:             {study['data_source']['path']}")
            print(f"  Target Column:    {study.get('target_column') or 'None (Unsupervised)'}")
            print(f"  Status:           {study['status']}")
            print(f"  Created:          {study['created_at']}")
            print(f"  Findings count:   {len(findings)}")
            print(f"  Charts specs:     {len(viz)}")
            print(f"  Hypotheses count: {len(hyps)}")
            print(f"  Evidence links:   {len(evidence)}")
        return 0

    elif action == "findings":
        study_id = args.study_id
        study = qry.dispatch(GetStudyQuery(study_id=study_id))
        if not study:
            print(f"Error: Study '{study_id}' not found.", file=sys.stderr)
            return 1

        ftype = getattr(args, "type", None)
        min_sig = getattr(args, "min_sig", None)
        limit = getattr(args, "limit", None)

        findings = qry.dispatch(
            ListFindingsQuery(
                study_id=study_id,
                finding_type=ftype,
                min_significance=min_sig,
                limit=limit,
            )
        )
        if getattr(args, "json", False):
            print(json.dumps(findings, indent=2))
        else:
            if not findings:
                print(f"No matching findings recorded for study '{study_id}'.")
            else:
                print(f"Statistical Findings for {study_id} ({len(findings)} matching):")
                print("-" * 65)
                for f in findings:
                    col_str = f"Col: {f.get('column_name') or '-'}"
                    p_str = f"p={f['p_value']:.4e}" if f.get("p_value") is not None else ""
                    print(f"  [{f['finding_type']:22s}]  {col_str:20s}  {p_str}")
                    print(f"    Summary: {f['summary']}\n")
        return 0

    elif action == "export":
        study_id = args.study_id
        study = qry.dispatch(GetStudyQuery(study_id=study_id))
        if not study:
            print(f"Error: Study '{study_id}' not found.", file=sys.stderr)
            return 1

        findings = qry.dispatch(ListFindingsQuery(study_id=study_id))
        viz = qry.dispatch(ListVisualizationsQuery(study_id=study_id))
        hyps = qry.dispatch(ListHypothesesQuery(study_id=study_id))
        evidence = qry.dispatch(ListEvidenceLinksQuery(study_id=study_id))
        fmt = getattr(args, "format", "markdown").lower()
        output_path = getattr(args, "output", None)

        if fmt in ("json", "js"):
            payload = {
                "study": study,
                "findings": findings,
                "visualizations": viz,
                "hypotheses": hyps,
                "evidence_links": evidence,
            }
            content = json.dumps(payload, indent=2, default=str)
        else:
            content = generate_study_markdown_report(
                study=study,
                findings=findings,
                visualizations=viz,
                hypotheses=hyps,
                evidence_links=evidence,
            )

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(content, encoding="utf-8")
            print(f"Successfully exported {fmt.upper()} report to: {output_path}")
        else:
            print(content)
        return 0

    elif action == "verify":
        hypothesis_id = args.hypothesis_id
        run_id = getattr(args, "run_id", None)
        min_imp = getattr(args, "min_improvement", 0.0) or 0.0
        try:
            cmd_obj = VerifyHypothesisCommand(
                hypothesis_id=hypothesis_id,
                run_id=run_id,
                min_improvement=min_imp,
            )
            link = cmd.dispatch(cmd_obj)
            if getattr(args, "json", False):
                print(json.dumps(link, indent=2, default=str))
            else:
                status_str = "ACCEPTED" if link.get("accepted") else "REJECTED"
                print(f"Hypothesis Verification Result: [{status_str}]")
                print("-" * 60)
                print(f"  Link ID:         {link.get('id')}")
                print(f"  Hypothesis ID:   {link.get('hypothesis_id')}")
                print(f"  Experiment ID:   {link.get('experiment_id')}")
                print(f"  Metric:          {link.get('metric')}")
                print(f"  Baseline Score:  {link.get('baseline_score', 0.0):.4f}")
                print(f"  Candidate Score: {link.get('candidate_score', 0.0):.4f}")
                print(f"  Decision Notes:  {link.get('notes')}")
            return 0
        except Exception as exc:
            print(f"Error verifying hypothesis: {exc}", file=sys.stderr)
            return 1

    else:
        print("Error: Unknown explore action. Use --help for usage.", file=sys.stderr)
        return 1
