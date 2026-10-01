"""CLI interface for CATML agent governance, approval requests, and audit ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from automl.domain.agents.entities import ApprovalStatus
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger


def _resolve_ledger(workspace_path: str | Path | None = None) -> SqliteAgentLedger:
    """Resolve the SQLite agent ledger from the given workspace or standard directory."""
    if workspace_path:
        p = Path(workspace_path)
        if p.is_file() or str(p).endswith(".db"):
            ledger_file = p
        else:
            ledger_file = p / "agent_ledger.db"
    else:
        # Check standard locations
        default_dir = Path.cwd() / ".automl" / "default"
        if (default_dir / "agent_ledger.db").exists():
            ledger_file = default_dir / "agent_ledger.db"
        elif (Path.cwd() / ".automl" / "agent_ledger.db").exists():
            ledger_file = Path.cwd() / ".automl" / "agent_ledger.db"
        else:
            ledger_file = default_dir / "agent_ledger.db"

    return SqliteAgentLedger(ledger_file)


def approvals_list_cli(args: argparse.Namespace) -> int:
    """List pending or filtered agent approval requests."""
    try:
        ledger = _resolve_ledger(getattr(args, "workspace", None))
        run_id = getattr(args, "run_id", None)
        status_arg = getattr(args, "status", "pending")

        status_filter: ApprovalStatus | None = None
        if status_arg and status_arg.lower() != "all":
            try:
                status_filter = ApprovalStatus(status_arg.lower())
            except ValueError:
                sys.stderr.write(f"Error: Invalid status filter '{status_arg}'.\n")
                return 1

        approvals = ledger.list_approvals(run_id=run_id, status=status_filter)

        if getattr(args, "json", False):
            payload = []
            for a in approvals:
                payload.append(
                    {
                        "approval_id": a.approval_id,
                        "action": a.action,
                        "actor": a.actor,
                        "run_id": a.run_id,
                        "status": a.status.value,
                        "arguments": a.arguments,
                        "reviewer": a.reviewer,
                        "expires_at": a.expires_at,
                        "created_at": a.created_at,
                        "updated_at": a.updated_at,
                    }
                )
            print(json.dumps(payload, indent=2))
            return 0

        if not approvals:
            filter_desc = f" with status '{status_arg}'" if status_arg != "all" else ""
            if run_id:
                filter_desc += f" for run '{run_id}'"
            print(f"No approval requests found{filter_desc}.")
            return 0

        # Formatted text table
        header = f"{'APPROVAL ID':<26} {'ACTION':<20} {'RUN ID':<18} {'STATUS':<10} {'REVIEWER':<14} {'EXPIRES AT':<20}"
        print(header)
        print("-" * len(header))
        for a in approvals:
            reviewer_str = a.reviewer or "-"
            print(
                f"{a.approval_id:<26} {a.action:<20} {a.run_id:<18} {a.status.value:<10} {reviewer_str:<14} {a.expires_at:<20}"
            )
        return 0
    except Exception as e:
        sys.stderr.write(f"Error listing approvals: {e}\n")
        return 1


def approve_cli(args: argparse.Namespace) -> int:
    """Approve or reject a pending agent action approval request."""
    try:
        ledger = _resolve_ledger(getattr(args, "workspace", None))
        approval_id = args.approval_id
        reject = getattr(args, "reject", False)
        reviewer = getattr(args, "reviewer", "cli_user") or "cli_user"
        notes = getattr(args, "notes", None)

        reviewer_info = f"{reviewer} ({notes})" if notes else reviewer
        target_status = ApprovalStatus.REJECTED if reject else ApprovalStatus.APPROVED

        resolved = ledger.update_approval_status(
            approval_id=approval_id,
            status=target_status,
            reviewer=reviewer_info,
        )

        action_word = "rejected" if reject else "approved"
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "approval_id": resolved.approval_id,
                        "status": resolved.status.value,
                        "reviewer": resolved.reviewer,
                        "action": resolved.action,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Approval request '{resolved.approval_id}' successfully {action_word} by '{resolved.reviewer}'.")
        return 0
    except KeyError:
        sys.stderr.write(f"Error: Approval request '{args.approval_id}' not found.\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"Error updating approval request: {e}\n")
        return 1


def register_agent_subparser(agent_parser: argparse.ArgumentParser) -> None:
    """Register 'agent' command and subcommands into the main CLI argument parser."""
    sub = agent_parser.add_subparsers(dest="agent_subcommand", required=True)

    # 1. approvals subparser: automl agent approvals <list|approve>
    approvals_p = sub.add_parser("approvals", help="Manage agent action approval requests")
    appr_sub = approvals_p.add_subparsers(dest="approvals_subcommand", required=True)

    # 1a. approvals list
    list_p = appr_sub.add_parser("list", help="List approval requests")
    list_p.add_argument("--run-id", help="Filter approvals by run ID")
    list_p.add_argument(
        "--status",
        choices=["pending", "approved", "rejected", "revoked", "expired", "all", "PENDING", "APPROVED", "REJECTED", "REVOKED", "EXPIRED", "ALL"],
        default="pending",
        help="Filter by status (default: pending)",
    )
    list_p.add_argument("--workspace", help="Path to workspace directory or ledger db file")
    list_p.add_argument("--json", action="store_true", help="Output in JSON format")
    list_p.set_defaults(func=approvals_list_cli)

    # 1b. approvals approve
    appr_action_p = appr_sub.add_parser("approve", help="Approve or reject a pending approval request")
    appr_action_p.add_argument("approval_id", help="ID of the approval request to resolve")
    appr_action_p.add_argument("--reject", action="store_true", help="Reject the request instead of approving it")
    appr_action_p.add_argument("--reviewer", default="cli_user", help="Reviewer identifier (default: cli_user)")
    appr_action_p.add_argument("--notes", help="Optional review notes")
    appr_action_p.add_argument("--workspace", help="Path to workspace directory or ledger db file")
    appr_action_p.add_argument("--json", action="store_true", help="Output in JSON format")
    appr_action_p.set_defaults(func=approve_cli)

    # 2. direct alias: automl agent approve <approval_id>
    direct_appr_p = sub.add_parser("approve", help="Approve or reject a pending approval request")
    direct_appr_p.add_argument("approval_id", help="ID of the approval request to resolve")
    direct_appr_p.add_argument("--reject", action="store_true", help="Reject the request instead of approving it")
    direct_appr_p.add_argument("--reviewer", default="cli_user", help="Reviewer identifier (default: cli_user)")
    direct_appr_p.add_argument("--notes", help="Optional review notes")
    direct_appr_p.add_argument("--workspace", help="Path to workspace directory or ledger db file")
    direct_appr_p.add_argument("--json", action="store_true", help="Output in JSON format")
    direct_appr_p.set_defaults(func=approve_cli)
