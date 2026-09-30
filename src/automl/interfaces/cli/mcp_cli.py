"""CLI handler for running the Model Context Protocol (MCP) server."""
from __future__ import annotations

import argparse
import sys


def run_mcp_cli(args: argparse.Namespace) -> int:
    """Launch the CATML MCP stdio server."""
    workspace_path = getattr(args, "workspace", None)
    try:
        from automl.interfaces.mcp.server import HAS_MCP, run_stdio_server
        if not HAS_MCP:
            sys.stderr.write(
                "Error: El componente MCP requiere dependencias opcionales.\n"
                "Instálalas con: pip install '.[mcp]'\n"
            )
            return 1
        run_stdio_server(root_dir=workspace_path)
        return 0
    except ImportError as e:
        sys.stderr.write(
            f"Error: Dependencia faltante para MCP ({e}).\n"
            "Instálala con: pip install '.[mcp]'\n"
        )
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        sys.stderr.write(f"Error al iniciar servidor MCP: {e}\n")
        return 1
