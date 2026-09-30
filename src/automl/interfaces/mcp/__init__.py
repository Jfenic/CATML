"""Model Context Protocol (MCP) interface adapter for CATML."""
from __future__ import annotations

from automl.interfaces.mcp.server import HAS_MCP, create_mcp_server, run_stdio_server

__all__ = ["HAS_MCP", "create_mcp_server", "run_stdio_server"]
