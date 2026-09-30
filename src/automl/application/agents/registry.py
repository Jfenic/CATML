"""Agent tool registry maintaining tool definitions and executable handlers."""
from __future__ import annotations

from typing import Any, Callable

from automl.application.agents.contracts import ToolDefinition

ToolHandler = Callable[[dict[str, Any], Any], Any]


class ToolRegistry:
    """Registry maintaining available agent tools and their associated handlers."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        """Register a tool definition and its execution handler."""
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' is already registered")
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Retrieve a tool definition by name."""
        return self._tools.get(name)

    def get_handler(self, name: str) -> ToolHandler | None:
        """Retrieve a tool handler by name."""
        return self._handlers.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tool definitions."""
        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def unregister(self, name: str) -> None:
        """Unregister a tool if needed."""
        self._tools.pop(name, None)
        self._handlers.pop(name, None)
