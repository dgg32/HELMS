"""Shared types for the agentic-extraction tool registry.

Kept in a separate module so gleif_tools/umls_tools can import ResolverToolset
without creating a circular dependency with resolver_tools.py (which imports them).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

ToolHandler = Callable[[object, dict], str]  # (session, args) -> str


@dataclass
class ResolverToolset:
    """A naming service's agentic tools.

    name:     service identifier ("gleif", "umls", …) — for logging/clarity.
    specs:    list of OpenAI function-tool schema dicts exposed to the LLM.
    handlers: {tool_name: (session, args) -> str} executed on tool call.
    """
    name: str
    specs: list[dict]
    handlers: dict[str, ToolHandler]
