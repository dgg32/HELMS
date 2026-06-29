#!/usr/bin/env python3
"""Step sequencing shared by pipeline.py (CLI) and htmx_app/main.py (web UI).

Module-level helpers are used directly by the web UI for step-sequencing.
PipelineOrchestrator is used by pipeline.py for sequential end-to-end execution.
"""
from __future__ import annotations

import inspect
from argparse import Namespace
from collections.abc import Callable

STEP_ORDER: tuple[str, ...] = ("convert", "extract", "apply")


def step_after(key: str) -> str | None:
    """Return the step that follows key, or None if key is last."""
    try:
        idx = STEP_ORDER.index(key)
    except ValueError:
        return None
    return STEP_ORDER[idx + 1] if idx + 1 < len(STEP_ORDER) else None


def pending_after(key: str, *, include_apply: bool = True) -> list[str]:
    """Return steps that follow key — used to sequence pending steps in the web UI."""
    try:
        idx = STEP_ORDER.index(key)
    except ValueError:
        return []
    result = list(STEP_ORDER[idx + 1:])
    if not include_apply:
        result = [s for s in result if s != "apply"]
    return result


class PipelineOrchestrator:
    """
    Encapsulates step sequence for the 3-step pipeline.

    CLI (pipeline.py):     build an instance with step lambdas, call run_all().
    UI  (htmx_app/main.py): use module-level step_after() / pending_after() helpers.
    """

    STEPS = STEP_ORDER
    step_after = staticmethod(step_after)
    pending_after = staticmethod(pending_after)

    def __init__(
        self,
        steps: list[tuple[str, Callable, Callable[[], Namespace]]],
    ) -> None:
        """
        steps: list of (key, fn, args_builder)
          key          — "convert" | "extract" | "apply"
          fn           — sync or async callable; receives args_builder()
          args_builder — zero-arg callable returning a Namespace
        """
        self._steps = steps

    async def run_all(
        self,
        *,
        on_step_start: Callable[[str], None] | None = None,
        on_step_done: Callable[[str], None] | None = None,
        on_step_error: Callable[[str, Exception], None] | None = None,
    ) -> None:
        """Run steps sequentially; re-raise first error after calling on_step_error."""
        for key, fn, build_args in self._steps:
            if on_step_start:
                on_step_start(key)
            try:
                ns = build_args()
                if inspect.iscoroutinefunction(fn):
                    await fn(ns)
                else:
                    fn(ns)
            except Exception as exc:
                if on_step_error:
                    on_step_error(key, exc)
                raise
            if on_step_done:
                on_step_done(key)
