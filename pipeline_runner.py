#!/usr/bin/env python3
"""In-process pipeline runner — replaces subprocess.Popen for extract/convert/apply.

sys.stdout and sys.stderr are replaced at import time with a _ThreadLocalStream.
In the main thread, all writes pass through to the original streams
unchanged.  In background runner threads, writes go to a queue.Queue so that
the web UI's log poller can read them line by line, exactly as it did with a
subprocess pipe.

PipelineRunner mimics the subprocess.Popen interface used by _poll_cmd:
  .poll()           → None while running; returncode (int) when done
  .wait(timeout)    → blocks; raises subprocess.TimeoutExpired on timeout
  .terminate()/.kill() → best-effort (sets a flag; can't hard-kill a thread)
  .returncode       → int once finished, None before
"""
from __future__ import annotations

import asyncio
import os
import queue as _queue_mod
import subprocess
import sys
import threading
from typing import Any, Callable


# ── Thread-local stdout/stderr router ────────────────────────────────────────

class _ThreadLocalStream:
    """Thin sys.stdout/stderr wrapper that dispatches writes per-thread.

    In threads that haven't called .set(), writes fall through to the original
    stream.  Runner threads call .set(writer) before starting work and
    .clear() in the finally block.
    """

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._local = threading.local()
        self.encoding = getattr(fallback, "encoding", "utf-8")
        self.errors   = getattr(fallback, "errors",   "strict")

    def _target(self) -> Any:
        return getattr(self._local, "target", self._fallback)

    def write(self, s: str) -> int:
        return self._target().write(s)

    def flush(self) -> None:
        self._target().flush()

    def fileno(self) -> int:
        return self._fallback.fileno()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def set(self, target: Any) -> None:
        self._local.target = target

    def clear(self) -> None:
        try:
            del self._local.target
        except AttributeError:
            pass


class _QueueWriter:
    """Writes text lines to a queue.Queue.  Buffers incomplete lines until \\n."""

    encoding = "utf-8"
    errors   = "strict"

    def __init__(self, q: _queue_mod.Queue) -> None:
        self._q   = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._q.put(line + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._q.put(self._buf)
            self._buf = ""

    def isatty(self) -> bool:
        return False


# Install once when this module is first imported.
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
_tls_out = _ThreadLocalStream(sys.stdout)
_tls_err = _ThreadLocalStream(sys.stderr)
sys.stdout = _tls_out  # type: ignore[assignment]
sys.stderr = _tls_err  # type: ignore[assignment]

# Serialises the save+update and restore windows for os.environ and
# _extract_mod._CACHE_DIR so concurrent runners don't stomp each other.
_env_lock = threading.Lock()


# ── PipelineRunner ────────────────────────────────────────────────────────────

class PipelineRunner:
    """Run a callable (sync or async) in a background thread.

    Parameters
    ----------
    fn:           Callable that accepts a single ``args`` argument (argparse.Namespace).
                  May return a coroutine — it will be driven with asyncio.run().
    args:         argparse.Namespace to pass to fn.
    output_queue: queue.Queue that receives stdout/stderr lines + None sentinel.
    env:          Dict of env-var overrides applied for the duration of the run.
                  KG_CACHE_DIR is also patched directly onto the extract module.
    """

    def __init__(
        self,
        fn: Callable,
        args: Any,
        output_queue: _queue_mod.Queue,
        env: dict | None = None,
    ) -> None:
        self.returncode: int | None = None
        self._q   = output_queue
        self._fn  = fn
        self._args = args
        self._env  = env or {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cancelled = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    # ── background thread body ────────────────────────────────────────────────

    def _run(self) -> None:
        writer = _QueueWriter(self._q)
        _tls_out.set(writer)
        _tls_err.set(writer)

        # Apply env overrides atomically w.r.t. other runner threads.
        _prev_env: dict[str, str | None] = {}
        _extract_mod      = sys.modules.get("extract")
        _llm_client_mod   = sys.modules.get("llm_client")
        _sentinel = object()
        _orig_cache_dir = getattr(_extract_mod, "_CACHE_DIR", _sentinel) if _extract_mod else _sentinel
        _orig_lm = getattr(_extract_mod, "_LLM_MODEL", _sentinel) if _extract_mod else _sentinel
        _orig_max_tok = getattr(_extract_mod, "_LLM_MAX_COMPLETION_TOKENS", _sentinel) if _extract_mod else _sentinel
        _orig_lc_max_tok = getattr(_llm_client_mod, "LLM_MAX_COMPLETION_TOKENS", _sentinel) if _llm_client_mod else _sentinel
        _orig_lc_timeout = getattr(_llm_client_mod, "LLM_TIMEOUT", _sentinel) if _llm_client_mod else _sentinel
        _patched_cache = False
        _patched_lm = False
        _patched_max_tok = False
        _patched_lc_max_tok = False
        _patched_lc_timeout = False
        with _env_lock:
            _prev_env = {k: os.environ.get(k) for k in self._env}
            os.environ.update(self._env)
            _cache_override = self._env.get("KG_CACHE_DIR")
            if _cache_override and _extract_mod is not None:
                from pathlib import Path as _Path
                _extract_mod._CACHE_DIR = _Path(_cache_override)
                _patched_cache = True
            _lm_override = self._env.get("LLM_MODEL")
            if _lm_override and _extract_mod is not None:
                _extract_mod._LLM_MODEL = _lm_override.replace("azure/", "")
                _patched_lm = True
            _max_tok_override = self._env.get("LLM_MAX_COMPLETION_TOKENS")
            if _max_tok_override and _extract_mod is not None:
                try:
                    _extract_mod._LLM_MAX_COMPLETION_TOKENS = int(_max_tok_override)
                    _patched_max_tok = True
                except (ValueError, TypeError):
                    pass
            # Also patch llm_client module so modules that read LLM_MAX_COMPLETION_TOKENS
            # live from llm_client (e.g. extraction_agent) see the override (R3).
            if _max_tok_override and _llm_client_mod is not None:
                try:
                    _llm_client_mod.LLM_MAX_COMPLETION_TOKENS = int(_max_tok_override)
                    _patched_lc_max_tok = True
                except (ValueError, TypeError):
                    pass
            _timeout_override = self._env.get("LLM_TIMEOUT")
            if _timeout_override and _llm_client_mod is not None:
                try:
                    _llm_client_mod.LLM_TIMEOUT = float(_timeout_override)
                    _patched_lc_timeout = True
                except (ValueError, TypeError):
                    pass


        try:
            result = self._fn(self._args)
            if asyncio.iscoroutine(result):
                loop = asyncio.new_event_loop()
                self._loop = loop
                try:
                    loop.run_until_complete(result)
                finally:
                    self._loop = None
                    loop.close()
            self.returncode = -1 if self._cancelled.is_set() else 0
        except SystemExit as e:
            code = e.code
            if isinstance(code, str) and code:
                self._q.put(f"[ERROR] {code}\n")
                self.returncode = 1
            elif code is None or code == 0:
                self.returncode = 0
            else:
                self.returncode = 1 if not isinstance(code, int) else code
        except Exception as e:
            self._q.put(f"\n[ERROR] {type(e).__name__}: {e}\n")
            self.returncode = 1
        finally:
            # Restore env vars and module attribute atomically.
            with _env_lock:
                for k, v in _prev_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                if _patched_cache and _extract_mod is not None:
                    if _orig_cache_dir is _sentinel:
                        try:
                            del _extract_mod._CACHE_DIR
                        except AttributeError:
                            pass
                    else:
                        _extract_mod._CACHE_DIR = _orig_cache_dir
                if _patched_lm and _extract_mod is not None:
                    if _orig_lm is _sentinel:
                        try:
                            del _extract_mod._LLM_MODEL
                        except AttributeError:
                            pass
                    else:
                        _extract_mod._LLM_MODEL = _orig_lm
                if _patched_max_tok and _extract_mod is not None:
                    if _orig_max_tok is _sentinel:
                        try:
                            del _extract_mod._LLM_MAX_COMPLETION_TOKENS
                        except AttributeError:
                            pass
                    else:
                        _extract_mod._LLM_MAX_COMPLETION_TOKENS = _orig_max_tok
                if _patched_lc_max_tok and _llm_client_mod is not None:
                    if _orig_lc_max_tok is _sentinel:
                        try:
                            del _llm_client_mod.LLM_MAX_COMPLETION_TOKENS
                        except AttributeError:
                            pass
                    else:
                        _llm_client_mod.LLM_MAX_COMPLETION_TOKENS = _orig_lc_max_tok
                if _patched_lc_timeout and _llm_client_mod is not None:
                    if _orig_lc_timeout is _sentinel:
                        try:
                            del _llm_client_mod.LLM_TIMEOUT
                        except AttributeError:
                            pass
                    else:
                        _llm_client_mod.LLM_TIMEOUT = _orig_lc_timeout

            writer.flush()  # push any partial last line before the sentinel
            _tls_out.clear()
            _tls_err.clear()
            self._q.put(None)  # sentinel — signals _poll_cmd the process is done

    # ── subprocess.Popen-compatible interface ─────────────────────────────────

    def poll(self) -> int | None:
        """Return None if still running, returncode once finished."""
        if self._thread.is_alive():
            return None
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until done.  Raises subprocess.TimeoutExpired on timeout."""
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise subprocess.TimeoutExpired(cmd="<PipelineRunner>", timeout=timeout)
        return self.returncode

    def terminate(self) -> None:
        self._cancelled.set()
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)

    def kill(self) -> None:
        self.terminate()
