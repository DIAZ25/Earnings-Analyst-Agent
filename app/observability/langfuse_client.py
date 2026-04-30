"""
LangFuse observability wrapper.

If LangFuse credentials are not configured, the @observe decorator
is a no-op so the system works without it.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, Any


def _is_langfuse_configured() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


def observe(name: str) -> Callable:
    """
    Decorator that wraps a node function with LangFuse tracing.
    Falls back to a passthrough if LangFuse is not configured.
    """
    def decorator(fn: Callable) -> Callable:
        if not _is_langfuse_configured():
            return fn  # No-op passthrough

        try:
            from langfuse.decorators import observe as lf_observe
            return lf_observe(name=name)(fn)
        except ImportError:
            return fn

    return decorator


def get_langfuse_client():
    """Return a LangFuse client instance, or None if not configured."""
    if not _is_langfuse_configured():
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except ImportError:
        return None


def score_trace(trace_id: str, name: str, value: float, comment: str = "") -> None:
    """Post a numeric score to a LangFuse trace."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception:
        pass
