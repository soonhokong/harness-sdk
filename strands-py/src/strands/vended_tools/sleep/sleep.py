"""Sleep tool: pause execution for a bounded, cooperative duration.

Provides :func:`make_sleep` (a factory that lets the caller configure the
maximum permitted duration) and :data:`sleep` (a default instance with a 60-second
cap). Sleeps are implemented with :func:`asyncio.sleep` and watch the agent's
cancel signal: ``agent.cancel()`` ends a sleep early with
:class:`asyncio.CancelledError`, which the tool executor surfaces to the model as
a cancelled tool-error result. Cancelling the task that runs the agent interrupts
the sleep and propagates as :class:`asyncio.CancelledError`; direct callers awaiting
the underlying coroutine observe that cancellation directly.
"""

from __future__ import annotations

import asyncio
import math
import threading
from typing import TYPE_CHECKING

from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import DEFAULT_MAX_DURATION, sleep_description

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool


def make_sleep(
    *,
    max_duration: float = DEFAULT_MAX_DURATION,
    name: str = "sleep",
    description: str | None = None,
) -> DecoratedFunctionTool:
    """Create a sleep tool with a configurable maximum duration.

    The returned tool pauses execution for ``duration`` seconds via
    :func:`asyncio.sleep`. ``agent.cancel()`` ends the sleep early rather than
    waiting for the full duration: the tool raises :class:`asyncio.CancelledError`,
    which the tool executor surfaces to the model as a cancelled tool-error result.
    Cancelling the task that runs the agent also interrupts the sleep; that
    cancellation propagates.

    Args:
        max_duration: Upper bound on ``duration`` in seconds. Must be a finite,
            positive number. Defaults to :data:`DEFAULT_MAX_DURATION` (60 s).
        name: Tool name. Defaults to ``"sleep"``.
        description: Tool description shown to the model.

    Returns:
        A decorated tool that pauses execution for the requested duration.

    Raises:
        ValueError: If ``max_duration`` is not a positive, finite number.
    """
    if not isinstance(max_duration, (int, float)) or isinstance(max_duration, bool):
        raise ValueError(f"max_duration must be a number, got {type(max_duration).__name__}")
    if not math.isfinite(max_duration) or max_duration <= 0:
        raise ValueError(f"max_duration must be positive and finite, got {max_duration!r}")

    resolved_max = float(max_duration)
    resolved_description = description if description is not None else sleep_description(resolved_max)

    @tool(name=name, description=resolved_description, context=True)
    async def sleep_tool(duration: float, tool_context: ToolContext | None = None) -> str:
        """Pauses execution for the given number of seconds.

        The sleep is cooperative: it aborts when the agent is cancelled or the
        enclosing task is cancelled. Negative, non-finite, non-numeric, or
        oversized durations are rejected before the sleep begins.

        Args:
            duration: Seconds to pause. Must be a finite, non-negative number
                no larger than the tool's configured maximum.
            tool_context: Framework-injected. Not model-visible. Carries the
                agent's cancel signal.
        """
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(f"duration must be a number, got {type(duration).__name__}")
        seconds = float(duration)
        if not math.isfinite(seconds):
            raise ValueError(f"duration must be a finite number, got {duration!r}")
        if seconds < 0:
            raise ValueError(f"duration must be non-negative, got {seconds}")
        if seconds > resolved_max:
            raise ValueError(f"duration {seconds} exceeds maximum of {resolved_max} seconds")

        await _sleep(seconds, tool_context.cancel_signal if tool_context is not None else None)
        return f"Slept for {duration} seconds"

    return sleep_tool


_CANCEL_POLL_INTERVAL = 0.05
"""Seconds between checks of the agent's cancel signal while sleeping."""


async def _sleep(seconds: float, cancel_signal: threading.Event | None) -> None:
    """Sleep for ``seconds``; raise :class:`asyncio.CancelledError` early once ``cancel_signal`` is set."""
    if cancel_signal is None:
        await asyncio.sleep(seconds)
        return
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while not cancel_signal.is_set():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        # threading.Event has no async wait; poll it on this loop (as the MCP client does).
        await asyncio.sleep(min(remaining, _CANCEL_POLL_INTERVAL))
    raise asyncio.CancelledError("Sleep cancelled")


sleep = make_sleep()
"""Default sleep tool with a 60-second maximum duration."""
