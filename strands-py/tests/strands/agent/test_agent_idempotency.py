"""Idempotency-token behavior across concurrent threads.

Where a race needs a specific thread interleaving, the test wraps a method of the agent's own
``_concurrency`` controller instance to hold a thread at a point where the interpreter can preempt
it anyway; the wrappers do not change what the wrapped method does.
"""

import contextvars
import threading

import pytest

from strands import Agent
from strands.types.exceptions import ConcurrencyException
from tests.fixtures.mocked_model_provider import MockedModelProvider

_WAIT = 5.0
_caller = contextvars.ContextVar("caller", default=None)


class _FailFirstModel(MockedModelProvider):
    """Raises on the first model call, then serves the scripted responses."""

    def __init__(self, agent_responses):
        super().__init__(agent_responses)
        self.calls = 0

    async def stream(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("primary failed")
        async for event in super().stream(*args, **kwargs):
            yield event


def _run_as(name, fn, outcomes):
    """Run ``fn`` in a thread whose context carries ``name``; record its return value or exception."""

    def target():
        _caller.set(name)
        try:
            outcomes[name] = fn()
        except BaseException as error:  # noqa: BLE001 - the test inspects the outcome
            outcomes[name] = error

    thread = threading.Thread(target=target, name=name)
    thread.start()
    return thread


def _wait(event, what):
    assert event.wait(_WAIT), f"timed out waiting for {what}"


def _stale_complete_agent():
    """Agent whose controller holds threads so that they interleave as follows.

    Primary A registers token T and fails; A's ``except`` path completes its registration; retry C
    registers T; duplicate D waits on C; A's ``finally`` path completes; A releases the lock; C
    acquires the lock and runs.
    """
    agent = Agent(
        model=_FailFirstModel([{"role": "assistant", "content": [{"text": "second attempt"}]}]),
        callback_handler=None,
    )
    controller = agent._concurrency
    events = {name: threading.Event() for name in ("a_first_complete", "c_registered", "d_waiting", "a_released")}

    original_complete = controller.complete
    original_check = controller._check_idempotency
    original_begin = controller.begin
    original_release = controller.release_lock

    def complete(registered_token, *, result=None, error=None):
        original_complete(registered_token, result=result, error=error)
        if _caller.get() == "A" and error is not None:
            # A is between its `except` complete (agent.py:1425) and its `finally` complete (agent.py:1443).
            events["a_first_complete"].set()
            _wait(events["d_waiting"], "D to wait on C's record")

    def check_idempotency(token):
        outcome = original_check(token)
        if _caller.get() == "C":
            # C registered T and has not tried the lock yet (_concurrency.py:120-126).
            events["c_registered"].set()
            _wait(events["a_released"], "A to release the lock")
        return outcome

    def begin(token):
        outcome = original_begin(token)
        if _caller.get() == "D" and outcome.waiting_on is not None:
            events["d_waiting"].set()
        return outcome

    def release_lock():
        original_release()
        if _caller.get() == "A":
            events["a_released"].set()

    controller.complete = complete
    controller._check_idempotency = check_idempotency
    controller.begin = begin
    controller.release_lock = release_lock
    return agent, events


def test_idempotency_duplicate_of_a_retry_gets_the_retry_result_after_a_failed_primary():
    """A duplicate gets the outcome of the primary it waits on, even after an earlier primary failed.

    Retry C registers token T between failed primary A's two ``complete`` calls, and duplicate D
    waits on C. A's second call must not settle C's registration.
    """
    agent, events = _stale_complete_agent()
    outcomes = {}

    thread_a = _run_as("A", lambda: agent("first", idempotency_token="T"), outcomes)
    _wait(events["a_first_complete"], "A's except-path complete")
    thread_c = _run_as("C", lambda: agent("retry", idempotency_token="T"), outcomes)
    _wait(events["c_registered"], "C to register T")
    thread_d = _run_as("D", lambda: agent("duplicate", idempotency_token="T"), outcomes)
    for thread in (thread_a, thread_c, thread_d):
        thread.join(_WAIT)
        assert not thread.is_alive(), f"{thread.name} hung"

    assert isinstance(outcomes["A"], RuntimeError)
    assert not isinstance(outcomes["C"], BaseException), outcomes["C"]
    assert str(outcomes["C"]) == "second attempt\n"
    assert not isinstance(outcomes["D"], BaseException), f"duplicate D raised {outcomes['D']!r}"
    assert str(outcomes["D"]) == str(outcomes["C"])


def test_idempotency_retry_waits_for_inflight_primary_after_a_failed_primary():
    """While retry C runs as the primary for token T, another call with T waits for C's result."""
    agent, events = _stale_complete_agent()
    c_running = threading.Event()
    e_began = threading.Event()
    controller = agent._concurrency
    wrapped_begin = controller.begin

    def begin(token):
        outcome = wrapped_begin(token)
        if _caller.get() == "E":
            e_began.set()
        return outcome

    controller.begin = begin
    model = agent.model
    original_stream = model.stream

    async def stream(*args, **kwargs):
        if _caller.get() == "C":
            c_running.set()
            _wait(e_began, "E to call begin")
        async for event in original_stream(*args, **kwargs):
            yield event

    model.stream = stream
    outcomes = {}

    thread_a = _run_as("A", lambda: agent("first", idempotency_token="T"), outcomes)
    _wait(events["a_first_complete"], "A's except-path complete")
    thread_c = _run_as("C", lambda: agent("retry", idempotency_token="T"), outcomes)
    _wait(events["c_registered"], "C to register T")
    events["d_waiting"].set()  # no duplicate D in this scenario
    _wait(c_running, "C to run")
    thread_e = _run_as("E", lambda: agent("retry again", idempotency_token="T"), outcomes)
    for thread in (thread_a, thread_c, thread_e):
        thread.join(_WAIT)
        assert not thread.is_alive(), f"{thread.name} hung"

    assert not isinstance(outcomes["C"], BaseException), outcomes["C"]
    assert not isinstance(outcomes["E"], ConcurrencyException), "retry E was rejected while C ran with token T"
    assert str(outcomes["E"]) == str(outcomes["C"])


@pytest.fixture(autouse=True)
def _reset_caller():
    token = _caller.set(None)
    yield
    _caller.reset(token)
