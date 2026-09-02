"""Thread-safe circuit breaker (reliability pattern).

Three states:
  CLOSED   — normal operation; every call passes through.
  OPEN     — fast-fail; calls raise CircuitOpenError immediately.
  HALF_OPEN — probe mode; allows `half_open_max_calls` calls through.
              If any succeeds → CLOSED; if any fails → OPEN again.

Usage:
    cb = CircuitBreaker(name="groq", failure_threshold=5, recovery_timeout_s=30)

    try:
        result = cb.call(my_fn, arg1, arg2)
    except CircuitOpenError:
        # fast-fail path — try next provider
        ...
    except Exception as exc:
        # real failure from my_fn
        ...
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable

log = logging.getLogger("controlplane.circuit_breaker")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(Exception):
    """Raised by CircuitBreaker.call() when the circuit is OPEN (fast-fail)."""
    def __init__(self, name: str, retry_after_s: float) -> None:
        self.name = name
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Circuit '{name}' is OPEN — fast-failing. "
            f"Retry after {retry_after_s:.1f}s."
        )


class CircuitBreaker:
    """
    Simple, thread-safe circuit breaker.

    Parameters
    ----------
    name : str
        Human-readable label (logged with every state transition).
    failure_threshold : int
        Consecutive failures before tripping OPEN.  Default 5.
    recovery_timeout_s : float
        Seconds to wait in OPEN before probing with HALF_OPEN.  Default 30.
    half_open_max_calls : int
        How many probe calls to allow in HALF_OPEN.  Default 2.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_opened_at: float = 0.0
        self._half_open_calls: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._check_and_maybe_transition()

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* through the circuit breaker.

        Raises
        ------
        CircuitOpenError
            When the circuit is OPEN — caller should fast-fail / skip provider.
        Exception
            Whatever *fn* raises (after recording the failure).
        """
        with self._lock:
            state = self._check_and_maybe_transition()

            if state == CircuitState.OPEN:
                retry_after = self.recovery_timeout_s - (time.monotonic() - self._last_opened_at)
                raise CircuitOpenError(self.name, max(retry_after, 0.0))

            if state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        # Execute outside the lock so we don't block other threads
        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            raise

    def reset(self) -> None:
        """Force the circuit to CLOSED (useful in tests / manual recovery)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_opened_at = 0.0
            self._half_open_calls = 0
            log.info("CircuitBreaker '%s' manually reset to CLOSED.", self.name)

    def status(self) -> dict:
        """Return a JSON-serialisable status dict for health endpoints."""
        with self._lock:
            state = self._check_and_maybe_transition()
        return {
            "name": self.name,
            "state": state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout_s,
        }

    # ------------------------------------------------------------------
    # Internal helpers (must be called with _lock held where noted)
    # ------------------------------------------------------------------

    def _check_and_maybe_transition(self) -> CircuitState:
        """If OPEN and recovery timeout has elapsed, move to HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_opened_at
            if elapsed >= self.recovery_timeout_s:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                log.info(
                    "CircuitBreaker '%s': OPEN → HALF_OPEN after %.1fs.",
                    self.name, elapsed,
                )
        return self._state

    def _on_success(self) -> None:
        with self._lock:
            prev = self._state
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._half_open_calls = 0
            if prev != CircuitState.CLOSED:
                log.info(
                    "CircuitBreaker '%s': %s → CLOSED (success).", self.name, prev.value
                )

    def _on_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN → back to OPEN
                self._state = CircuitState.OPEN
                self._last_opened_at = time.monotonic()
                log.warning(
                    "CircuitBreaker '%s': HALF_OPEN → OPEN (probe failed: %s).",
                    self.name, exc,
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._last_opened_at = time.monotonic()
                log.error(
                    "CircuitBreaker '%s': CLOSED → OPEN after %d consecutive failures.",
                    self.name, self._failure_count,
                )
