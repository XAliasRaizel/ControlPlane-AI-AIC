"""ControlPlane.ai — FastAPI Entrypoint (Section 5.1).

This is the single ingress point for every AI interaction. It wires
together all components: gateway, detectors, risk, policy, decision,
audit, and async pipeline.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request as FastAPIRequest
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.shared.config import settings
from backend.shared.schemas import (
    DecisionAction,
    FeedbackRequest,
    GovernanceDecision,
    GovernanceRequest,
    GovernanceResponse,
    PolicyMatch,
    PolicySummary,
    RiskAssessment,
)
from backend.audit.store import Database, build_audit_context, fingerprint
from backend.gateway.auth import verify_api_key, verify_admin_key
from backend.gateway.context_enrichment import enrich_context
from backend.review.queue import ReviewQueue
from backend.feedback.evaluator import FeedbackEvaluator
from backend.shared.gpu_adapter import GPUAdapter
from backend.shared import llm_simulator

# Trigger detector self-registration by importing the package
from backend.detectors import DETECTOR_REGISTRY, run_hot_path  # noqa: F401
from backend.risk.engine import calculate_risk
from contextlib import asynccontextmanager

from backend.policy.engine import evaluate_policy, policy_engine
from backend.decision.engine import make_decision, sanitize_response
from backend.agents.router import router as agent_router
from rlhf.sampler import maybe_collect_pair

from backend.shared.logging_config import configure_logging, request_id_var, trace_id_var
from backend.shared.metrics import record_govern, refresh_dead_letter_count
from backend.shared.tracing import configure_tracing, get_current_trace_id
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Configure structured logging and OpenTelemetry tracing
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
configure_tracing(
    service_name=settings.otel_service_name,
    otlp_endpoint=settings.otlp_endpoint,
)
logger = logging.getLogger("controlplane.gateway")
# unbounded memory growth under spike traffic.
# ---------------------------------------------------------------------------
class AsyncTaskQueue:
    """Wraps asyncio.Queue with a pool of N worker coroutines.

    Callers enqueue (coro_func, *args, **kwargs) tuples.  Workers pull from
    the queue and await each task.  If the queue is full, enqueue() drops the
    task and logs a warning (back-pressure) rather than blocking the request.
    """

    def __init__(self, maxsize: int = 500, num_workers: int = 4):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._num_workers = num_workers
        self._workers: list[asyncio.Task] = []

    async def start(self):
        self._workers = [
            asyncio.create_task(self._drain(), name=f"task-queue-worker-{i}")
            for i in range(self._num_workers)
        ]

    async def stop(self):
        for _ in self._workers:
            await self._queue.put(None)  # sentinel
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def _drain(self):
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            coro_func, args, kwargs = item
            try:
                await coro_func(*args, **kwargs)
            except Exception as exc:
                _q_logger = logging.getLogger("controlplane.task_queue")
                _q_logger.warning("Task queue worker error: %s", exc)
            finally:
                self._queue.task_done()

    def enqueue(self, coro_func, *args, **kwargs) -> bool:
        """Non-blocking enqueue. Returns False and warns if queue is full."""
        try:
            self._queue.put_nowait((coro_func, args, kwargs))
            return True
        except asyncio.QueueFull:
            _q_logger = logging.getLogger("controlplane.task_queue")
            _q_logger.warning(
                "AsyncTaskQueue full (%d/%d) — dropping background task %s. "
                "Increase CONTROLPLANE_ASYNC_QUEUE_SIZE to handle higher traffic.",
                self._queue.qsize(), self._queue.maxsize, coro_func.__name__,
            )
            return False


# Singleton — initialised in lifespan
_task_queue: AsyncTaskQueue | None = None


# ---------------------------------------------------------------------------
# TTLCache — simple time-aware dict for expensive read-only SQLite scans
# ---------------------------------------------------------------------------
class TTLCache:
    """Tiny in-memory TTL cache. Thread-safe, no external dependencies."""

    def __init__(self, ttl_seconds: int):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = __import__("threading").Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value) -> None:
        with self._lock:
            self._store[key] = (value, time.time() + self._ttl)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)


# Module-level cache instances (created after settings is imported)
_metrics_cache: TTLCache | None = None
_audits_cache: TTLCache | None = None


def _get_metrics_cache() -> TTLCache:
    global _metrics_cache
    if _metrics_cache is None:
        _metrics_cache = TTLCache(settings.metrics_cache_ttl_s)
    return _metrics_cache


def _get_audits_cache() -> TTLCache:
    global _audits_cache
    if _audits_cache is None:
        _audits_cache = TTLCache(settings.metrics_cache_ttl_s)
    return _audits_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _task_queue
    # Start bounded async task queue (upstream feature)
    _task_queue = AsyncTaskQueue(
        maxsize=settings.async_queue_size,
        num_workers=4,
    )
    await _task_queue.start()
    logger.info(
        "AsyncTaskQueue started (maxsize=%d, workers=4)",
        settings.async_queue_size,
    )

    # Parallel warmup of all ML models, Presidio, and Policy RAG for fast cold-start
    import concurrent.futures
    logger.info("Initializing background warm-up for ML models, Presidio, and Policy RAG...")

    def _warmup_rag():
        try:
            from rag.policy.policy_rag import get_policy_evidence
            get_policy_evidence(
                user_role="employee",
                application_id="support-bot",
                department="HR",
                matched_rule_description="test",
                data_classification="PUBLIC",
                action="ALLOW",
            )
            logger.info("Policy RAG warm-up complete.")
        except Exception as exc:
            logger.warning("Policy RAG warm-up skipped: %s", exc)

    def _warmup_presidio():
        try:
            from backend.shared.model_backend import consult_presidio
            consult_presidio("warm-up")
            logger.info("Presidio warm-up complete.")
        except Exception as exc:
            logger.warning("Presidio warm-up skipped: %s", exc)

    def _warmup_detector(task: str):
        try:
            from backend.shared.model_backend import (
                consult,
                consult_sensitive_intent,
                get_grounding_scorer,
            )
            if task == "grounding":
                scorer = get_grounding_scorer(task)
                if scorer:
                    scorer._ensure_model()
            elif task == "sensitive_intent":
                consult_sensitive_intent("warmup query intent")
            else:
                consult(task, "warmup text")
            logger.info("Model %s warm-up complete.", task)
        except Exception as exc:
            logger.warning("Model %s warm-up skipped: %s", task, exc)

    # Run all warmups concurrently across thread workers for fast cold-start
    warmup_tasks = ["injection", "safety", "fairness", "grounding", "sensitive_intent"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        f_rag = executor.submit(_warmup_rag)
        f_presidio = executor.submit(_warmup_presidio)
        f_models = [executor.submit(_warmup_detector, t) for t in warmup_tasks]
        concurrent.futures.wait([f_rag, f_presidio] + f_models, timeout=20.0)

    logger.info("All components pre-warmed. FastAPI ready to accept traffic.")
    yield
    # Graceful shutdown: drain and stop the task queue
    await _task_queue.stop()
    logger.info("AsyncTaskQueue drained and stopped.")
    try:
        from backend.shared.model_backend import _EXECUTOR
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        logger.info("ML ThreadPoolExecutor shut down.")
    except Exception as exc:
        logger.warning("ML ThreadPoolExecutor shutdown: %s", exc)


app = FastAPI(
    title="ControlPlane.ai",
    description="Enterprise AI governance control plane — observe, reason, act, learn.",
    version="0.4.0",
    lifespan=lifespan,
)
app.include_router(agent_router)

# ---------------------------------------------------------------------------
# Rate limiter (slowapi)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS — allow Streamlit frontend + any extra origins from env
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers on every response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

app.add_middleware(SecurityHeadersMiddleware)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject request_id and trace_id into log contextvars and response headers.

    Reads X-Request-ID header if provided (useful for distributed tracing);
    otherwise generates a new UUID4. Sets X-Request-ID on the response so
    clients can correlate logs.
    """

    async def dispatch(self, request: Request, call_next):
        # Honour an upstream X-Request-ID (e.g. from an API gateway)
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        token_req = request_id_var.set(req_id)

        # Propagate OTel trace_id into logs when a span is active
        try:
            from backend.shared.tracing import get_current_trace_id
            trace_id = get_current_trace_id()
        except Exception:
            trace_id = "-"
        token_trace = trace_id_var.set(trace_id)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_var.reset(token_req)
            trace_id_var.reset(token_trace)


app.add_middleware(RequestIDMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: FastAPIRequest, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check backend logs for details."},
    )

db = Database(settings.db_path)
review_queue = ReviewQueue(db=db)
feedback_evaluator = FeedbackEvaluator()
gpu = GPUAdapter()

# Tamper-Evident Audit Ledger — wired to all governance decisions.
# Uses a separate SQLite table (audit_records) + an append-only anchor
# file (.integrity.jsonl) so the chain can be verified independently.
import pathlib as _pathlib
from backend.audit_integrity.backends import AuditRecordBackend, AnchorBackend
from backend.audit_integrity.ledger import TamperEvidentAuditLedger

_ledger_db_path   = settings.db_path
_anchor_path      = str(_pathlib.Path(_ledger_db_path).with_suffix(".integrity.jsonl"))
_hmac_secret      = settings.audit_hash_key.encode("utf-8")
_ledger_records   = AuditRecordBackend(_ledger_db_path)
_ledger_anchors   = AnchorBackend(_anchor_path)
ledger            = TamperEvidentAuditLedger(
    record_backend=_ledger_records,
    anchor_backend=_ledger_anchors,
    hmac_secret=_hmac_secret,
    checkpoint_interval=5,          # seal a Merkle checkpoint every 5 records
)


@app.post("/admin/reload-models", tags=["admin"])
async def reload_models(_admin_key: str = Depends(verify_admin_key)):
    """Hot-reload all ML models and clear caches for zero-downtime updates."""
    from backend.shared.model_backend import reset_cache
    reset_cache()
    logger.info("Model cache cleared via /admin/reload-models")
    return {"status": "ok", "message": "Model cache cleared. Models will be lazily reloaded on next request."}

class ReviewResolution(BaseModel):
    reviewer_id: str = "reviewer"
    final_action: DecisionAction = "BLOCK"
    notes: str = ""


# ---------------------------------------------------------------------------
# Health & metrics
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": gpu.status(),
        "registered_detectors": list(DETECTOR_REGISTRY.keys()),
        "policy": policy_engine.summary().model_dump(),
    }


@app.get("/metrics", tags=["observability"])
async def prometheus_metrics():
    """Prometheus metrics endpoint — returns text/plain exposition format.

    This endpoint is intentionally unauthenticated so Prometheus can scrape it
    without API key management. Protect via network policy / firewall in production.
    """
    from fastapi.responses import Response as FastAPIResponse

    # Refresh dead-letter count gauge before each scrape
    refresh_dead_letter_count()

    data = generate_latest()
    return FastAPIResponse(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/health/deep", tags=["health"])
async def health_deep():
    """Deep liveness probe — verifies DB writable, LLM provider reachable, session store writable.

    Returns component-level status. Suitable for Kubernetes liveness/readiness probes.
    Returns HTTP 503 if any critical component is unhealthy.
    """
    import uuid as _uuid
    from backend.app.llm.client import _PROVIDER_BREAKERS
    from backend.risk.session_store import get_session_store, reset_store_cache, SessionState

    components: dict[str, dict] = {}
    overall_ok = True

    # 1. Database write check
    try:
        probe_id = f"health-probe-{_uuid.uuid4().hex[:8]}"
        db.create_job(probe_id, probe_id)
        db.update_job(probe_id, "HEALTH_CHECK", {"probe": True})
        components["database"] = {"status": "ok", "mode": "WAL"}
    except Exception as exc:
        components["database"] = {"status": "error", "detail": str(exc)}
        overall_ok = False

    # 2. LLM provider circuit breaker status (no real network call — just CB state)
    cb_statuses = {name: cb.status() for name, cb in _PROVIDER_BREAKERS.items()}
    any_provider_closed = any(
        s["state"] == "CLOSED" for s in cb_statuses.values()
    )
    components["llm_providers"] = {
        "status": "ok" if any_provider_closed else "degraded",
        "circuit_breakers": cb_statuses,
    }
    # Degraded LLM is not critical (extractive fallback exists)

    # 3. Session store write check
    try:
        store = get_session_store()
        probe_state = SessionState(
            session_id="__health_probe__",
            created_at=time.time(),
            last_updated_at=time.time(),
        )
        store.set("__health_probe__", probe_state, ttl_seconds=10)
        recovered = store.get("__health_probe__")
        store.delete("__health_probe__")
        if recovered is None:
            raise RuntimeError("session not found after write")
        components["session_store"] = {"status": "ok", "backend": type(store).__name__}
    except Exception as exc:
        components["session_store"] = {"status": "error", "detail": str(exc)}
        overall_ok = False

    # 4. Async ML thread pool check
    try:
        from backend.shared.model_backend import _EXECUTOR
        components["ml_executor"] = {
            "status": "ok",
            "max_workers": getattr(_EXECUTOR, "_max_workers", 4),
        }
    except Exception as exc:
        components["ml_executor"] = {"status": "error", "detail": str(exc)}

    status_code = 200 if overall_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if overall_ok else "degraded",
            "components": components,
        },
    )


# ---------------------------------------------------------------------------
# Dead-letter admin endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/admin/dead-letters", tags=["admin"])
async def list_dead_letters(
    request: Request,
    limit: int = 50,
    _admin_key: str = Depends(verify_admin_key),
):
    """List failed async events written to the dead-letter store."""
    from backend.async_pipeline.publisher import _get_dead_letter_store
    return _get_dead_letter_store().list_all(limit=min(limit, 500))


@app.post("/v1/admin/dead-letters/{record_id}/retry", tags=["admin"])
async def retry_dead_letter(
    request: Request,
    record_id: str,
    background_tasks: BackgroundTasks,
    _admin_key: str = Depends(verify_admin_key),
):
    """Re-enqueue a dead-letter record for processing and mark it as retried."""
    from backend.async_pipeline.publisher import _get_dead_letter_store, publish_event
    store = _get_dead_letter_store()
    records = store.list_all(limit=500)
    record = next((r for r in records if r["id"] == record_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Dead-letter record not found")
    # Re-publish the event using stored request_id + a fresh GovernanceRequest stub
    store.mark_retried(record_id)
    return {"status": "queued", "record_id": record_id, "job_id": record.get("job_id")}


@app.delete("/v1/admin/dead-letters/{record_id}", tags=["admin"])
async def delete_dead_letter(
    request: Request,
    record_id: str,
    _admin_key: str = Depends(verify_admin_key),
):
    """Delete a dead-letter record (e.g. after manual remediation)."""
    from backend.async_pipeline.publisher import _get_dead_letter_store
    _get_dead_letter_store().delete(record_id)
    return {"status": "deleted", "record_id": record_id}


# ---------------------------------------------------------------------------
# Fast-lane background task
# ---------------------------------------------------------------------------
async def run_fast_lane(request: GovernanceRequest, async_job_id: str):
    fast_detectors = [d for d in DETECTOR_REGISTRY.values() if getattr(d, 'fast_async', False)]
    if not fast_detectors:
        return

    start = time.perf_counter()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[d.analyze(request, {}) for d in fast_detectors]),
            timeout=0.250
        )
        latency_ms = (time.perf_counter() - start) * 1000
    except asyncio.TimeoutError:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("fast_lane_decision request_id=%s corrections=0 latency=%.1fms timeout=True option=none", request.request_id, latency_ms)
        return
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("Fast lane error: %s", exc)
        logger.info("fast_lane_decision request_id=%s corrections=0 latency=%.1fms timeout=False option=none error=True", request.request_id, latency_ms)
        return

    try:
        job = db.get_job(async_job_id)
        if job:
            result_data = job.get("result") or {}
            if isinstance(result_data, str):
                try:
                    result_data = json.loads(result_data)
                except Exception:
                    result_data = {}
            result_data["fast_lane_results"] = [r.model_dump(mode="json") for r in results]
            status = "FAST_LANE_COMPLETED" if job.get("status") == "QUEUED" else job.get("status")
            db.update_job(async_job_id, status, result_data)
    except Exception as e:
        logger.warning("Could not update fast lane job in DB: %s", e)

    high_risk = [r for r in results if r.score > 0.65 or r.label in ("HIGH", "BIASED")]
    correction_count = len(high_risk)
    option_used = "option2" if request.fast_lane_webhook else "option1"

    logger.info(
        "fast_lane_decision request_id=%s corrections=%d latency=%.1fms timeout=False option=%s",
        request.request_id, correction_count, latency_ms, option_used
    )

    if high_risk and request.fast_lane_webhook:
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                await client.post(request.fast_lane_webhook, json={
                    "request_id": request.request_id,
                    "action": "RETRACT",
                    "reason": "Fast-lane analysis detected high risk.",
                    "detectors": [r.model_dump(mode="json") for r in high_risk]
                })
        except Exception as e:
            logger.warning("Failed to push fast lane correction to webhook: %s", e)


# ---------------------------------------------------------------------------
# Core governance endpoint (Section 5.1)
# ---------------------------------------------------------------------------
async def execute_governance(
    request: GovernanceRequest,
    background_tasks: BackgroundTasks,
    x_controlplane_session_id: Optional[str] = None,
) -> GovernanceResponse:
    # Populate session_id from header when not supplied in the request body.
    # Header takes lower priority than an explicit body field.
    if not request.session_id and x_controlplane_session_id:
        request = request.model_copy(update={"session_id": x_controlplane_session_id})
    if len(request.prompt) > settings.max_prompt_chars:
        raise HTTPException(status_code=413, detail="Prompt exceeds configured maximum length.")

    start = time.perf_counter()

    # Assign request_id and timestamp if not provided
    if not request.request_id:
        request.request_id = str(uuid.uuid4())
    if not request.timestamp:
        request.timestamp = datetime.now(timezone.utc)

    request_id = request.request_id

    # 1. Context enrichment (Section 5.2)
    context = enrich_context(request)

    # 2. Hot path — parallel detectors (Section 5.3)
    detector_results, hot_path_ms = await run_hot_path(request, context)

    # 3. Risk engine (Section 5.4)
    risk = calculate_risk(request, detector_results, context)

    # 4. Policy engine (Section 5.5)
    policy = evaluate_policy(request, risk, context)

    # 5. Decision engine (Section 5.6)
    decision = make_decision(request, risk, policy)

    # 6. Human review queue
    decision = review_queue.enqueue(decision, prompt=request.prompt or "")

    # 7. LLM generation & response sanitization
    # If no candidate response was provided and the decision is ALLOW or MODIFY, generate one!
    candidate_response = request.response
    if candidate_response is None and decision.action in {"ALLOW", "MODIFY"}:
        candidate_response = llm_simulator.generate(
            prompt=request.prompt,
            user_id=request.user_id,
            app_id=request.application_id,
        )

    sanitized = sanitize_response(candidate_response, decision)

    # RLHF sampling — fire-and-forget, never affects latency.
    # 1-in-N requests triggers dual-response generation and storage.
    if request.prompt:
        background_tasks.add_task(maybe_collect_pair, request, sanitized or candidate_response, context)


    latency_ms = (time.perf_counter() - start) * 1000


    # 8. Audit (Section 5.9)
    audit_ctx = build_audit_context(request)
    db.save_request(
        request_id=request_id,
        audit_context=audit_ctx,
        decision=decision.action,
        risk=risk.overall_risk,
        latency_ms=latency_ms,
        prompt_fingerprint=fingerprint(request.prompt),
        detector_results=[r.model_dump(mode="json") for r in detector_results],
        risk_details=risk.model_dump(mode="json"),
        policy=policy.model_dump(mode="json"),
        decision_details=decision.model_dump(mode="json"),
    )
    # Invalidate cached metrics so dashboards see fresh counts within one TTL window
    _get_metrics_cache().invalidate("metrics")
    _get_metrics_cache().invalidate("metrics_rich")

    # Tamper-evident Merkle ledger — append alongside the SQLite audit.
    # Fail-open: a ledger error must never interrupt the governance response.
    try:
        ledger.append({
            "request_id":        request_id,
            "prompt_fingerprint": fingerprint(request.prompt),
            "decision":          decision.action,
            "risk":              round(risk.overall_risk, 4),
            "policy_id":         policy.policy_id,
            "latency_ms":        round(latency_ms, 2),
            "user_role":         request.user_role or "unknown",
            "application_id":    request.application_id or "unknown",
        })
    except Exception as _ledger_exc:
        logger.warning("Merkle ledger append failed (non-fatal): %s", _ledger_exc)

    logger.info(
        "governance_decision request_id=%s action=%s risk=%.3f latency=%.1fms fingerprint=%s",
        request_id,
        decision.action,
        risk.overall_risk,
        latency_ms,
        fingerprint(request.prompt),
        extra={
            "event": "GOVERN",
            "request_id": request_id,
            "decision": decision.action,
            "risk_score": round(risk.overall_risk, 4),
            "latency_ms": round(latency_ms, 2),
            "policy_id": policy.policy_id if policy else "unknown",
            "tenant_id": request.application_id or "default",
            "detector_count": len(detector_results),
            "session_id": request.session_id or "-",
        },
    )

    # Prometheus instrumentation
    record_govern(
        decision=decision.action,
        tenant_id=request.application_id or "default",
        policy_id=policy.policy_id if policy else "unknown",
        latency_seconds=latency_ms / 1000.0,
    )

    # --- Phase 9: Session telemetry + entity reconstruction hook ---
    if risk.session_risk is not None and request.session_id:
        session_ctx = risk.contextual_factors.get("session", {})
        logger.info(
            "session_telemetry request_id=%s session_id=%s session_risk=%.3f "
            "session_band=%d ewma=%.4f peak=%.4f turn_count=%d "
            "contamination_active=%s fast_lane_corrections=%d",
            request_id,
            request.session_id,
            risk.session_risk,
            risk.session_band or 1,
            session_ctx.get("ewma", 0.0),
            session_ctx.get("peak", 0.0),
            session_ctx.get("turn_count", 0),
            session_ctx.get("contamination_active", False),
            session_ctx.get("fast_lane_correction_count", 0),
        )

    # Entity reconstruction check: detect PII split across turns.
    # Run after risk is computed; positive result boosts the NEXT turn's signal
    # (via the pii_fragment parameter of update_session) rather than retroactively
    # re-scoring this turn.
    if (
        request.session_id
        and os.environ.get("CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED", "").lower() == "true"
    ):
        try:
            import re
            from backend.risk.session_store import get_session_store
            from backend.risk.accumulator import check_entity_reconstruction
            from backend.detectors.pii import _VALUE_PATTERNS

            def _pii_check(text: str):
                class _Result:
                    triggered = any(
                        re.search(p, text, re.I) for p in _VALUE_PATTERNS.values()
                    )
                return _Result()

            _state = get_session_store().get(request.session_id)
            if _state:
                _recon = check_entity_reconstruction(_state, _pii_check)
                if _recon:
                    logger.info(
                        "entity_reconstruction_triggered request_id=%s session_id=%s turn_count=%d",
                        request_id,
                        request.session_id,
                        _state.turn_count,
                    )
        except Exception:
            pass  # fail closed
    # --- End Phase 9 session hooks ---

    # 9. Async path — fire and forget with DB tracking (Section 5.8)
    async_job_id: str = f"async-{request_id[:8]}"

    fast_async_detectors = [d for d in DETECTOR_REGISTRY.values() if getattr(d, 'fast_async', False)]
    fast_lane_pending = len(fast_async_detectors) > 0

    try:
        db.create_job(async_job_id, request_id)
        from backend.async_pipeline.publisher import publish_event
        # Include response in async analysis request so engines have full interaction context
        await publish_event(
            request_id,
            async_request,
            async_job_id,
            hot_path_risk=risk.overall_risk,
            hot_path_results=detector_results,
        )

        if fast_lane_pending:
            # Route through bounded AsyncTaskQueue; fall back to BackgroundTasks if queue unavailable
            if _task_queue is not None:
                _task_queue.enqueue(run_fast_lane, async_request, async_job_id)
            else:
                background_tasks.add_task(run_fast_lane, async_request, async_job_id)

    except Exception as exc:
        logger.warning("Async publish failed (non-blocking): %s", exc)

    # 10. Policy RAG explanation (never affects decision)
    policy_evidence = None
    try:
        from rag.policy.policy_rag import get_policy_evidence
        pe = get_policy_evidence(
            user_role=request.user_role or "user",
            application_id=request.application_id or "default",
            department=request.department or "General",
            matched_rule_description=policy.policy_id if policy else "",
            data_classification=request.data_classification or "PUBLIC",
            action=decision.action if decision else "ALLOW",
        )
        policy_evidence = pe.model_dump(mode="json")
    except Exception as exc:
        logger.warning("Policy RAG retrieval skipped: %s", exc)

    return GovernanceResponse(
        request_id=request_id,
        decision=decision,
        risk=risk,
        detectors=detector_results,
        policy=policy,
        sanitized_response=sanitized,
        async_job_id=async_job_id,
        policy_evidence=policy_evidence,
        fast_lane_pending=fast_lane_pending,
        session_risk=risk.session_risk,
        session_band=risk.session_band,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/v1/govern", response_model=GovernanceResponse)
@limiter.limit(f"{settings.rate_limit_govern}/minute")
async def govern(
    request: Request,
    payload: GovernanceRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
    x_controlplane_session_id: Optional[str] = Header(None),
):
    return await execute_governance(
        payload,
        background_tasks=background_tasks,
        x_controlplane_session_id=x_controlplane_session_id,
    )


# ---------------------------------------------------------------------------
# Chat endpoint (Interactive Chatbot with real-time governance)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str = "employee-101"
    user_role: str = "employee"
    department: str = "HR"
    application_id: str = "support-bot"
    prompt: str
    data_classification: Optional[str] = "PUBLIC"
    session_id: Optional[str] = None  # Phase 9 — pass through to GovernanceRequest


class ChatResponse(BaseModel):
    request_id: str
    action: str
    message: str
    governance: GovernanceResponse


@app.post("/v1/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_govern}/minute")
async def chat(
    request: Request,
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
):
    if len(payload.prompt) > settings.max_prompt_chars:
        raise HTTPException(status_code=413, detail="Prompt exceeds configured maximum length.")
    gov_req = GovernanceRequest(
        user_id=payload.user_id,
        user_role=payload.user_role,
        department=payload.department,
        application_id=payload.application_id,
        prompt=payload.prompt,
        data_classification=payload.data_classification,
        session_id=payload.session_id,
    )
    gov_resp = await execute_governance(gov_req, background_tasks=background_tasks)


    if gov_resp.decision.action == "BLOCK":
        reply = f"🛡️ [BLOCKED BY CONTROLPLANE.AI]\nYour request was blocked by enterprise governance policy ({gov_resp.policy.policy_id}).\nReason: {gov_resp.decision.reason}"
    elif gov_resp.decision.action == "HUMAN_REVIEW":
        reply = f"⏳ [HELD FOR HUMAN REVIEW]\nYour request involves high-risk data/policy and has been queued for human approval.\nReason: {gov_resp.decision.reason}"
    elif gov_resp.decision.action == "MODIFY":
        sanitized_text = gov_resp.sanitized_response or "Response sanitized by policy."
        reply = sanitized_text
    else:
        reply = gov_resp.sanitized_response or llm_simulator.generate(payload.prompt, payload.user_id, payload.application_id)

    return ChatResponse(
        request_id=gov_resp.request_id,
        action=gov_resp.decision.action,
        message=reply,
        governance=gov_resp,
    )


# ---------------------------------------------------------------------------
# Supporting & Async endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/jobs/{job_id}")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def get_job(request: Request, job_id: str, _api_key: str = Depends(verify_api_key)):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("result") is not None and isinstance(job["result"], str):
        try:
            job["result"] = json.loads(job["result"])
        except Exception:
            pass
    return job


@app.get("/v1/async/{request_id}")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def get_async_by_request(request: Request, request_id: str, _api_key: str = Depends(verify_api_key)):
    job_id = f"async-{request_id[:8]}"
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Async job for request not found")
    if job.get("result") is not None and isinstance(job["result"], str):
        try:
            job["result"] = json.loads(job["result"])
        except Exception:
            pass
    return job


@app.get("/v1/metrics")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def metrics(request: Request, _api_key: str = Depends(verify_api_key)):
    cache = _get_metrics_cache()
    cached = cache.get("metrics")
    if cached is not None:
        return cached
    result = db.metrics()
    cache.set("metrics", result)
    return result



@app.get("/v1/metrics/rich")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def rich_metrics(request: Request, _api_key: str = Depends(verify_api_key)):
    """Extended metrics: detector fire rates, risk/latency trends, blocked-by-rule breakdown."""
    cache = _get_metrics_cache()
    cached = cache.get("metrics_rich")
    if cached is not None:
        return cached
    result = db.richer_metrics()
    cache.set("metrics_rich", result)
    return result

@app.get("/v1/requests")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def requests_list(request: Request, limit: int = 50, _api_key: str = Depends(verify_api_key)):
    return db.recent_requests(min(limit, 200))


@app.get("/v1/audits")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def audits(request: Request, limit: int = 50, _api_key: str = Depends(verify_api_key)):
    cache = _get_audits_cache()
    cache_key = f"audits:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    result = db.recent_audits(min(limit, 200))
    cache.set(cache_key, result)
    return result


@app.get("/v1/audits/{request_id}")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def audit(request: Request, request_id: str, _api_key: str = Depends(verify_api_key)):
    record = db.get_audit(request_id)
    if not record:
        raise HTTPException(status_code=404, detail="Audit record not found")
    return record


@app.get("/v1/policies", response_model=PolicySummary)
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def policy_summary(request: Request, _api_key: str = Depends(verify_api_key)):
    return policy_engine.summary()


@app.post("/v1/feedback")
async def feedback(payload: FeedbackRequest):
    if not db.request_exists(payload.request_id):
        raise HTTPException(status_code=404, detail="Request not found")
    db.save_feedback(
        payload.request_id,
        payload.final_action,
        payload.original_action == payload.final_action,
        payload.notes,
    )
    # FIX: FeedbackEvaluator existed (backend/feedback/evaluator.py) but was
    # never imported or called anywhere -- this endpoint stored a bare
    # correct/incorrect boolean and threw away the false_positive vs.
    # false_negative classification the evaluator already knew how to make.
    classification = feedback_evaluator.record_override(
        request_id=payload.request_id,
        original_action=payload.original_action,
        final_action=payload.final_action,
        notes=payload.notes,
    )
    return {"status": "stored", **classification}


# ---------------------------------------------------------------------------
# Threshold Auto-Tuner endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/feedback/tuning", tags=["feedback"])
async def tuning_preview():
    """Dry-run the self-governing threshold auto-tuner.

    Returns what WOULD happen if the tuner ran — no policy files are changed.
    Use /v1/feedback/tuning/apply (POST) to actually write YAML changes.
    """
    from backend.feedback.feedback_engine import run_tuning_cycle
    try:
        return run_tuning_cycle(dry_run=True)
    except Exception as exc:
        logger.error("Tuning preview failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/feedback/tuning/apply", tags=["feedback"])
async def tuning_apply():
    """Apply threshold auto-tuning decisions to policy YAML files.

    NUDGE decisions raise a rule's detector threshold by one bounded step.
    ESCALATE decisions flag the rule for mandatory human review (no YAML change).
    Every applied change is logged with override rate, sample size, and reasoning.
    """
    from backend.feedback.feedback_engine import run_tuning_cycle
    try:
        return run_tuning_cycle(dry_run=False)
    except Exception as exc:
        logger.error("Tuning apply failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/feedback/tuning/seed-demo", tags=["feedback"])
async def tuning_seed_demo():
    """Seed realistic review override history to showcase NUDGE, ESCALATE, and HOLD."""
    from backend.feedback.feedback_engine import seed_demo_feedback_records
    try:
        return seed_demo_feedback_records()
    except Exception as exc:
        logger.error("Seeding demo feedback failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/feedback/tuning/history", tags=["feedback"])
async def tuning_history(limit: int = 50):
    """Retrieve audit trail of applied threshold tuning modifications."""
    from backend.feedback.feedback_engine import get_tuning_history
    try:
        return get_tuning_history(limit)
    except Exception as exc:
        logger.error("Loading tuning history failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/reviews")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def list_pending_reviews(request: Request, limit: int = 50, _api_key: str = Depends(verify_api_key)):
    return review_queue.list_pending(min(limit, 200))


@app.post("/v1/reviews/{request_id}/resolve")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def resolve_review(request: Request, request_id: str, payload: ReviewResolution, _api_key: str = Depends(verify_api_key)):
    if not review_queue.db.get_review(request_id):
        raise HTTPException(status_code=404, detail="No pending review for this request_id")
    return review_queue.resolve(
        request_id=request_id,
        final_action=payload.final_action,
        reviewer_id=payload.reviewer_id,
        notes=payload.notes,
    )


@app.get("/v1/gpu")
async def gpu_status():
    return gpu.status()


# ---------------------------------------------------------------------------
# Ask ControlPlane endpoints (RAG over policy & audit knowledge bases)
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


@app.post("/v1/ask-controlplane")
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def ask_controlplane(request: Request, payload: AskRequest, _api_key: str = Depends(verify_api_key)):
    if len(payload.question) > settings.max_prompt_chars:
        raise HTTPException(status_code=413, detail="Question exceeds configured maximum length.")
    from rag.ask_controlplane.chat import ask
    return ask(payload.question, db=db)


@app.post("/v1/ask-controlplane/reindex")
async def reindex_audit(_admin_key: str = Depends(verify_admin_key)):
    from rag.ask_controlplane.retrieval import rebuild_audit_index
    return {"indexed": rebuild_audit_index(db)}


# ---------------------------------------------------------------------------
# Advanced Inspector endpoint (slow path -- NOT the hot detector pipeline)
# ---------------------------------------------------------------------------

class InspectRequest(BaseModel):
    prompt: str
    response: Optional[str] = None
    context: list[str] = []


@app.post("/v1/inspect", tags=["inspector"])
@limiter.limit(f"{settings.rate_limit_default}/minute")
async def advanced_inspect(
    request: Request,
    payload: InspectRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
):
    """LLM-backed governance inspector for a prompt/response pair.

    This endpoint runs on a SEPARATE slow path with its own latency budget.
    It is NEVER inserted into or blocking the sub-50ms hot-path detector pipeline.

    The LLM only *describes* evidence and *suggests* a recommendation --
    it never enforces policy. All policy enforcement remains in the hot path.
    """
    # Input size guard: prompt + response + all context items
    total_chars = len(payload.prompt) + len(payload.response or "") + sum(len(c) for c in payload.context)
    if total_chars > settings.max_prompt_chars * 3:
        raise HTTPException(status_code=413, detail="Inspect payload exceeds configured maximum size.")
    from backend.app.llm.client import LLMClient, build_evidence_block
    from backend.app.llm.prompts import (
        build_inspector_system_prompt,
        parse_inspection_result,
    )

    # Build context list: user-supplied context + response as evidence if present
    context_items = list(payload.context)
    if payload.response:
        context_items.insert(0, f"Candidate response: {payload.response}")

    api_key = settings.api_key if hasattr(settings, "api_key") else os.getenv("GROQ_API_KEY", "")

    client = LLMClient(
        api_key_getter=lambda: api_key or None,
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        max_completion_tokens=600,
    )

    llm_response = client.generate(
        system_prompt=build_inspector_system_prompt(),
        user_prompt=payload.prompt,
        context=context_items,
    )

    result = parse_inspection_result(
        raw_json=llm_response.text,
        generation_mode=llm_response.generation_mode,
        citation_check=llm_response.citation_check,
    )

    # Automatically queue prompt for RLHF dual-model generation & human review
    if payload.prompt:
        import uuid
        gov_proxy = GovernanceRequest(
            user_id="inspector",
            application_id="advanced-inspector",
            department="GENERAL",
            prompt=payload.prompt,
        )
        background_tasks.add_task(maybe_collect_pair, gov_proxy, payload.response, {})

        insp_risk = 1.0 if result.detected_risk == "high" else (0.5 if result.detected_risk == "medium" else 0.0)
        review_queue.db.create_review(
            request_id=str(uuid.uuid4()),
            policy_id=result.applicable_policy or "llm-inspector",
            reason=result.reason or "LLM Governance Inspection",
            risk=insp_risk,
            prompt=payload.prompt,
        )

    return {
        "applicable_policy": result.applicable_policy,
        "evidence_refs": result.evidence_refs,
        "detected_risk": result.detected_risk,
        "reason": result.reason,
        "required_controls": result.required_controls,
        "recommendation": result.recommendation,
        "generation_mode": result.generation_mode,
        "citation_check": result.citation_check,
        "latency_ms": llm_response.latency_ms,
    }




# ---------------------------------------------------------------------------
# RLHF monitoring endpoint
# ---------------------------------------------------------------------------

@app.get("/v1/rlhf/status", tags=["rlhf"])
async def rlhf_status():
    """Return live RLHF data-collection statistics.

    Includes daily API call counts, total / labelled pair counts broken
    down by category, and an export_ready flag (True when at least one
    labelled pair is available for DPO export).

    This endpoint is read-only and never triggers any model training.
    """
    try:
        from rlhf.config import get_daily_counts, Category, SAMPLING_RATE_N
        from rlhf.storage.json_store import query

        daily = get_daily_counts()

        # Count pairs per category.
        pairs_by_category: dict[str, int] = {}
        total_pairs = 0
        labeled_pairs = 0
        for cat in Category:
            cat_pairs = query(category=cat)
            pairs_by_category[cat.value] = len(cat_pairs)
            total_pairs += len(cat_pairs)
            labeled_pairs += sum(1 for p in cat_pairs if p.chosen is not None)

        return {
            "daily_counts": daily,
            "total_pairs": total_pairs,
            "labeled_pairs": labeled_pairs,
            "pairs_by_category": pairs_by_category,
            "export_ready": labeled_pairs > 0,
            "sampling_rate_n": SAMPLING_RATE_N,
        }

    except Exception as exc:
        logger.warning("RLHF status endpoint error: %s", exc)
        return {"error": str(exc), "total_pairs": 0, "labeled_pairs": 0, "export_ready": False}


# ---------------------------------------------------------------------------
# RLHF DPO export endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/rlhf/export", tags=["rlhf"])
async def rlhf_export_dpo(category: Optional[str] = None):
    """Trigger a DPO-format export and return the file path + record count."""
    try:
        from rlhf.export.dpo_export import export_for_dpo
        from rlhf.config import Category
        cat = None
        if category:
            try:
                cat = Category(category.upper())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown category: {category}")
        path = export_for_dpo(category=cat)
        import pathlib
        lines = pathlib.Path(path).read_text(encoding="utf-8").strip().splitlines()
        return {"status": "ok", "path": path, "records": len(lines), "category": category or "ALL"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("RLHF export error: %s", exc)
        return {"status": "error", "detail": str(exc)}


@app.get("/v1/rlhf/export/latest", tags=["rlhf"])
async def rlhf_get_latest_export():
    """Return the content of the most recent DPO export file."""
    import pathlib
    import json as _json
    exports_dir = pathlib.Path("rlhf/data/exports")
    files = sorted(exports_dir.glob("dpo_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No DPO exports found. Call POST /v1/rlhf/export first.")
    latest = files[0]
    lines = latest.read_text(encoding="utf-8").strip().splitlines()
    records = []
    for line in lines:
        try:
            records.append(_json.loads(line))
        except Exception:
            pass
    return {"file": latest.name, "records": len(records), "data": records[:10], "total_available": len(records)}


# ---------------------------------------------------------------------------
# Audit Integrity endpoint
# ---------------------------------------------------------------------------

@app.get("/v1/audit/integrity", tags=["audit"])
async def audit_integrity_check():
    """Verify the tamper-evident audit chain (Merkle tree + hash chain)."""
    try:
        from backend.app.audit_integrity.backends import (
            AuditRecordBackend,
            AnchorBackend,
        )
        from backend.app.audit_integrity.verifier import verify_ledger
        import pathlib

        hmac_secret = settings.audit_hash_key.encode("utf-8")
        db_path = settings.db_path
        anchor_path = str(pathlib.Path(db_path).with_suffix(".integrity.jsonl"))

        record_backend = AuditRecordBackend(db_path)
        anchor_backend = AnchorBackend(anchor_path)

        result = verify_ledger(record_backend, anchor_backend, hmac_secret)
        return {
            "ok": result.ok,
            "records_checked": result.records_checked,
            "checkpoints_checked": result.checkpoints_checked,
            "first_broken_seq": result.first_broken_seq,
            "first_broken_checkpoint": result.first_broken_checkpoint,
            "details": result.details,
            "status": "TAMPER_FREE" if result.ok else "TAMPERED",
        }
    except Exception as exc:
        logger.warning("Audit integrity check error: %s", exc)
        return {
            "ok": False,
            "records_checked": 0,
            "checkpoints_checked": 0,
            "details": [str(exc)],
            "status": "ERROR",
        }


# ---------------------------------------------------------------------------
# Session accumulator status endpoint
# ---------------------------------------------------------------------------

@app.get("/v1/session/{session_id}", tags=["session"])
async def session_status(session_id: str):
    """Return live session accumulator state for a given session ID."""
    try:
        from backend.risk.session_store import get_session_store
        store = get_session_store()
        state = store.get(session_id)
        if not state:
            return {"session_id": session_id, "found": False, "message": "No session data found"}
        return {
            "session_id": session_id,
            "found": True,
            "ewma_score": round(state.ewma_score, 4),
            "peak_score": round(state.peak_score, 4),
            "session_risk": round(state.session_risk, 4),
            "last_band": state.last_band,
            "turn_count": state.turn_count,
            "contamination_active": state.contamination_active,
            "contaminated_tools": state.contaminated_tools,
            "fast_lane_correction_count": state.fast_lane_correction_count,
        }
    except Exception as exc:
        logger.warning("Session status error: %s", exc)
        return {"session_id": session_id, "found": False, "error": str(exc)}
