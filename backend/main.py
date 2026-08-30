"""ControlPlane.ai — FastAPI Entrypoint (Section 5.1).

This is the single ingress point for every AI interaction. It wires
together all components: gateway, detectors, risk, policy, decision,
audit, and async pipeline.
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

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
from backend.gateway.auth import verify_api_key
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



@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield


app = FastAPI(
    title="ControlPlane.ai",
    description="Enterprise AI governance control plane — observe, reason, act, learn.",
    version="0.4.0",
    lifespan=lifespan,
)
app.include_router(agent_router)

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("controlplane.gateway")

db = Database(settings.db_path)
review_queue = ReviewQueue(db=db)
feedback_evaluator = FeedbackEvaluator()
gpu = GPUAdapter()


@app.post("/admin/reload-models", tags=["admin"])
async def reload_models():
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
@app.post("/v1/govern", response_model=GovernanceResponse)
async def govern(
    request: GovernanceRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
    x_controlplane_session_id: Optional[str] = Header(None),
):
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

    # 6. Human review fallback (Section 5.7)
    decision = review_queue.enqueue(decision)

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

    logger.info(
        "governance_decision request_id=%s action=%s risk=%.3f latency=%.1fms fingerprint=%s",
        request_id,
        decision.action,
        risk.overall_risk,
        latency_ms,
        fingerprint(request.prompt),
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
        async_request = request.model_copy(update={"response": sanitized or candidate_response})
        await publish_event(request_id, async_request, async_job_id)
        
        if fast_lane_pending:
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
async def chat(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
):
    gov_req = GovernanceRequest(
        user_id=payload.user_id,
        user_role=payload.user_role,
        department=payload.department,
        application_id=payload.application_id,
        prompt=payload.prompt,
        data_classification=payload.data_classification,
        session_id=payload.session_id,
    )
    gov_resp = await govern(gov_req, background_tasks=background_tasks, _api_key=_api_key)


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
async def get_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/v1/async/{request_id}")
async def get_async_by_request(request_id: str):
    job_id = f"async-{request_id[:8]}"
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Async job for request not found")
    return job


@app.get("/v1/metrics")
async def metrics():
    return db.metrics()


@app.get("/v1/requests")
async def requests_list(limit: int = 50):
    return db.recent_requests(min(limit, 200))


@app.get("/v1/audits")
async def audits(limit: int = 50):
    return db.recent_audits(min(limit, 200))


@app.get("/v1/audits/{request_id}")
async def audit(request_id: str):
    record = db.get_audit(request_id)
    if not record:
        raise HTTPException(status_code=404, detail="Audit record not found")
    return record


@app.get("/v1/policies", response_model=PolicySummary)
async def policy_summary():
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


@app.get("/v1/reviews")
async def list_pending_reviews(limit: int = 50):
    return review_queue.list_pending(min(limit, 200))


@app.post("/v1/reviews/{request_id}/resolve")
async def resolve_review(request_id: str, payload: ReviewResolution):
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
async def ask_controlplane(payload: AskRequest):
    from rag.ask_controlplane.chat import ask
    return ask(payload.question, db=db)


@app.post("/v1/ask-controlplane/reindex")
async def reindex_audit():
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
async def advanced_inspect(payload: InspectRequest):
    """LLM-backed governance inspector for a prompt/response pair.

    This endpoint runs on a SEPARATE slow path with its own latency budget.
    It is NEVER inserted into or blocking the sub-50ms hot-path detector pipeline.

    The LLM only *describes* evidence and *suggests* a recommendation --
    it never enforces policy. All policy enforcement remains in the hot path.
    """
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
