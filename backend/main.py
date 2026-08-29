"""ControlPlane.ai — FastAPI Entrypoint (Section 5.1).

This is the single ingress point for every AI interaction. It wires
together all components: gateway, detectors, risk, policy, decision,
audit, and async pipeline.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from rag.policy.policy_rag import get_policy_evidence
        get_policy_evidence(
            role="employee",
            app="support-bot",
            dept="HR",
            matched_rule="test",
            data_class="PUBLIC",
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
# Core governance endpoint (Section 5.1)
# ---------------------------------------------------------------------------
@app.post("/v1/govern", response_model=GovernanceResponse)
async def govern(
    request: GovernanceRequest,
    _api_key: str = Depends(verify_api_key),
):
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

    # 9. Async path — fire and forget with DB tracking (Section 5.8)
    async_job_id: str = f"async-{request_id[:8]}"
    try:
        db.create_job(async_job_id, request_id)
        from backend.async_pipeline.publisher import publish_event
        # Include response in async analysis request so engines have full interaction context
        async_request = request.model_copy(update={"response": sanitized or candidate_response})
        await publish_event(request_id, async_request, async_job_id)
    except Exception as exc:
        logger.warning("Async publish failed (non-blocking): %s", exc)

    # 10. Policy RAG explanation (never affects decision)
    policy_evidence = None
    try:
        from rag.policy.policy_rag import get_policy_evidence
        pe = get_policy_evidence(
            role=request.user_role or "user",
            app=request.application_id or "default",
            dept=request.department or "General",
            matched_rule=policy.policy_id if policy else "",
            data_class=request.data_classification or "PUBLIC",
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


class ChatResponse(BaseModel):
    request_id: str
    action: str
    message: str
    governance: GovernanceResponse


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, _api_key: str = Depends(verify_api_key)):
    gov_req = GovernanceRequest(
        user_id=payload.user_id,
        user_role=payload.user_role,
        department=payload.department,
        application_id=payload.application_id,
        prompt=payload.prompt,
        data_classification=payload.data_classification,
    )
    gov_resp = await govern(gov_req, _api_key)

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


