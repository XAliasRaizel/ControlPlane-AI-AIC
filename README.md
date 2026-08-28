# ControlPlane.ai — Enterprise AI Governance Control Plane

> **Accenture Innovation Challenge 2026 · Track 1**

ControlPlane.ai is a governance control plane that sits between AI applications and the models/tools they call. It intercepts every prompt, response, and agent tool-call, resolves the business context (who, what app, what data, how sensitive), runs risk detectors concurrently, evaluates policies, and decides whether to `ALLOW`, `MODIFY`, `REROUTE`, `HUMAN_REVIEW`, or `BLOCK` the interaction — then records the decision in a tamper-evident audit trail and learns from human corrections over time.

```
OBSERVE -> REASON -> ACT -> LEARN
   ^_________________________|
```

## Key Capabilities

| Capability | What it does | Where it lives |
|---|---|---|
| **PII Detection & Redaction** | Detects emails, phones, Aadhaar, SSNs, API keys; redacts in responses | `detectors/pii.py` |
| **Prompt Injection Detection** | Catches jailbreak, role-override, and system-prompt extraction attempts | `detectors/injection.py` |
| **Authorization & RBAC** | Deterministic access control — salary, bank, medical records by role | `detectors/authorization.py` |
| **Safety & Harmful Content** | Harassment, exploit, deception pattern detection | `detectors/safety.py` |
| **Hallucination Detection** | Hot-path: ungrounded-claim gate; Async: NLI + LLM-judge + SelfCheckGPT | `detectors/hallucination.py`, `async_engines/grounding.py` |
| **Bias & Fairness Detection** | Hot-path: protected-attribute-as-reason gate; Async: counterfactual probes + LLM-judge | `detectors/bias.py`, `async_engines/fairness.py` |
| **Agentic Tool-Call Governance** | Intercepts agent actions (refund, email, delete) before execution — score, review, allow/block | `agents/` |
| **Tamper-Evident Audit Log** | SHA-256 hash chain + Merkle checkpoints anchored to a separate store | `audit_integrity/` |
| **Human-in-the-Loop Review** | Persistent review queue for HUMAN_REVIEW decisions with approve/reject/modify | `review/queue.py` |
| **Feedback & Learning** | Human overrides feed back into FPR/FNR tracking and threshold optimization | `feedback/evaluator.py` |
| **Declarative Policy Engine** | YAML rules with App > Department > Global precedence, hot-reloadable | `policy/` |

## Architecture

```
AI Applications (chatbots, RAG, agents, copilots, internal tools)
        |  prompt / response / tool-call
        v
================================ CONTROLPLANE.AI ================================

  [1] GATEWAY (FastAPI + API Key Auth)
      auth . rate limit . request shaping . assigns request_id
        |
        v
  [2] CONTEXT ENRICHMENT
      resolves user role . app criticality . data classification
        |
        v
  [3] HOT PATH (Synchronous, ~50ms Latency Budget)
      +-- PII detector ----------+
      |-- Injection detector     |  all run concurrently via asyncio.gather(...)
      |-- Authorization check    |  registered via @register plugin decorator
      |-- Safety detector        |
      |-- Hallucination (fast)   |  claim-level ungrounded-fact regex gate
      +-- Bias (fast) -----------+  protected-attribute-as-decision-reason gate
        |
        v
  [4] RISK ENGINE
      noisy-OR signal fusion + context amplification -> overall_risk, confidence
        |
        v
  [5] POLICY ENGINE
      evaluates YAML rules (Application > Department > Global precedence)
        |
        v
  [6] DECISION ENGINE
      ALLOW . MODIFY . REROUTE . BLOCK . HUMAN_REVIEW --+
        |                                                |
        |                                                v
        |                                   [7] HUMAN REVIEW QUEUE
        |                                       approve / reject / modify
        |<-----------------------------------------------+
        v
  response (possibly modified/sanitized) returned to application
        |
        |  fire-and-forget event
        v
  [8] ASYNC PATH (Non-blocking Background Pipeline)
      Safety . Privacy . Fairness . Grounding . Cost . Performance . Business
      + Deep Grounding Engine (NLI entailment + LLM judge + self-consistency)
      + Deep Fairness Engine  (counterfactual attribute swap + LLM judge)
        |
        v
  [9] AUDIT LOG (Privacy-Preserving + Tamper-Evident)
      HMAC fingerprinting . hash-chained records . Merkle checkpoints
      anchored to a separate append-only store
        |
        +------------------------------+
        v                              v
  [10] DATA LAYER                [11] FEEDBACK & LEARNING
      SQLite / PostgreSQL            human overrides -> threshold &
      . event/metrics store          policy optimization

  [12] AGENTIC TOOL-CALL GOVERNANCE
      agent proposes tool call -> ToolGovernor scores risk ->
      ALLOW / HUMAN_REVIEW / BLOCK -> only then execute

===================================================================================
```

## Repository Structure

```
controlplane_ai/
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- README.md
|
|-- backend/
|   |-- main.py                          # FastAPI entrypoint (mounts agent router)
|   |-- shared/
|   |   |-- schemas.py                   # Canonical Pydantic contracts (single source of truth)
|   |   |-- config.py                    # Environment configuration
|   |   +-- gpu_adapter.py              # Optional hardware inference seam
|   |-- gateway/
|   |   |-- auth.py                      # API key validation dependency
|   |   +-- context_enrichment.py       # Resolves role, criticality, data classification
|   |-- detectors/
|   |   |-- base.py                      # BaseDetector ABC + @register decorator + hot-path runner
|   |   |-- pii.py                       # PII scanner (value patterns + request patterns)
|   |   |-- injection.py                 # Prompt injection signature scanner
|   |   |-- authorization.py             # Deterministic RBAC access check
|   |   |-- safety.py                    # Harmful content rules
|   |   |-- hallucination.py             # [NEW] Hot-path ungrounded-claim gate
|   |   |-- bias.py                      # [NEW] Hot-path protected-attribute-as-reason gate
|   |   +-- async_analytics.py          # 7 async-only analysis engine detectors
|   |-- async_engines/                   # [NEW] Deep async analysis engines
|   |   |-- grounding.py                # NLI entailment + LLM judge + SelfCheckGPT
|   |   +-- fairness.py                 # Counterfactual probe + LLM judge bias rubric
|   |-- utils/                           # [NEW] Shared utilities
|   |   |-- claims.py                    # Lightweight claim decomposition (numbers, dates, entities)
|   |   +-- llm_judge.py                # Provider-agnostic LLM-as-judge (OpenAI/Anthropic/Mock)
|   |-- risk/
|   |   +-- engine.py                   # Noisy-OR fusion + context amplification
|   |-- policy/
|   |   |-- engine.py                    # Multi-scope policy evaluator (App > Dept > Global)
|   |   +-- loader.py                   # Dynamic YAML loader & validator with hot reload
|   |-- decision/
|   |   +-- engine.py                   # Decision resolution & response redaction
|   |-- review/
|   |   +-- queue.py                    # Human review queue & phase-1 fallback
|   |-- async_pipeline/
|   |   |-- publisher.py                 # Fire-and-forget async dispatcher
|   |   |-- worker.py                    # Background task executor
|   |   +-- consumers.py               # Async engine orchestrator
|   |-- agents/                          # [NEW] Agentic tool-call governance
|   |   |-- models.py                    # ToolCallContext, GovernanceDecision dataclasses
|   |   |-- tools.py                     # Tool registry & execution layer
|   |   |-- risk.py                      # Tool-call risk scoring
|   |   |-- policy.py                    # YAML-driven agent policy evaluation
|   |   |-- queue.py                     # Pending tool-call review queue
|   |   |-- governance.py               # ToolGovernor: score -> decide -> execute
|   |   +-- router.py                   # FastAPI router (/agent/act, /agent/pending, etc.)
|   |-- audit/
|   |   +-- store.py                    # SQLite database & privacy-safe HMAC audit store
|   |-- audit_integrity/                 # [NEW] Tamper-evident audit layer
|   |   |-- models.py                    # AuditRecord, Checkpoint, VerificationResult
|   |   |-- hashing.py                   # Canonical JSON + SHA-256 + HMAC helpers
|   |   |-- merkle.py                    # RFC 6962 Merkle tree (root, inclusion proofs)
|   |   |-- backends.py                  # SQLite record store + append-only JSONL anchor store
|   |   |-- ledger.py                    # TamperEvidentAuditLedger (append + seal_checkpoint)
|   |   +-- verifier.py                 # Independent read-only chain + checkpoint verifier
|   +-- feedback/
|       +-- evaluator.py                # Labeled error classification (FPR/FNR)
|
|-- policies/
|   |-- global.yaml                      # Universal fallthrough governance rules
|   |-- hr.yaml                          # HR-scoped policy (PII & authorization rules)
|   |-- finance.yaml                     # Finance-scoped policy (loan decision rules)
|   |-- support.yaml                     # Support bot policy (injection prevention & redaction)
|   |-- agent_tools.yaml                 # [NEW] 7 declarative rules for agent tool-call governance
|   +-- hallucination_bias_rules.yaml   # [NEW] Signal-driven hallucination & bias policy rules
|
|-- frontend/
|   +-- streamlit_app.py                # Interactive governance dashboard & audit viewer
|
|-- scripts/
|   |-- run_golden_path.py               # End-to-end governance demo scenario
|   |-- run_agent_governance_demo.py     # [NEW] 8-scenario agent governance demo
|   +-- run_audit_integrity_demo.py     # [NEW] 3-act tamper-detection demo
|
+-- tests/
    |-- test_golden_path.py              # Section 14 golden path scenario verification
    |-- test_governance.py               # Unit and component governance tests
    |-- test_gateway_api.py              # FastAPI HTTP client integration tests
    |-- test_policy_engine.py            # Multi-file policy engine precedence tests
    |-- test_async_service.py            # Async analysis engine pipeline tests
    |-- test_agent_governance.py         # [NEW] 13 tests: refund, delete, email, session risk
    |-- test_audit_integrity.py          # [NEW] 14 tests: chain, tamper, checkpoint, Merkle math
    +-- test_hallucination_bias.py      # [NEW] 6 tests: hot-path detectors + deep async engines
```

## Quick Start

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Full Test Suite (51 tests)
```powershell
pytest -v
```

Expected: **51 passed** — 7 golden path, 5 governance, 1 gateway, 3 policy, 2 async, 13 agent governance, 14 audit integrity, 6 hallucination/bias.

### 3. Run Demo Scenarios
```powershell
# Golden path (HR PII + unauthorized access -> BLOCK)
python scripts/run_golden_path.py

# Agent governance (8 scenarios: refund, email, delete, session risk)
python scripts/run_agent_governance_demo.py

# Tamper-evident audit (3 acts: clean, naive tamper, sophisticated tamper)
python scripts/run_audit_integrity_demo.py
```

### 4. Start the FastAPI Gateway
```powershell
python -m uvicorn backend.main:app --port 8000
```
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) (lists all registered detectors)

### 5. Launch the Streamlit Dashboard
```powershell
python -m streamlit run frontend/streamlit_app.py --server.headless true
```
Open [http://localhost:8501](http://localhost:8501) to interact with the governance UI.

### 6. Docker Compose
```powershell
docker compose up --build
```

## Feature Deep Dives

### Hallucination Detection (Two-Layer)

**Hot Path** (`detectors/hallucination.py`) — runs inline, ~0.2ms, zero network calls:
- Decomposes the AI response into checkable claims (numbers, dates, named entities) using `utils/claims.py`
- Compares each claim against the `retrieved_context` field of the governance request
- Flags claims with numbers/entities not found in the source context
- Falls back to "confident assertion without context" detection when no source documents exist

**Deep Async** (`async_engines/grounding.py`) — runs in background, may call models:
- **NLI entailment**: Cross-encoder model scores each claim as entailed/not-entailed by context (RAGAS Faithfulness style)
- **LLM-as-judge**: Structured grounding rubric catching numeric/causal flips NLI misses
- **SelfCheckGPT self-consistency**: Resamples the same prompt N times; low agreement across resamples signals hallucination (for no-context case)
- Fuses all signals — published benchmarks show ensembles reach ~0.82 AUROC vs ~0.60 for any single method (BEACON, arXiv:2606.07528)

### Bias & Fairness Detection (Two-Layer)

**Hot Path** (`detectors/bias.py`) — runs inline, ~0.1ms:
- Catches the single highest-liability pattern: a protected attribute (age, gender, race, disability, religion, marital/family) cited as an explicit causal reason in a covered decision (loan, hiring, triage)
- Direct compliance tripwire under ECOA, Fair Housing Act, Title VII, EU AI Act

**Deep Async** (`async_engines/fairness.py`) — runs in background:
- **Counterfactual fairness probe** (Kusner et al. 2017, LangFair-style): Swaps protected-attribute tokens (names, pronouns), re-runs the model, measures output divergence. Material divergence = direct bias evidence.
- **LLM-as-judge rubric**: Structured bias audit across categories with quoted evidence

### Agentic Tool-Call Governance

AI agents that take actions (not just generate text) introduce compounding risk. This feature governs *actions*:

```
Agent proposes tool call (e.g. issue_refund, send_email, delete_record)
  -> ToolGovernor.invoke() intercepts
  -> Risk scoring (amount thresholds, role checks, PII detection, session risk)
  -> Policy evaluation (7 declarative YAML rules in agent_tools.yaml)
  -> Decision: ALLOW (execute immediately) / HUMAN_REVIEW (queue) / BLOCK (reject)
  -> Only ALLOW executes the tool; the agent never calls tools directly
```

**Endpoints**: `POST /agent/act`, `GET /agent/pending`, `POST /agent/pending/{id}/resolve`

### Tamper-Evident Audit Log

HMAC fingerprinting answers "who was this about, without storing raw PII." This feature answers "did anyone edit this record after it was written":

- **Hash chain**: Every record's SHA-256 hash depends on the previous record's hash
- **Merkle checkpoints**: Periodically seals a batch of records into an RFC 6962 Merkle tree root, HMAC-signed and written to a separate append-only store
- **Independent verifier**: Read-only chain integrity + checkpoint consistency checks
- Catches both naive tampering (content edit without hash update) and sophisticated tampering (re-chaining the entire database — still caught by the externally anchored checkpoint)

### LLM Judge (Provider-Agnostic)

Both the hallucination and bias engines use `utils/llm_judge.py` for structured AI-as-judge verdicts:

- **Providers**: OpenAI, Anthropic, or offline Mock (set via `CP_JUDGE_PROVIDER` env var)
- **Graceful degradation**: Mock provider returns conservative heuristic verdicts; pipeline never crashes if a provider is unavailable
- **Caching**: Content-hash-keyed memoization for repeated identical requests
- **Honesty guards**: Mock mode explicitly marks results as `degraded=True` and skips self-consistency/counterfactual probes rather than fabricating fake signals

## Golden Path Demo Scenario

An employee asks the HR Copilot:
> *"Give me Rahul's salary and personal phone number."*

### Execution Trace:
- **Gateway**: Authenticated caller (`user=aryan`, `role=employee`, `app=hr-copilot`)
- **Context Enrichment**: `department=HR`, `data_classification=HIGH`, `criticality=high`
- **Hot Path Detectors (Parallel via `asyncio.gather`)**:
  - `pii` -> score `0.85`, label `PII_DETECTED`
  - `authorization` -> score `1.00`, label `DENIED` (caller unauthorized for salary data)
  - `injection` -> score `0.00`, label `CLEAN`
  - `safety` -> score `0.00`, label `CLEAN`
  - `hallucination_fast` -> score `0.00`, label `no_checkable_claims`
  - `bias_fast` -> score `0.00`, label `no_causal_bias_pattern`
  - **Latency**: `< 5 ms` (well within the 50 ms budget)
- **Risk Engine**: Noisy-OR fusion -> `overall_risk=0.46`, `confidence=0.95`
- **Policy Engine**: Matches `hr.yaml` rule `hr-pii-unauthorized`
- **Decision Engine**: `BLOCK`, Reason: `Unauthorized access to PII-classified data`
- **Audit Store**: Privacy-safe `AuditRecord` stored with HMAC prompt fingerprint (no raw prompt/PII saved)
- **Async Path**: Background analytics (fairness, grounding, safety, privacy, cost, performance, business) scheduled non-blockingly

## Test Coverage

| Test File | Tests | What it covers |
|---|---|---|
| `test_golden_path.py` | 7 | End-to-end governance pipeline, individual detectors, risk engine, policy matching |
| `test_governance.py` | 5 | PII detection, authorization, injection blocking, redaction, audit privacy |
| `test_gateway_api.py` | 1 | FastAPI HTTP integration (injection -> BLOCK + redacted audit) |
| `test_policy_engine.py` | 3 | Multi-file precedence, invalid rule rejection, priority ordering |
| `test_async_service.py` | 2 | Async analytics engine pipeline and workflow |
| `test_agent_governance.py` | 13 | Refund (5 scenarios), delete (3), email (2), session risk (2), unknown tool (1) |
| `test_audit_integrity.py` | 14 | Clean ledger (4), naive tamper (2), sophisticated tamper (3), checkpoint signature (2), Merkle math (3) |
| `test_hallucination_bias.py` | 6 | Hot-path hallucination (2), hot-path bias (2), deep grounding engine (1), deep fairness engine (1) |
| **Total** | **51** | |

## Research Grounding

Every detection technique used in this system is grounded in published industry practice:

| Technique | Used by | Implemented in |
|---|---|---|
| Claim-level NLI entailment (premise=context, hypothesis=claim) | Vectara HHEM; RAGAS Faithfulness metric | `async_engines/grounding.py` |
| LLM-as-judge with structured rubric | Patronus Lynx; NeMo Guardrails; Azure AI Content Safety Groundedness | `utils/llm_judge.py` |
| Self-consistency resampling (no-context fallback) | SelfCheckGPT (Manakul et al. 2023); NeMo self-check rail | `async_engines/grounding.py` |
| Counterfactual attribute swap + compare | LangFair CounterfactualGenerator (CVS Health); Kusner et al. 2017 | `async_engines/fairness.py` |
| Protected-attribute-as-decision-reason pattern match | Direct ECOA / Fair Housing Act / Title VII / EU AI Act compliance tripwire | `detectors/bias.py` |
| Multi-signal fusion (not trusting one method alone) | BEACON benchmark (arXiv:2606.07528): single ~0.60 AUROC, ensemble ~0.82 | Both engines |
| Hash-chained records + Merkle checkpoints | Google Certificate Transparency; Sigstore Rekor; Azure SQL Database Ledger | `audit_integrity/` |
| Noisy-OR risk fusion with context amplification | Standard Bayesian sensor fusion; AWS Fraud Detector composite scores | `risk/engine.py` |

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CP_JUDGE_PROVIDER` | `mock` | LLM judge provider: `mock`, `openai`, or `anthropic` |
| `CP_JUDGE_MODEL` | (provider default) | Model name for judge calls (e.g. `gpt-4o-mini`, `claude-haiku-4-5-20251001`) |
| `OPENAI_API_KEY` | — | Required when `CP_JUDGE_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | — | Required when `CP_JUDGE_PROVIDER=anthropic` |
| `CP_API_KEY` | `demo-key-123` | API key for gateway authentication |

## Known Scope Limits

- **Claim extraction** is regex-based, not real NER — catches numbers, dates, and capitalized multi-word names; misses single-word names and implicit claims. Intentional hot-path speed tradeoff.
- **Counterfactual swaps** only cover gender (name/pronoun pairs) currently. Extending to race/ethnicity or age needs stratified name pools before it constitutes a complete fairness audit.
- **NLI model** in `grounding.py` is a general-purpose cross-encoder (`nli-deberta-v3-base`), not the literal Vectara HHEM checkpoint — swapping is a one-line change.
- **Tamper-evident audit** has a bounded detection window: tampering within un-checkpointed records is only caught once the next checkpoint seals (same as Certificate Transparency's maximum merge delay).
- The two hot-path hallucination/bias detectors benefit from `retrieved_context` flowing through the gateway request. Without it, grounding only runs in self-consistency mode.

## License

This project was built for the Accenture Innovation Challenge 2026.
