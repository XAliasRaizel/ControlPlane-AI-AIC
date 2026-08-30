# ControlPlane.ai — Enterprise AI Governance Control Plane

> **Accenture Innovation Challenge 2026 · Track 1**  
> *A Production-Grade, Multi-Layered Governance Gateway for Generative AI & Autonomous Agent Systems.*

---

ControlPlane.ai is a governance middleware layer that sits between AI applications (LLM chatbots, copilots, RAG pipelines, autonomous agents) and underlying foundation models. Every interaction is intercepted, analysed across multiple dimensions in parallel, evaluated against layered policies, and resolved into an enforceable decision — in a single synchronous round-trip under 50ms.

Decisions are recorded in a **cryptographically tamper-evident Merkle audit ledger**, enriched by an **async deep-analytics pipeline**, session risk is tracked across turns by a **dual-signal session accumulator**, and preference data is collected automatically for **RLHF / DPO fine-tuning**.

---

## 📑 Table of Contents

1. [Key Capabilities](#-key-capabilities)
2. [System Architecture](#-system-architecture)
3. [Repository Layout](#-repository-layout)
4. [Quick Start & Setup](#-quick-start--setup)
5. [API Reference](#-api-reference)
6. [Feature Deep Dives](#-feature-deep-dives)
   - [Hot-Path Detector Pipeline](#1-hot-path-detector-pipeline)
   - [Session Risk Accumulator](#2-session-risk-accumulator-phase-9)
   - [Fast-Lane Post-Response Analysis](#3-fast-lane-post-response-analysis)
   - [Policy RAG & Ask ControlPlane](#4-policy-rag--ask-controlplane)
   - [LLM-Powered Advanced Inspector](#5-llm-powered-advanced-inspector)
   - [Agentic Tool-Call Governance](#6-agentic-tool-call-governance)
   - [RLHF / DPO Preference Data Pipeline](#7-rlhf--dpo-preference-data-pipeline)
   - [Tamper-Evident Merkle Audit Ledger](#8-tamper-evident-merkle-audit-ledger)
   - [Two-Layer Hallucination & Grounding Detection](#9-two-layer-hallucination--grounding-detection)
   - [Two-Layer Bias & Fairness Detection](#10-two-layer-bias--fairness-detection)
   - [Unified Sensitive Data Protection](#11-unified-sensitive-data-protection)
7. [Streamlit Dashboard](#-streamlit-dashboard)
8. [Testing & Evaluation Harness](#-testing--evaluation-harness)
9. [Runnable Demonstrations](#-runnable-demonstrations)
10. [Research Grounding & Standards Compliance](#-research-grounding--standards-compliance)
11. [Environment Variables](#-environment-variables)

---

## 🌟 Key Capabilities

| Layer / Feature | Description | Primary Location |
|---|---|---|
| **Hot-Path Gateway** | FastAPI ingress with API key auth, prompt-length enforcement, UUID-stamped request lifecycle, and sub-50ms budget. | `backend/gateway/`, `backend/main.py` |
| **Parallel Hot-Path Detectors** | 7 detectors run concurrently via `asyncio.gather`: PII, Injection, Authorization, Safety, Hallucination, Bias, Sensitive-Query-Intent. | `backend/detectors/` |
| **Session Risk Accumulator** | Dual-signal (EWMA + peak-with-decay) cross-turn risk memory with entity-reconstruction detection, tool-chain contamination tracking, and Redis support. | `backend/risk/accumulator.py`, `backend/risk/session_store.py` |
| **Fast-Lane Post-Response Analysis** | Background async detectors (250ms timeout) that fire after the response is returned. High-risk findings push a webhook RETRACT signal. | `backend/main.py` (`run_fast_lane`) |
| **Noisy-OR Risk Engine** | Bayesian evidence fusion preventing any high-severity signal from being diluted by low-severity signals. | `backend/risk/engine.py` |
| **Hierarchical Policy Engine** | Multi-file YAML rules with Application > Department > Global precedence, hot-reloading, and priority-ordered evaluation. | `backend/policy/` |
| **Five Governance Decisions** | `ALLOW`, `MODIFY` (with automatic PII redaction), `REROUTE`, `HUMAN_REVIEW`, `BLOCK`. | `backend/decision/engine.py` |
| **Policy RAG Explainer** | Bounded (<40ms) semantic retrieval over GDPR, EU AI Act, HIPAA, and internal policies to annotate every decision with regulatory citations. | `rag/policy/policy_rag.py` |
| **Ask ControlPlane** | Compliance Q&A assistant with hybrid retrieval over policy corpora and audit logs, powered by Groq (`openai/gpt-oss-120b`). | `rag/ask_controlplane/` |
| **Advanced Inspector** | Slow-path LLM-backed governance inspector (`POST /v1/inspect`) with indirect prompt-injection hardening. | `backend/app/llm/` |
| **Agentic Tool-Call Governance** | `ToolGovernor` intercepts agent tool calls before execution; scores risk, applies YAML policies, respects session risk carryover. | `backend/agents/` |
| **RLHF / DPO Pipeline** | Automatic 1-in-N preference pair collection per category (HR, Finance, Safety …), LLM-judge labelling, DPO JSONL export, and LoRA training stubs. | `rlhf/` |
| **Tamper-Evident Merkle Ledger** | SHA-256 hash chain + RFC 6962 Merkle tree checkpoints anchored to an external HMAC anchor file. Detects both naive and rechaining attacks. | `backend/audit_integrity/` |
| **Two-Layer Hallucination Detection** | Hot-path claim gate (<0.5ms) + async deep NLI grounding (cross-encoder entailment, LLM-judge, SelfCheckGPT). | `backend/detectors/hallucination.py`, `backend/async_engines/grounding.py` |
| **Two-Layer Bias & Fairness** | Hot-path ECOA/Title-VII causal tripwire + async counterfactual flip-rate + LLM bias rubric. | `backend/detectors/bias.py`, `backend/async_engines/fairness.py` |
| **Human Review Queue** | SQLite-backed review queue for HUMAN_REVIEW decisions; resolve via `POST /v1/reviews/{id}/resolve`. | `backend/review/queue.py` |
| **FPR/FNR Feedback Loop** | Feedback endpoint classifies overrides as false-positive or false-negative and feeds the calibration record. | `backend/feedback/evaluator.py` |

---

## 🏛️ System Architecture

```
AI Applications (chatbots, RAG pipelines, agents, copilots)
        │  prompt / candidate response / proposed tool-call
        ▼
============================== CONTROLPLANE.AI ==============================

  [1] GATEWAY (FastAPI v0.4.0 + API Key Auth)
      auth · prompt-length guard · UUID request_id · timestamp
        │
        ▼
  [2] CONTEXT ENRICHMENT
      user role · department · app criticality · data classification · RBAC
        │
        ▼
  [3] HOT-PATH DETECTORS (asyncio.gather, <50ms)
      ┌─ PII & Sensitive Terms Scanner (6 categories + fail-cautious safety net)
      ├─ Prompt Injection & Jailbreak (instruction overrides, DAN signatures)
      ├─ Authorization & RBAC (role entitlements vs. resource sensitivity)
      ├─ Safety & Harmful Content (violence, hacking, exploit patterns)
      ├─ Hallucination Fast Gate (claim-level verification)
      ├─ Bias Fast Gate (protected-attribute-as-reason causal tripwire)
      └─ Sensitive Query Intent (detail-seeking language heuristic)
        │
        ▼
  [4] RISK ENGINE (Noisy-OR Evidence Fusion)
      P(risk) = 1 - ∏(1 - pᵢ) + context multiplier
      + session_risk injection (EWMA / peak-with-decay if accumulator enabled)
        │
        ▼
  [5] POLICY ENGINE & POLICY RAG
      YAML rules: Application > Department > Global precedence
      └─► Policy RAG retrieves regulatory citations in parallel (~2ms warm)
        │
        ▼
  [6] DECISION ENGINE & SANITIZATION
      ALLOW · MODIFY (regex redaction) · REROUTE · BLOCK · HUMAN_REVIEW ──┐
        │                                                                  │
        │  fire-and-forget background                                      ▼
        │                                                     [7] HUMAN REVIEW QUEUE
        │                                                         approve / reject / modify
        │◄─────────────────────────────────────────────────────────────────┘
        ▼
  Sanitized response returned to application (+ policy_evidence annotation)
        │
        ├── background: RLHF pair sampling (1-in-N)
        ├── background: fast-lane async detectors (250ms, webhook RETRACT)
        └── background: deep async analytics pipeline
              ├─ Grounding RAG & Deep NLI Engine (DeBERTa + SelfCheckGPT)
              ├─ Deep Fairness Engine (Counterfactual probe + LLM judge)
              └─ Performance, Cost, Privacy, Safety, Business engines

  [8] TAMPER-EVIDENT AUDIT LEDGER
      HMAC-hashed audit context · SHA-256 hash chain · Merkle checkpoints

  [9] SESSION ACCUMULATOR (cross-turn, opt-in)
      EWMA score + peak-with-decay → session_risk, session_band (1/2/3)
      PII entity-reconstruction detection across rolling fragment window
      Tool-chain contamination tracking (sticky per session TTL)
      Backends: InMemorySessionStore (default) | Redis (multi-worker)

  [10] RLHF / DPO PIPELINE (background, non-blocking)
       Category-validated preference pairs → LLM judge labelling
       → DPO JSONL export → LoRA fine-tuning (training/ stubs)

===========================================================================

  [11] AGENTIC GOVERNANCE (/agent/act)
       Agent proposes tool call → ToolGovernor intercepts → Risk + Policy
       → ALLOW (executes) | HUMAN_REVIEW (held) | BLOCK (aborts)
```

---

## 📁 Repository Layout

```
controlplane-ai/
├── Dockerfile                           # Production container spec
├── docker-compose.yml                   # Multi-service stack (gateway + UI)
├── requirements.txt                     # Core runtime dependencies
├── start.ps1                            # PowerShell one-shot launcher
│
├── backend/
│   ├── main.py                          # FastAPI v0.4.0 entrypoint (759 lines)
│   ├── shared/
│   │   ├── schemas.py                   # Canonical Pydantic v2 contracts (single source of truth)
│   │   ├── sensitive_terms.py           # Unified taxonomy: financial, IDs, HR, medical, auth
│   │   ├── config.py                    # .env loader + Settings dataclass
│   │   ├── model_backend.py             # Lazy model loader (CONTROLPLANE_MODEL_<TASK>)
│   │   ├── gpu_adapter.py               # Hardware inference interface
│   │   └── llm_simulator.py             # Synthetic LLM response generator (dev/test)
│   ├── gateway/
│   │   ├── auth.py                      # API key authentication dependency
│   │   └── context_enrichment.py        # RBAC and sensitivity resolver
│   ├── detectors/
│   │   ├── base.py                      # BaseDetector ABC + self-registration DETECTOR_REGISTRY
│   │   ├── pii.py                       # Sensitive term & regex value scanner (+ Presidio opt-in)
│   │   ├── injection.py                 # Jailbreak & instruction-override scanner
│   │   ├── authorization.py             # Deterministic RBAC access check
│   │   ├── safety.py                    # Harmful content & exploit scanner
│   │   ├── hallucination.py             # Hot-path ungrounded-claim gate
│   │   ├── bias.py                      # Hot-path protected-attribute causal detector
│   │   ├── sensitive_query_intent.py    # Detail-seeking language heuristic
│   │   └── async_analytics.py           # 7 background analytics engine detectors
│   ├── async_engines/
│   │   ├── grounding.py                 # Deep NLI entailment + LLM-judge + SelfCheckGPT
│   │   └── fairness.py                  # Counterfactual probe + LLM bias rubric
│   ├── app/
│   │   ├── llm/
│   │   │   ├── client.py                # Groq-backed LLM client (injection-hardened evidence blocks)
│   │   │   └── prompts.py               # Inspector system prompt + result parser
│   │   ├── agents/                      # App-layer agent governance helpers
│   │   └── audit_integrity/             # App-layer audit integrity access
│   ├── utils/
│   │   ├── claims.py                    # Lightweight claim decomposition utility
│   │   └── llm_judge.py                 # Provider-agnostic AI-as-judge (OpenAI/Anthropic/Mock)
│   ├── risk/
│   │   ├── engine.py                    # Noisy-OR Bayesian risk fusion & session injection
│   │   ├── accumulator.py               # Dual-signal EWMA+peak accumulator & entity reconstruction
│   │   └── session_store.py             # InMemorySessionStore + Redis SessionStore protocol
│   ├── policy/
│   │   ├── engine.py                    # Multi-scope hierarchical policy evaluator
│   │   └── loader.py                    # Hot-reloading YAML policy loader & validator
│   ├── decision/
│   │   └── engine.py                    # Decision resolution & pattern-based PII redaction
│   ├── review/
│   │   └── queue.py                     # SQLite human review queue
│   ├── async_pipeline/
│   │   ├── publisher.py                 # Fire-and-forget async dispatcher
│   │   ├── worker.py                    # Background job executor
│   │   └── consumers.py                 # Analytics orchestrator
│   ├── agents/
│   │   ├── models.py                    # ToolCallContext, GovernanceDecision dataclasses
│   │   ├── tools.py                     # Tool execution layer & registry
│   │   ├── risk.py                      # Tool-call risk scoring
│   │   ├── policy.py                    # YAML-driven agent policy evaluator
│   │   ├── queue.py                     # Pending action human review queue
│   │   ├── governance.py                # ToolGovernor: intercept → score → decide → execute
│   │   └── router.py                    # FastAPI routes (/agent/act, /agent/pending, ...)
│   ├── audit/
│   │   └── store.py                     # SQLite database & privacy-safe HMAC audit store
│   ├── audit_integrity/
│   │   ├── models.py                    # AuditRecord, Checkpoint, VerificationResult
│   │   ├── hashing.py                   # Canonical JSON + SHA-256 + HMAC utilities
│   │   ├── merkle.py                    # RFC 6962 Merkle tree with inclusion proofs
│   │   ├── backends.py                  # SQLite record store & append-only anchor file
│   │   ├── ledger.py                    # TamperEvidentAuditLedger (append + seal)
│   │   └── verifier.py                  # Independent chain & checkpoint verifier
│   └── feedback/
│       └── evaluator.py                 # FPR/FNR labelled error evaluator
│
├── rag/
│   ├── config.py                        # RAG settings & latency budgets
│   ├── schemas.py                       # Chunk, Query, Document, RetrievalResult
│   ├── embeddings.py                    # Local TF-IDF/LSA + Sentence-Transformers embedder
│   ├── vector_store.py                  # ChromaDB + zero-dependency NumPy/JSON fallback store
│   ├── chunking.py                      # Paragraph-aware text chunking
│   ├── retriever.py                     # Hybrid vector + lexical retriever
│   ├── evaluation.py                    # End-to-end RAG evaluation harness (12 cases)
│   ├── corpus/
│   │   ├── regulatory/                  # GDPR, EU AI Act, HIPAA knowledge bases
│   │   └── internal_kb/                 # Leave, IT security, expense, company policies
│   ├── ingestion/
│   │   ├── ingest.py                    # Corpus ingestion & index builder
│   │   ├── document_loader.py           # Text & Markdown loader
│   │   ├── policy_loader.py             # YAML policy → prose converter
│   │   └── audit_loader.py              # Privacy-safe audit record indexer
│   ├── policy/
│   │   └── policy_rag.py                # Hot-path Policy RAG explainer (~2ms warm)
│   ├── grounding/
│   │   ├── claim_extractor.py           # Sentence-level checkable claim extractor
│   │   ├── entailment.py                # Lexical & number-penalty entailment checker
│   │   └── grounding_checker.py         # Grounding verification orchestrator
│   └── ask_controlplane/
│       ├── retrieval.py                 # Hybrid policy + audit retrieval
│       ├── chat.py                      # Q&A synthesizer with citations
│       └── llm_client.py               # Groq LLM client with graceful fallback
│
├── rlhf/
│   ├── config.py                        # Category enum, storage selector, sampling rate
│   ├── schema.py                        # PreferencePair Pydantic model
│   ├── sampler.py                       # maybe_collect_pair — 1-in-N fire-and-forget hook
│   ├── generators/                      # Dual-model response generation (API + local)
│   ├── judges/                          # LLM-as-Judge (position-bias controlled) + human CLI judge
│   ├── storage/                         # Active: JSONL append-only; ready: SQLite drop-in
│   ├── export/                          # DPO JSONL export with label filtering
│   └── training/                        # DPO fine-tuning + LoRA evaluation (TRL/PEFT stubs)
│
├── ml/
│   ├── calibration.example.json         # Example calibration file for session accumulator
│   ├── common/                          # Shared LoRA utilities
│   ├── grounding/                       # NLI model training & download scripts
│   ├── fairness/                        # Counterfactual fairness model training
│   ├── prompt_injection/                # Injection classifier training pipeline
│   ├── safety/                          # Safety classifier training pipeline
│   └── notebooks/                       # Orchestrated training notebooks
│
├── policies/
│   ├── global.yaml                      # Universal fallthrough governance rules
│   ├── hr.yaml                          # HR-scoped policy (PII & authorization)
│   ├── finance.yaml                     # Finance-scoped policy (loan decisions)
│   ├── support.yaml                     # Support bot policy (injection & redaction)
│   ├── agent_tools.yaml                 # 7 rules governing tool calls (refunds, email, delete)
│   └── hallucination_bias_rules.yaml    # Hallucination and bias threshold rules
│
├── frontend/
│   └── streamlit_app.py                 # Interactive Streamlit UI (7 tabs, 1106 lines)
│
├── scripts/
│   ├── run_golden_path.py               # HR golden-path end-to-end demo
│   ├── run_agent_governance_demo.py     # 8-scenario agent tool-call demo
│   └── run_audit_integrity_demo.py      # 3-act tamper-evident ledger demo
│
└── tests/                               # 19 test modules, 145 tests
    ├── test_golden_path.py
    ├── test_governance.py
    ├── test_gateway_api.py
    ├── test_policy_engine.py
    ├── test_async_service.py
    ├── test_agent_governance.py
    ├── test_audit_integrity.py
    ├── test_hallucination_bias.py
    ├── test_sensitive_data_coverage.py
    ├── test_rag.py
    ├── test_session_accumulator.py
    ├── test_model_backend.py
    ├── test_fast_lane.py
    ├── test_rlhf_integration.py
    ├── test_groq_llm_client.py
    ├── test_llm_client.py
    ├── test_ml_pipeline.py
    ├── test_paraphrase_consistency.py
    └── test_accumulator_calibration_integrity.py
```

---

## 🚀 Quick Start & Setup

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com) for LLM-powered features (Ask ControlPlane, Advanced Inspector)

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```powershell
# Copy the example and fill in your values
copy .env.example .env
```

At minimum, set your Groq API key to enable LLM-powered features:
```
GROQ_API_KEY=your-key-here
GROQ_MODEL=openai/gpt-oss-120b
```

### 3. Build RAG Indices
```powershell
# Ingest regulatory corpora, internal KB, and YAML policies into vector store
python -m rag.ingestion.ingest
```

### 4. Run the Full Test Suite
```powershell
pytest -q
```
Expected: **145 passed, 2 skipped** across 19 test modules.

### 5. Start the Application

**Option A: Local (two terminals)**
```powershell
# Terminal 1 — FastAPI Gateway (Port 8000)
python -m uvicorn backend.main:app --port 8000

# Terminal 2 — Streamlit Dashboard (Port 8501)
python -m streamlit run frontend/streamlit_app.py --server.headless true
```

**Option B: Docker Compose**
```powershell
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| API (Swagger / OpenAPI) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/govern` | Core governance endpoint — full pipeline evaluation |
| `POST` | `/v1/chat` | Chat endpoint — simplified wrapper with human-readable reply |
| `POST` | `/v1/inspect` | Advanced LLM-backed inspector (slow path, never blocks hot path) |
| `GET` | `/health` | Gateway health + registered detectors + policy summary |
| `GET` | `/v1/metrics` | Aggregate request counts and decision distribution |
| `GET` | `/v1/audits` | Recent audit records (up to 200) |
| `GET` | `/v1/audits/{request_id}` | Single audit record |
| `GET` | `/v1/audit/integrity` | Run Merkle + hash-chain tamper verification |
| `GET` | `/v1/policies` | Loaded policy rule summary |
| `GET` | `/v1/reviews` | Pending human review queue |
| `POST` | `/v1/reviews/{id}/resolve` | Resolve a human review decision |
| `POST` | `/v1/feedback` | Submit a feedback override (FPR/FNR classification) |
| `POST` | `/v1/ask-controlplane` | Compliance Q&A with citation-grounded answers |
| `POST` | `/v1/ask-controlplane/reindex` | Rebuild the audit index for Ask ControlPlane |
| `GET` | `/v1/session/{session_id}` | Live session accumulator state |
| `GET` | `/v1/jobs/{job_id}` | Async analytics job status |
| `GET` | `/v1/rlhf/status` | RLHF preference pair collection statistics |
| `POST` | `/v1/rlhf/export` | Trigger DPO JSONL export |
| `GET` | `/v1/rlhf/export/latest` | Retrieve latest export content |
| `POST` | `/admin/reload-models` | Hot-reload all ML model caches |
| `POST` | `/agent/act` | Agent tool-call governance endpoint |
| `GET` | `/agent/pending` | Pending agent actions awaiting human review |

---

## 🔍 Feature Deep Dives

### 1. Hot-Path Detector Pipeline

Seven detectors are loaded via a self-registration `DETECTOR_REGISTRY` and executed concurrently via `asyncio.gather`. Each returns a `DetectorResult` with a `score` (0–1), `label`, `confidence`, and `evidence` list.

| Detector | What it catches |
|---|---|
| `pii` | Regex patterns across 6 data categories: financial (card, CVV, UPI, IFSC), government IDs (PAN, Aadhaar, SSN, Passport, DL), HR records, medical history, account credentials, plus fail-cautious safety net for detail-seeking language |
| `injection` | Instruction-override signatures, DAN jailbreaks, prompt extraction, and role-manipulation patterns |
| `authorization` | RBAC: verifies the user's role against the sensitivity of the data they are accessing |
| `safety` | Violence, hacking, exploit, and harmful content patterns |
| `hallucination` | Extracts checkable claims (dates, numbers, named entities) and cross-references the request's `retrieved_context` (<0.5ms) |
| `bias` | Protected-attribute proximity detector — flags when age, gender, race, religion, or disability are cited as reasons in decision contexts (ECOA / Title VII / EU AI Act) |
| `sensitive_query_intent` | Heuristic for detail-seeking language about named third parties even when explicit keywords are absent |

All seven produce `DetectorResult` objects that feed directly into the Noisy-OR Risk Engine.

---

### 2. Session Risk Accumulator (Phase 9)

Enabled via `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED=true`. Tracks risk across multiple turns in a single session using a **dual-signal** design:

- **EWMA score** (`alpha=0.3` default) — exponentially weighted per-turn average; responsive to sustained patterns.
- **Peak-with-decay** (`peak_decay=0.9` default) — remembers the worst turn seen but lets it slowly fade; a spike persists ~22 turns before decaying to baseline. This prevents session-evasion (behaving well after a bad turn).
- **Session risk** = `max(EWMA, peak)`, classified into Band 1 / 2 / 3.

Additional features:
- **Entity reconstruction**: Maintains a rolling PII fragment window across turns. If concatenating the last N turns' fragments triggers the PII regex that no individual turn triggered, reconstruction is flagged.
- **Tool-chain contamination**: Once sensitive data touches a tool in a session, that tool is marked contaminated for the session TTL.
- **Fast-lane integration**: A fired fast-lane correction adds an additional signal spike of 1.0 on top of the detector-driven update.
- **Calibration**: Parameters loaded from `calibration.json` (path via `CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG`); falls back to safe defaults.
- **Storage backends**: `InMemorySessionStore` (default, single-process) or `RedisSessionStore` (multi-worker, set `CONTROLPLANE_SESSION_STORE=redis://...`).

Session state is exposed via `GET /v1/session/{session_id}` and displayed live in the Streamlit sidebar.

---

### 3. Fast-Lane Post-Response Analysis

After the synchronous governance decision is returned to the caller, any detectors marked `fast_async=True` are run as a background task with a **250ms timeout**. If high-risk results are produced:

- The job record in SQLite is updated with `fast_lane_results`.
- If the request included a `fast_lane_webhook` URL, a `POST` with `action=RETRACT` is sent to that URL, allowing the application to retract an already-delivered response.
- The fast-lane correction count feeds into the session accumulator for the next turn.

---

### 4. Policy RAG & Ask ControlPlane

**Policy RAG** (`rag/policy/policy_rag.py`) runs in parallel with the decision engine:
- Constructs a domain-specific retrieval query from the request context (role, department, matched rule, data classification, action).
- Retrieves matching clauses from GDPR, EU AI Act, HIPAA, and internal YAML policies using TF-IDF/LSA or Sentence-Transformers embeddings with ChromaDB (or a zero-dependency NumPy fallback).
- Annotates the `GovernanceResponse` with a `policy_evidence` field. Never blocks or alters the decision.
- Warm path: ~2ms; cold path (first request): ~40ms budget enforced.

**Ask ControlPlane** (`rag/ask_controlplane/`) is an interactive compliance Q&A assistant:
- Uses hybrid retrieval over both the policy corpus and the SQLite audit database.
- Synthesises answers via the Groq LLM client (`openai/gpt-oss-120b` by default) with citation verification.
- Falls back to extractive mode (lists evidence directly) when no API key is configured or on any LLM failure.
- Evidence is wrapped in explicit injection-hardening delimiters before being passed to the LLM.

Available via: `POST /v1/ask-controlplane` and the **Ask ControlPlane (RAG)** tab in the UI.

---

### 5. LLM-Powered Advanced Inspector

`POST /v1/inspect` runs a **separate slow-path** analysis using the Groq LLM. It:
- Accepts a `prompt`, optional `response`, and optional `context` list.
- Returns: `applicable_policy`, `evidence_refs`, `detected_risk`, `reason`, `required_controls`, `recommendation`, `generation_mode`, `citation_check`, and `latency_ms`.
- Wraps all evidence in injection-hardening delimiters (`<evidence>...</evidence>`) to prevent malicious audit entries from being obeyed as instructions.
- Verifies every `[N]` citation in the answer against the actual evidence count.
- **Never enforces policy** — the LLM describes evidence and suggests a recommendation; all enforcement remains in the hot path.

Available as the **Advanced Inspector** tab in the Streamlit UI.

---

### 6. Agentic Tool-Call Governance

Autonomous agents propose tool calls via `POST /agent/act`. The `ToolGovernor` intercepts the call before execution:

- **Risk scoring** (`backend/agents/risk.py`): Evaluates action sensitivity, parameters (e.g., refund amounts), user roles, and session risk carryover from the main governance pipeline.
- **Policy evaluation** (`policies/agent_tools.yaml`):
  - Refunds < $50 by support agents → `ALLOW`
  - Refunds $50–$500 → `HUMAN_REVIEW`
  - Refunds > $500 → `BLOCK`
  - Deleting records containing PII without admin role → `BLOCK`
  - Elevated session risk from a prior turn escalates otherwise clean tool calls
- **Outcomes**: `ALLOW` (tool executes), `HUMAN_REVIEW` (held in agent pending queue), `BLOCK` (aborted with reason).

---

### 7. RLHF / DPO Preference Data Pipeline

A production-ready data flywheel that collects preference data from live traffic for fine-tuning governance models:

1. **Sampling** (`rlhf/sampler.py`): `maybe_collect_pair()` is called as a background task on every `/v1/chat` request. 1-in-N requests (configurable) trigger dual-model generation.
2. **Generation** (`rlhf/generators/`): Two model responses are generated concurrently (API-vs-API or local-vs-local).
3. **Category validation** (`rlhf/storage/categorize.py`): Every pair is assigned a `Category` (HR, FINANCIAL, SAFETY, FAIRNESS, AGENTIC, GENERAL) at write time, preventing cross-contamination between fine-tuning runs.
4. **Storage** (`rlhf/storage/json_store.py`): Append-only JSONL; a SQLite drop-in is ready and can be activated via `RLHF_STORAGE_BACKEND=sqlite`.
5. **Labelling** (`rlhf/judges/`): LLM-as-Judge with position-bias control (response order is swapped; ties on disagreement) and a human CLI judge.
6. **Export** (`rlhf/export/dpo_export.py`): Filters unlabelled/tie/error/duplicate pairs and writes `{prompt, chosen, rejected}` DPO JSONL files.
7. **Training** (`rlhf/training/`): Per-category `DPORunConfig` with LoRA r/alpha/modules, integrated with TRL's `DPOTrainer` and PEFT.
8. **Evaluation** (`rlhf/training/evaluate.py`): Average reward margin + human-prompt consistency check with position-bias control.

Monitor via: `GET /v1/rlhf/status`, `POST /v1/rlhf/export`, `GET /v1/rlhf/export/latest`, and the **RLHF Monitor** tab in the UI.

---

### 8. Tamper-Evident Merkle Audit Ledger

Every governance decision is recorded with a cryptographic chain of custody:

- **Privacy-preserving hashing**: Audit contexts are HMAC-hashed with `CONTROLPLANE_AUDIT_HASH_KEY` before storage, so the raw prompt never appears in the ledger in plaintext.
- **SHA-256 hash chain**: Each record's hash is computed over `SHA-256(H_{i-1} || canonical_JSON(R_i))`. Any modification breaks the chain.
- **RFC 6962 Merkle tree checkpoints**: Every 10 records, a Merkle tree is computed over the batch. The root is HMAC-signed and written to an independent append-only anchor file (`.integrity.jsonl`).
- **Independent verifier**: Detects both naive (modify record content) and sophisticated (re-compute hash chain) attacks, because a rechained ledger's root will mismatch the externally-signed checkpoint.
- **Integrity endpoint**: `GET /v1/audit/integrity` runs the full verification and returns `TAMPER_FREE` or `TAMPERED` with the first broken sequence number.

---

### 9. Two-Layer Hallucination & Grounding Detection

**Hot-path** (`detectors/hallucination.py`):
- Extracts specific checkable claims (dates, numbers, named entities) from the response.
- Cross-references them against `retrieved_context` in the request in <0.5ms.
- Raises a risk signal if unverifiable claims are detected without supporting context.

**Deep async** (`async_engines/grounding.py`):
- Ensembles **NLI cross-encoder entailment** (`cross-encoder/nli-deberta-v3-base`), **LLM-as-judge**, and **SelfCheckGPT self-consistency resampling**.

**Grounding RAG** (`rag/grounding/`):
- Lexical and number-penalty entailment checker scoring claims against the internal knowledge base corpus.

---

### 10. Two-Layer Bias & Fairness Detection

**Hot-path** (`detectors/bias.py`):
- Fast regex proximity detector catching protected attributes (age, gender, race, religion, disability, nationality) cited as explicit reasons in decision contexts.
- Designed for ECOA, Title VII, and EU AI Act compliance.

**Deep async** (`async_engines/fairness.py`):
- Generates counterfactual name/pronoun/attribute swaps, re-evaluates the prompt, and computes the **counterfactual flip rate**.
- Applies an LLM-judge bias rubric across multiple fairness dimensions.

---

### 11. Unified Sensitive Data Protection

**Single source of truth** (`backend/shared/sensitive_terms.py`): A shared taxonomy imported by both the PII detector and the authorization checker. Covers:

| Category | Examples |
|---|---|
| Financial | Credit card numbers, CVV, UPI IDs, IFSC codes, bank account numbers |
| Government IDs | PAN, Aadhaar, Passport, Driver's License, SSN |
| HR Records | Salary, employment status, performance ratings |
| Medical | Diagnoses, prescriptions, health conditions |
| Account Credentials | Passwords, API keys, tokens |
| Fail-cautious Safety Net | Third-party detail-seeking language ("give me / show me details about [person]") |

The `_VALUE_PATTERNS` dictionary of compiled regex patterns is also used by the session accumulator's entity-reconstruction check, ensuring a single definition is used system-wide.

---

## 🖥️ Streamlit Dashboard

The dashboard at `http://localhost:8501` has seven tabs:

| Tab | Description |
|---|---|
| 💬 **Governance Chatbot** | Interactive chat with real-time governance. Each message shows decision, risk score, session risk band, detector results, policy evidence, and async job status. |
| 🔬 **Advanced Inspector** | Slow-path LLM analysis of any prompt/response pair with applicable policy, evidence refs, and recommendations. |
| 📊 **Platform Metrics** | Live aggregated metrics: total requests, block rate, avg latency, decision distribution. |
| 📜 **Policy Rules** | All loaded YAML policy rules in structured view. |
| 🗂️ **Review Queue** | Pending human review items with resolve controls. |
| 🧠 **Ask ControlPlane (RAG)** | Compliance Q&A backed by policy corpus + audit log retrieval with citations. |
| 🔁 **RLHF Monitor** | Preference pair counts by category, export-ready status, and sampling rate. |

The **sidebar** shows:
- Configurable caller context (user ID, role, department, application, data classification, API key).
- Live **Session Risk Monitor** — polls `GET /v1/session/{session_id}` to display EWMA score, peak score, session risk, band, turn count, and contamination status.

---

## 🧪 Testing & Evaluation Harness

### Full Automated Test Suite (145 Tests, 19 Modules)

```powershell
pytest -q
# Expected: 145 passed, 2 skipped
```

| Test Module | Description |
|---|---|
| `test_golden_path.py` | End-to-end HR salary violation + parallel hot-path + risk fusion |
| `test_sensitive_data_coverage.py` | Credit card, CVV, PAN, Aadhaar, passport, medical, injection rephrasings (19 scenarios) |
| `test_agent_governance.py` | Tool governance: refund tiers, PII deletions, session risk carryover (13 scenarios) |
| `test_audit_integrity.py` | Hash chains, Merkle proofs, naive & sophisticated tamper detection (14 cases) |
| `test_rag.py` | Policy RAG, chunking, vector store fallback, grounding, Ask ControlPlane (32 cases) |
| `test_session_accumulator.py` | Dual-signal math, entity reconstruction, contamination tracking, Redis fallback, config loading (25+ cases) |
| `test_hallucination_bias.py` | Hot-path & async hallucination and counterfactual fairness engines |
| `test_model_backend.py` | Lazy model loader, cache behaviour, grounding RAG integration |
| `test_rlhf_integration.py` | Preference pair collection, category validation, DPO export, LLM judge |
| `test_llm_client.py` | Groq client evidence-block injection hardening, citation verification, extractive fallback |
| `test_groq_llm_client.py` | Ask ControlPlane Groq integration, graceful degradation |
| `test_fast_lane.py` | 250ms timeout fail-open, webhook RETRACT, high-risk correction |
| `test_accumulator_calibration_integrity.py` | Peak-dilution guarantee (calibration artifact or inline Option-B math) |
| `test_ml_pipeline.py` | ML training pipeline stubs & data utilities |
| `test_governance.py` | PII detection, redaction escalation, injection defence, audit privacy |
| `test_policy_engine.py` | Multi-file YAML policy precedence, priority resolution |
| `test_async_service.py` | Async analytics engine workflow and background consumers |
| `test_gateway_api.py` | FastAPI HTTP client integration & redacted audit verification |
| `test_paraphrase_consistency.py` | Paraphrase attack detection consistency |

### RAG Evaluation Harness

```powershell
python -m rag.evaluation
```

Evaluates precision across all three RAG subsystems:
```
Policy RAG (retrieval relevance):         4/4 passed (100%)
Grounding RAG (status accuracy):          4/4 passed (100%)
Ask ControlPlane (answer / refusal):      4/4 passed (100%)
Total RAG Eval Score:                    12/12 passed (100%)
```

---

## 🎬 Runnable Demonstrations

```powershell
# 1. HR Golden Path (salary data access → BLOCK)
python scripts/run_golden_path.py

# 2. Agentic Tool-Call Governance (8 scenarios: refund tiers, email, delete, session risk)
python scripts/run_agent_governance_demo.py

# 3. Tamper-Evident Merkle Audit (3 Acts: clean → naive tamper → rechaining attack)
python scripts/run_audit_integrity_demo.py

# 4. Session Accumulator Demo
python demo_session_persistence.py

# 5. RAG Capability Evaluation
python -m rag.evaluation
```

---

## 📚 Research Grounding & Standards Compliance

| Methodology / Principle | Reference / Standard | System Implementation |
|---|---|---|
| **AI Risk Management Framework** | NIST AI RMF 1.0 (Govern, Map, Measure, Manage) | Multi-layer risk fusion, context enrichment, and audit trail |
| **AI Management System** | ISO/IEC 42001:2023 | Programmatic policy enforcement and human-in-the-loop review queues |
| **EU AI Act Transparency & High-Risk** | Regulation (EU) 2024/1689 (Articles 50 & Annex III) | Policy RAG corpus, hot-path bias tripwire, counterfactual fairness probe |
| **Faithfulness & NLI Grounding** | RAGAS; BEACON (arXiv:2606.07528); Vectara HHEM | Claim decomposition, DeBERTa cross-encoder NLI, SelfCheckGPT resampling |
| **Counterfactual Fairness** | Kusner et al. (NeurIPS 2017); LangFair (CVS Health) | Token perturbation and counterfactual flip-rate measurement |
| **Tamper-Evident Transparency** | RFC 6962 (Certificate Transparency); Sigstore Rekor | SHA-256 hash chains with domain-separated Merkle tree checkpoints |
| **Bayesian Signal Fusion** | Dempster-Shafer Evidence Theory; AWS Fraud Detector | Noisy-OR probability fusion preventing dilution of high-severity risks |
| **RLHF / DPO Alignment** | Rafailov et al. (arXiv:2305.18290); TRL library | Preference pair collection, LLM-as-judge labelling, per-category DPO fine-tuning |
| **Session Risk** | EWMA + peak-with-decay dual-signal design | Cross-turn risk accumulation catching session evasion patterns |

---

## ⚙️ Environment Variables

### Core Backend

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_DB_PATH` | `controlplane.db` | SQLite audit store path |
| `CONTROLPLANE_LOG_LEVEL` | `INFO` | Logging level |
| `CONTROLPLANE_ASYNC_DELAY_MS` | `50` | Artificial async delay (dev/test) |
| `CONTROLPLANE_POLICIES_DIR` | `policies/` | YAML policy files directory |
| `CONTROLPLANE_AUDIT_HASH_KEY` | `local-prototype-not-a-secret` | HMAC key for audit privacy — **change in production** |
| `CONTROLPLANE_MAX_PROMPT_CHARS` | `12000` | Prompt length limit |

### Session Accumulator

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED` | `false` | Enable cross-turn session risk tracking |
| `CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG` | — | Path to `calibration.json` (alpha, peak_decay, thresholds) |
| `CONTROLPLANE_SESSION_STORE` | — | Redis URL for multi-worker deployments (e.g. `redis://localhost:6379`) |

### ML Model Paths (all optional — detectors fall back gracefully)

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_MODEL_GROUNDING` | — | Path to NLI grounding model |
| `CONTROLPLANE_MODEL_FAIRNESS` | — | Path to fairness model |

### RAG

| Variable | Default | Purpose |
|---|---|---|
| `RAG_EMBEDDING_BACKEND` | `tfidf_lsa` | `tfidf_lsa` (zero-dependency) or `sentence_transformers` |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model name |
| `RAG_EMBEDDING_DIMS` | `128` | Embedding dimensionality |
| `RAG_VECTOR_STORE_DIR` | `rag_store/` | ChromaDB / NumPy store location |
| `RAG_CORPUS_DIR` | `rag/corpus/` | Policy corpus directory |
| `RAG_CHUNK_SIZE_CHARS` | `800` | Chunking size |
| `RAG_CHUNK_OVERLAP_CHARS` | `120` | Chunking overlap |
| `RAG_TOP_K` | `5` | Retrieved chunks per query |
| `RAG_POLICY_THRESHOLD` | `0.20` | Minimum similarity score for policy retrieval |
| `RAG_GROUNDING_THRESHOLD` | `0.28` | Minimum similarity score for grounding checks |
| `RAG_HOT_PATH_BUDGET_MS` | `40` | Max latency budget for Policy RAG on hot path |
| `RAG_GENERATION_ENABLED` | `true` | Enable Groq LLM synthesis in Ask ControlPlane |
| `GROQ_API_KEY` | — | Groq API key for Ask ControlPlane + Advanced Inspector |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name |
| `GROQ_MAX_TOKENS` | `1024` | Max tokens per LLM completion |
| `GROQ_TEMPERATURE` | `0.3` | LLM temperature |

### RLHF

| Variable | Default | Purpose |
|---|---|---|
| `RLHF_STORAGE_BACKEND` | `json` | `json` (active) or `sqlite` (ready, flip to activate) |
| `CP_JUDGE_PROVIDER` | `mock` | LLM judge provider: `groq`, `mock`, `openai`, `anthropic` |

### Frontend

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_API_URL` | `http://127.0.0.1:8000` | Backend URL for Streamlit |

---

## 📄 License & Attribution

Built for the **Accenture Innovation Challenge 2026 (Track 1)**.  
All intellectual property and architecture designed for enterprise AI safety, transparency, and governance compliance.
