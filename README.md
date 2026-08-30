# ControlPlane.ai — Enterprise AI Governance Control Plane

> **Accenture Innovation Challenge 2026 · Track 1**  
> *A Production-Grade, Multi-Layered Governance Gateway for Generative AI & Autonomous Agent Systems.*

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.4.0-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-191%20passed%2C%204%20skipped-brightgreen?logo=pytest)
![License](https://img.shields.io/badge/License-Proprietary-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20openai%2Fgpt--oss--120b-f55036?logo=openai&logoColor=white)

---

## What is ControlPlane.ai?

**ControlPlane.ai** is a governance middleware layer that sits between AI applications — LLM chatbots, copilots, RAG pipelines, autonomous agents — and the underlying foundation models. Every interaction is intercepted, analysed across multiple dimensions in parallel, evaluated against layered policies, and resolved into an enforceable decision — all in a single synchronous round-trip targeting **sub-50ms latency**.

It is not a model or a prompt. It is an **operating system for AI trust**: a runtime governance plane that applies the same rigour to AI decisions that compliance departments apply to financial controls.

**Decisions are:**
- Recorded in a **cryptographically tamper-evident Merkle audit ledger** (SHA-256 hash chain + RFC 6962 checkpoints)
- Enriched by an **async deep-analytics pipeline** (DeBERTa NLI grounding, counterfactual fairness)
- Tracked across conversation turns by a **dual-signal session risk accumulator** (EWMA + peak-decay)
- Automatically calibrated via a **self-governing threshold auto-tuner** that learns from reviewer overrides
- Collected as **RLHF preference pairs** for continuous DPO fine-tuning of governance models

---

## 📑 Table of Contents

1. [Why ControlPlane.ai? — The Problem](#-why-controlplaneai--the-problem)
2. [Key Capabilities](#-key-capabilities)
3. [System Architecture](#-system-architecture)
4. [Repository Layout](#-repository-layout)
5. [Quick Start & Setup](#-quick-start--setup)
6. [API Reference](#-api-reference)
7. [Feature Deep Dives](#-feature-deep-dives)
   - [Hot-Path Detector Pipeline](#1-hot-path-detector-pipeline)
   - [Session Risk Accumulator](#2-session-risk-accumulator)
   - [Fast-Lane Post-Response Analysis](#3-fast-lane-post-response-analysis)
   - [Self-Governing Threshold Auto-Tuner](#4-self-governing-threshold-auto-tuner)
   - [Human Review Queue & Feedback Loop](#5-human-review-queue--feedback-loop)
   - [Policy RAG & Ask ControlPlane](#6-policy-rag--ask-controlplane)
   - [LLM-Powered Advanced Inspector](#7-llm-powered-advanced-inspector)
   - [Agentic Tool-Call Governance](#8-agentic-tool-call-governance)
   - [RLHF / DPO Preference Data Pipeline](#9-rlhf--dpo-preference-data-pipeline)
   - [Tamper-Evident Merkle Audit Ledger](#10-tamper-evident-merkle-audit-ledger)
   - [Two-Layer Hallucination & Grounding Detection](#11-two-layer-hallucination--grounding-detection)
   - [Two-Layer Bias & Fairness Detection](#12-two-layer-bias--fairness-detection)
   - [Unified Sensitive Data Protection](#13-unified-sensitive-data-protection)
8. [Streamlit Dashboard — 7 Tabs](#-streamlit-dashboard--7-tabs)
9. [Testing & Evaluation Harness](#-testing--evaluation-harness)
10. [Runnable Demonstrations](#-runnable-demonstrations)
11. [Research Grounding & Standards Compliance](#-research-grounding--standards-compliance)
12. [Environment Variables](#-environment-variables)

---

## 🎯 Why ControlPlane.ai? — The Problem

Modern enterprises are deploying generative AI at scale — in HR copilots, finance chatbots, customer support agents, and internal knowledge assistants. Each deployment carries significant risk:

| Risk | Example | What Happens Without Governance |
|---|---|---|
| **PII Leakage** | Finance chatbot returns salary data to a contractor | GDPR breach, regulatory fines |
| **Prompt Injection** | Attacker overrides system prompt via user input | Data exfiltration, policy bypass |
| **Authorization Bypass** | Intern queries restricted executive compensation data | Insider threat, compliance failure |
| **Hallucination** | LLM states incorrect regulatory facts in a patient context | Legal liability, patient harm |
| **Bias** | Loan decision cites applicant's ethnicity as a reason | ECOA/Title VII violation, reputational risk |
| **Session Evasion** | Attacker behaves well for 5 turns, then exfiltrates on turn 6 | Undetected multi-turn attack |
| **Agentic Overreach** | Autonomous agent deletes customer records without authorization | Irreversible damage, audit failure |

**ControlPlane.ai** addresses all of these with a single, unified governance gateway that enforces policy on every AI interaction — before any content reaches users.

---

## 🌟 Key Capabilities

| Layer / Feature | Description | Primary Location |
|---|---|---|
| **Hot-Path Gateway** | FastAPI ingress: API key auth, prompt-length guard, UUID-stamped request lifecycle, sub-50ms budget | `backend/gateway/`, `backend/main.py` |
| **7 Parallel Detectors** | PII, Injection, Authorization, Safety, Hallucination, Bias, Sensitive-Query-Intent run concurrently via `asyncio.gather` | `backend/detectors/` |
| **Noisy-OR Risk Engine** | Bayesian evidence fusion — prevents dilution of high-severity signals by low-severity ones | `backend/risk/engine.py` |
| **Session Risk Accumulator** | Dual-signal cross-turn memory (EWMA + peak-decay) with entity-reconstruction detection, tool contamination tracking, Redis support | `backend/risk/accumulator.py` |
| **Five Governance Decisions** | `ALLOW`, `MODIFY` (regex PII redaction), `REROUTE`, `HUMAN_REVIEW`, `BLOCK` | `backend/decision/engine.py` |
| **Hierarchical Policy Engine** | Multi-file YAML rules with Application > Department > Global precedence, hot-reloading | `backend/policy/` |
| **Self-Governing Auto-Tuner** | Learns from reviewer overrides — automatically raises detector thresholds (NUDGE) or escalates for human redesign (ESCALATE). Every decision is auditable. | `backend/feedback/feedback_engine.py` |
| **Human Review Queue** | SQLite-backed queue for HUMAN_REVIEW decisions; resolver propagates override data to the auto-tuner | `backend/review/queue.py` |
| **Fast-Lane Post-Response** | Background async detectors (250ms timeout) fire after response is returned; high-risk findings push a webhook RETRACT signal | `backend/main.py` |
| **Policy RAG Explainer** | Bounded (<40ms) semantic retrieval over GDPR, EU AI Act, HIPAA, and internal policies — annotates every decision with regulatory citations | `rag/policy/policy_rag.py` |
| **Ask ControlPlane** | Compliance Q&A assistant with hybrid retrieval over policy corpus and audit logs, powered by Groq | `rag/ask_controlplane/` |
| **Advanced Inspector** | Slow-path LLM-backed governance inspector (`POST /v1/inspect`) with injection-hardening | `backend/app/llm/` |
| **Agentic Tool Governance** | `ToolGovernor` intercepts agent tool calls before execution; scores risk, applies YAML policies, respects session risk carryover | `backend/agents/` |
| **RLHF / DPO Pipeline** | Automatic 1-in-N preference pair collection, LLM-judge labelling, DPO JSONL export, LoRA training stubs | `rlhf/` |
| **Merkle Audit Ledger** | SHA-256 hash chain + RFC 6962 Merkle tree checkpoints anchored to an external HMAC anchor file; detects naive and rechaining attacks | `backend/audit_integrity/` |
| **Two-Layer Hallucination** | Hot-path claim gate (<0.5ms) + async deep NLI grounding (DeBERTa + SelfCheckGPT) | `backend/detectors/hallucination.py`, `backend/async_engines/grounding.py` |
| **Two-Layer Bias & Fairness** | Hot-path ECOA/Title-VII causal tripwire + async counterfactual flip-rate + LLM bias rubric | `backend/detectors/bias.py`, `backend/async_engines/fairness.py` |
| **Rich Metrics Dashboard** | Plotly charts: decision distribution, risk histogram, latency trend, risk trend, detector fire rates, blocked-by-policy breakdown | `backend/audit/store.py`, `GET /v1/metrics/rich` |

---

## 🏛️ System Architecture

```
AI Applications (chatbots, RAG pipelines, agents, copilots)
        │  prompt / candidate response / proposed tool-call
        ▼
============================== CONTROLPLANE.AI ==============================

  [1] GATEWAY (FastAPI + API Key Auth)
      auth · prompt-length guard · UUID request_id · timestamp
        │
        ▼
  [2] CONTEXT ENRICHMENT
      user role · department · app criticality · data classification · RBAC
        │
        ▼
  [3] HOT-PATH DETECTORS  (asyncio.gather, <50ms budget)
      ┌─ PII & Sensitive Terms (6 categories + fail-cautious safety net)
      ├─ Prompt Injection & Jailbreak (DAN, instruction overrides)
      ├─ Authorization & RBAC (role entitlements vs. resource sensitivity)
      ├─ Safety & Harmful Content (violence, hacking, exploits)
      ├─ Hallucination Fast Gate (claim-level cross-reference, <0.5ms)
      ├─ Bias Fast Gate (protected-attribute causal tripwire)
      └─ Sensitive Query Intent (detail-seeking language heuristic)
        │
        ▼
  [4] RISK ENGINE  (Noisy-OR Evidence Fusion)
      P(risk) = 1 − ∏(1 − pᵢ) + context multiplier
      + session_risk injection (EWMA / peak-decay if accumulator enabled)
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
        │                                                                  │
        │                                                         Override data feeds
        │                                                                  │
        │                                                                  ▼
        │                                                  [8] SELF-GOVERNING AUTO-TUNER
        │                                                      NUDGE / ESCALATE / HOLD
        │                                                      → YAML threshold patches
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

  [9] TAMPER-EVIDENT AUDIT LEDGER
      HMAC-hashed audit context · SHA-256 hash chain · Merkle checkpoints

  [10] SESSION ACCUMULATOR (cross-turn, opt-in)
       EWMA score + peak-with-decay → session_risk, session_band (1/2/3)
       PII entity-reconstruction detection across rolling fragment window
       Tool-chain contamination tracking (sticky per session TTL)
       Backends: InMemorySessionStore (default) | Redis (multi-worker)

  [11] RLHF / DPO PIPELINE (background, non-blocking)
       Category-validated preference pairs → LLM judge labelling
       → DPO JSONL export → LoRA fine-tuning (training/ stubs)

===========================================================================

  [12] AGENTIC GOVERNANCE (/agent/act)
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
│   ├── main.py                          # FastAPI entrypoint — all routes, lifespan hooks
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
│   │   └── llm/
│   │       ├── client.py                # Groq-backed LLM client (injection-hardened evidence)
│   │       └── prompts.py               # Inspector system prompt + result parser
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
│   │   └── queue.py                     # SQLite human review queue (enqueue / resolve)
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
│   │   └── router.py                    # FastAPI routes (/agent/act, /agent/pending)
│   ├── audit/
│   │   └── store.py                     # SQLite audit DB, privacy-safe HMAC store, richer_metrics()
│   ├── audit_integrity/
│   │   ├── models.py                    # AuditRecord, Checkpoint, VerificationResult
│   │   ├── hashing.py                   # Canonical JSON + SHA-256 + HMAC utilities
│   │   ├── merkle.py                    # RFC 6962 Merkle tree with inclusion proofs
│   │   ├── backends.py                  # SQLite record store & append-only anchor file
│   │   ├── ledger.py                    # TamperEvidentAuditLedger (append + seal)
│   │   └── verifier.py                  # Independent chain & checkpoint verifier
│   └── feedback/
│       ├── evaluator.py                 # FPR/FNR labelled error evaluator — classifies overrides
│       └── feedback_engine.py           # Self-Governing Threshold Auto-Tuner (NUDGE/ESCALATE/HOLD)
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
│   ├── judges/                          # LLM-as-Judge (position-bias controlled) + human CLI
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
│   ├── finance.yaml                     # Finance-scoped policy (loan decisions, PII redaction)
│   ├── support.yaml                     # Support bot policy (injection & redaction)
│   ├── agent_tools.yaml                 # 7 rules governing tool calls (refunds, email, delete)
│   └── hallucination_bias_rules.yaml    # Hallucination and bias threshold rules
│
├── frontend/
│   └── streamlit_app.py                 # Interactive Streamlit UI — 7 tabs, ~1650 lines
│
├── scripts/
│   ├── run_golden_path.py               # HR golden-path end-to-end demo
│   ├── run_agent_governance_demo.py     # 8-scenario agent tool-call demo
│   └── run_audit_integrity_demo.py      # 3-act tamper-evident ledger demo
│
└── tests/                               # 19 test modules, 191 tests
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
- A [Groq API key](https://console.groq.com) for LLM-powered features (Ask ControlPlane, Advanced Inspector, RLHF judging)

### 1. Clone & Install

```powershell
git clone https://github.com/XAliasRaizel/ControlPlane-AI-AIC.git
cd ControlPlane-AI-AIC

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```powershell
# Copy the example file and fill in your values
copy .env.example .env
```

At minimum, set your Groq API key:
```env
GROQ_API_KEY=your-groq-key-here
GROQ_MODEL=openai/gpt-oss-120b
CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED=true
```

### 3. Build RAG Indices

```powershell
# Ingest regulatory corpora, internal KB, and YAML policies into the vector store
python -m rag.ingestion.ingest
```

### 4. Run the Full Test Suite

```powershell
pytest -q
# Expected: 191 passed, 4 skipped across 19 test modules
```

### 5. Start the Application

**Option A: Local (PowerShell — two terminals)**
```powershell
# Terminal 1 — FastAPI Backend (Port 8000)
$env:GROQ_API_KEY="your-key"
$env:GROQ_MODEL="openai/gpt-oss-120b"
$env:CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED="true"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Streamlit Dashboard (Port 8501)
python -m streamlit run frontend/streamlit_app.py --server.port 8501 --server.headless true
```

**Option B: Docker Compose**
```powershell
docker compose up --build
```

| Service | URL |
|---|---|
| **Streamlit Dashboard** | http://localhost:8501 |
| **API (Swagger / OpenAPI)** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |

---

## 📡 API Reference

### Core Governance

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/govern` | **Core governance endpoint** — full hot-path pipeline evaluation |
| `POST` | `/v1/chat` | Chat endpoint — human-readable wrapper around `/v1/govern` |
| `POST` | `/v1/inspect` | Advanced LLM-backed inspector (slow path, never blocks hot path) |
| `GET` | `/health` | Gateway health, registered detectors, and policy summary |

### Metrics & Audit

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/metrics` | Aggregate request counts and decision distribution |
| `GET` | `/v1/metrics/rich` | Extended metrics with chart data: risk distribution, latency trend, risk trend, detector fire rates, blocked-by-policy breakdown |
| `GET` | `/v1/audits` | Recent audit records (up to 200) |
| `GET` | `/v1/audits/{request_id}` | Single audit record by request ID |
| `GET` | `/v1/audit/integrity` | Run full Merkle + hash-chain tamper verification |

### Policy & Reviews

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/policies` | Loaded policy rules summary |
| `GET` | `/v1/reviews` | Pending human review queue |
| `POST` | `/v1/reviews/{id}/resolve` | Resolve a human review — action propagates to auto-tuner |
| `POST` | `/v1/feedback` | Submit a reviewer feedback override (FPR/FNR classification) |

### Self-Governing Auto-Tuner

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/feedback/tuning` | **Dry-run** — preview NUDGE / ESCALATE / HOLD decisions without writing any files |
| `POST` | `/v1/feedback/tuning/apply` | **Apply** — write NUDGE threshold changes to policy YAML files |
| `POST` | `/v1/feedback/tuning/seed-demo` | Seed 25 realistic review records to demonstrate all three decision patterns |
| `GET` | `/v1/feedback/tuning/history` | Audit trail of all threshold changes applied to YAML files |

### Ask ControlPlane (RAG)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/ask-controlplane` | Compliance Q&A with citation-grounded answers |
| `POST` | `/v1/ask-controlplane/reindex` | Rebuild the audit index for Ask ControlPlane |

### Session & Jobs

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/session/{session_id}` | Live session accumulator state (EWMA, peak, band, contamination) |
| `GET` | `/v1/jobs/{job_id}` | Async analytics job status |

### RLHF

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/rlhf/status` | Preference pair collection statistics by category |
| `POST` | `/v1/rlhf/export` | Trigger DPO JSONL export |
| `GET` | `/v1/rlhf/export/latest` | Retrieve latest export content |

### Agent Governance

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/agent/act` | Agent tool-call governance endpoint |
| `GET` | `/agent/pending` | Pending agent actions awaiting human review |
| `POST` | `/admin/reload-models` | Hot-reload all ML model caches |

---

## 🔍 Feature Deep Dives

### 1. Hot-Path Detector Pipeline

Seven detectors are loaded via a self-registration `DETECTOR_REGISTRY` and executed concurrently via `asyncio.gather`. Each returns a `DetectorResult` with a `score` (0–1), `label`, `confidence`, and `evidence` list.

| Detector | What it catches |
|---|---|
| `pii` | 6 data categories: financial (card, CVV, UPI, IFSC), government IDs (PAN, Aadhaar, SSN, Passport, DL), HR records, medical history, credentials, plus a fail-cautious safety net for detail-seeking language |
| `injection` | Instruction-override signatures, DAN jailbreaks, prompt extraction, and role-manipulation patterns |
| `authorization` | RBAC: verifies the requesting user's role against the sensitivity of the data they are accessing |
| `safety` | Violence, hacking, exploit, and harmful content patterns |
| `hallucination` | Extracts checkable claims (dates, numbers, named entities) and cross-references the request's `retrieved_context` in <0.5ms |
| `bias` | Protected-attribute proximity detector — flags age, gender, race, religion, or disability cited as reasons in decision contexts (ECOA / Title VII / EU AI Act) |
| `sensitive_query_intent` | Heuristic for detail-seeking language about named third parties even when explicit PII keywords are absent |

All seven produce `DetectorResult` objects that feed directly into the Noisy-OR Risk Engine.

---

### 2. Session Risk Accumulator

Enabled via `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED=true`. Tracks risk across multiple conversation turns using a **dual-signal** design:

- **EWMA score** (`alpha=0.01` calibrated) — exponentially weighted per-turn average; responsive to sustained patterns.
- **Peak-with-decay** (`peak_decay=0.99` calibrated) — remembers the worst turn seen but lets it slowly fade; prevents session-evasion (behaving well after a bad turn).
- **Session risk** = `max(EWMA, peak)`, classified into Band 1 / 2 / 3.

Additional capabilities:
- **Entity reconstruction**: Maintains a rolling PII fragment window across turns. If concatenating fragments from the last N turns triggers the PII regex that no individual turn triggered, the reconstruction attack is flagged.
- **Tool-chain contamination**: Once sensitive data touches a tool in a session, that tool is marked contaminated for the session TTL.
- **Fast-lane integration**: A fired fast-lane correction adds a risk spike of 1.0 on the next accumulator update.
- **Calibration**: Parameters loaded from `calibration.json`; falls back to safe defaults.
- **Storage backends**: `InMemorySessionStore` (default) or `RedisSessionStore` (multi-worker).

Session state exposed via `GET /v1/session/{session_id}` and displayed live in the Streamlit sidebar.

---

### 3. Fast-Lane Post-Response Analysis

After the synchronous governance decision is returned, any detectors marked `fast_async=True` run as a background task with a **250ms timeout**. On high-risk results:

- The SQLite job record is updated with `fast_lane_results`.
- If the request included a `fast_lane_webhook` URL, a `POST` with `action=RETRACT` is sent — allowing the application to retract an already-delivered response.
- The fast-lane correction count feeds into the session accumulator for the next turn.

---

### 4. Self-Governing Threshold Auto-Tuner

**Section 5.12** of the architecture — ControlPlane applies the same governance principle to **itself**.

When human reviewers repeatedly override a rule's decisions (e.g., approving requests that were flagged as BLOCK), it signals the rule is firing too aggressively. The auto-tuner reads those resolved reviews and computes an override rate per rule:

```
Review Queue Resolutions ──▶ Override Rate per Rule ──▶ Tuning Decision
```

| Condition | Decision | Effect |
|---|---|---|
| < 5 resolved reviews | **⏳ INSUFFICIENT DATA** | No change — too few signals |
| Override rate < 25% | **✅ HOLD** | Rule is performing correctly |
| 25% ≤ rate < 50% | **🔼 NUDGE** | Detector threshold raised by +0.05 in YAML |
| Override rate ≥ 50% | **🚨 ESCALATE** | Stop nudging — rule definition needs human redesign |

**Structural safety guarantee:** This mechanism can only push thresholds **up** (require more detector evidence). It structurally cannot lower thresholds or make the system less safe.

**Audit trail:** Every applied NUDGE is logged to the `tuning_history` table with timestamp, rule ID, old/new threshold, override rate, and reasoning.

**API:**
```
GET  /v1/feedback/tuning           → Dry-run preview
POST /v1/feedback/tuning/apply     → Write YAML changes
POST /v1/feedback/tuning/seed-demo → Seed realistic demo data
GET  /v1/feedback/tuning/history   → Full audit changelog
```

---

### 5. Human Review Queue & Feedback Loop

When the policy engine issues a `HUMAN_REVIEW` decision, the request is **held** in the review queue — nothing is silently auto-blocked or auto-approved.

Reviewers resolve via `POST /v1/reviews/{id}/resolve` with:
- `final_action` — the actual disposition (`ALLOW`, `BLOCK`, `MODIFY`, `REROUTE`)
- `reviewer_id` — audit identity
- `notes` — free-text rationale

**The feedback loop closes:** If a reviewer resolves `HUMAN_REVIEW` to `ALLOW` (different from the original implied `BLOCK`), that counts as an **override** — a false-positive signal. These accumulate per-rule and feed the Self-Governing Auto-Tuner, causing that rule's detector threshold to be automatically raised.

This is the only way the system learns from its own mistakes without requiring model retraining.

---

### 6. Policy RAG & Ask ControlPlane

**Policy RAG** (`rag/policy/policy_rag.py`) runs in parallel with the decision engine:
- Constructs a domain-specific retrieval query from request context (role, department, matched rule, data classification, action).
- Retrieves matching clauses from GDPR, EU AI Act, HIPAA, and internal YAML policies using TF-IDF/LSA or Sentence-Transformers + ChromaDB (or a zero-dependency NumPy fallback).
- Annotates the `GovernanceResponse` with a `policy_evidence` field. **Never blocks or alters the decision.**
- Warm path: ~2ms; cold path (first request): <40ms budget enforced.

**Ask ControlPlane** (`rag/ask_controlplane/`) is an interactive compliance Q&A assistant:
- Hybrid retrieval over both the policy corpus and the SQLite audit database.
- Synthesises answers via Groq (`openai/gpt-oss-120b`) with citation verification.
- Falls back to extractive mode (lists evidence directly) when no API key is configured or on any LLM failure.
- All evidence is wrapped in explicit injection-hardening delimiters before being passed to the LLM.

**Available at:** `POST /v1/ask-controlplane` and the **🧠 Ask ControlPlane (RAG)** tab in the UI.

---

### 7. LLM-Powered Advanced Inspector

`POST /v1/inspect` runs a **separate slow-path** LLM analysis using Groq. It:
- Accepts a `prompt`, optional `response`, and optional `context` list.
- Returns: `applicable_policy`, `evidence_refs`, `detected_risk`, `reason`, `required_controls`, `recommendation`, `generation_mode`, `citation_check`, and `latency_ms`.
- Wraps all evidence in injection-hardening delimiters (`<evidence>...</evidence>`).
- Verifies every `[N]` citation in the answer against the actual evidence count.
- **Never enforces policy** — the LLM describes evidence and recommends; all enforcement stays in the hot path.

**Available as:** the **🔬 Advanced Inspector** tab in the Streamlit UI.

---

### 8. Agentic Tool-Call Governance

Autonomous agents propose tool calls via `POST /agent/act`. The `ToolGovernor` intercepts before execution:

- **Risk scoring** (`backend/agents/risk.py`): Evaluates action sensitivity, parameters (e.g., refund amounts), user roles, and session risk carryover from the main governance pipeline.
- **Policy evaluation** (`policies/agent_tools.yaml`):
  - Refunds < $50 by support agents → `ALLOW`
  - Refunds $50–$500 → `HUMAN_REVIEW`
  - Refunds > $500 → `BLOCK`
  - Deleting records containing PII without admin role → `BLOCK`
  - Elevated session risk from a prior turn escalates otherwise clean tool calls
- **Outcomes**: `ALLOW` (tool executes), `HUMAN_REVIEW` (held in agent pending queue), `BLOCK` (aborted with reason).

---

### 9. RLHF / DPO Preference Data Pipeline

A production-ready data flywheel that collects preference data from live traffic for fine-tuning governance models:

1. **Sampling** (`rlhf/sampler.py`): `maybe_collect_pair()` is called as a background task on every `/v1/chat` request. 1-in-N requests trigger dual-model generation.
2. **Generation** (`rlhf/generators/`): Two model responses are generated concurrently (API-vs-API or local-vs-local).
3. **Category validation** (`rlhf/storage/categorize.py`): Every pair is assigned a `Category` (HR, FINANCIAL, SAFETY, FAIRNESS, AGENTIC, GENERAL), preventing cross-contamination between fine-tuning runs.
4. **Storage** (`rlhf/storage/json_store.py`): Append-only JSONL; SQLite drop-in ready via `RLHF_STORAGE_BACKEND=sqlite`.
5. **Labelling** (`rlhf/judges/`): LLM-as-Judge with position-bias control (response order is swapped; ties on disagreement).
6. **Export** (`rlhf/export/dpo_export.py`): Filters unlabelled/tie/error/duplicate pairs and writes `{prompt, chosen, rejected}` DPO JSONL files.
7. **Training** (`rlhf/training/`): Per-category `DPORunConfig` with LoRA r/alpha/modules, integrated with TRL's `DPOTrainer` and PEFT.
8. **Evaluation** (`rlhf/training/evaluate.py`): Average reward margin + human-prompt consistency check with position-bias control.

**Monitored via:** `GET /v1/rlhf/status`, `POST /v1/rlhf/export`, and the **🔁 RLHF Monitor** tab in the UI.

---

### 10. Tamper-Evident Merkle Audit Ledger

Every governance decision is recorded with a cryptographic chain of custody:

- **Privacy-preserving hashing**: Audit contexts are HMAC-hashed with `CONTROLPLANE_AUDIT_HASH_KEY` — the raw prompt never appears in the ledger in plaintext.
- **SHA-256 hash chain**: Each record's hash is computed over `SHA-256(H_{i-1} || canonical_JSON(R_i))`. Any modification breaks the chain.
- **RFC 6962 Merkle tree checkpoints**: Every 10 records, a Merkle tree is computed over the batch. The root is HMAC-signed and written to an independent append-only anchor file (`.integrity.jsonl`).
- **Independent verifier**: Detects both naive (modify record content) and sophisticated (re-compute hash chain) attacks — a rechained ledger's root will mismatch the externally-signed checkpoint.
- **Integrity endpoint**: `GET /v1/audit/integrity` returns `TAMPER_FREE` or `TAMPERED` with the first broken sequence number.

---

### 11. Two-Layer Hallucination & Grounding Detection

**Hot-path** (`detectors/hallucination.py`):
- Extracts specific checkable claims (dates, numbers, named entities) from the response.
- Cross-references them against `retrieved_context` in the request in <0.5ms.
- Raises a risk signal if unverifiable claims are detected without supporting context.

**Deep async** (`async_engines/grounding.py`):
- Ensembles **NLI cross-encoder entailment** (`cross-encoder/nli-deberta-v3-base`), **LLM-as-judge**, and **SelfCheckGPT self-consistency resampling**.

**Grounding RAG** (`rag/grounding/`):
- Lexical and number-penalty entailment checker scoring claims against the internal knowledge base corpus.

---

### 12. Two-Layer Bias & Fairness Detection

**Hot-path** (`detectors/bias.py`):
- Fast regex proximity detector catching protected attributes (age, gender, race, religion, disability, nationality) cited as explicit reasons in decision contexts.
- Designed for ECOA, Title VII, and EU AI Act compliance.

**Deep async** (`async_engines/fairness.py`):
- Generates counterfactual name/pronoun/attribute swaps, re-evaluates the prompt, and computes the **counterfactual flip rate**.
- Applies an LLM-judge bias rubric across multiple fairness dimensions.

---

### 13. Unified Sensitive Data Protection

**Single source of truth** (`backend/shared/sensitive_terms.py`): A shared taxonomy imported by both the PII detector and the authorization checker.

| Category | Examples |
|---|---|
| Financial | Credit card numbers, CVV, UPI IDs, IFSC codes, bank account numbers |
| Government IDs | PAN, Aadhaar, Passport, Driver's License, SSN |
| HR Records | Salary, employment status, performance ratings |
| Medical | Diagnoses, prescriptions, health conditions |
| Account Credentials | Passwords, API keys, tokens |
| Fail-cautious Safety Net | Third-party detail-seeking language ("give me / show me details about [person]") |

The `_VALUE_PATTERNS` dictionary of compiled regex patterns is also used by the session accumulator's entity-reconstruction check — a single definition is used system-wide.

---

## 🖥️ Streamlit Dashboard — 7 Tabs

The dashboard at **http://localhost:8501** provides a comprehensive operations interface for the governance platform.

### Sidebar
- **🟢 Backend Connected** / **🔴 Backend Offline** status pill (15-second cached health check).
- Configurable caller context: User ID, Role, Department, Application, Data Classification, API Key.
- **Live Session Risk Monitor** — polls `GET /v1/session/{session_id}` every 10 seconds to display EWMA score, peak score, session risk, band, turn count, and contamination status.

### Tab 1 — 💬 Governance Chatbot
Interactive chat with real-time governance. Before first message: a welcome hero card explains the 7-stage pipeline. Each response shows:
- Decision badge (ALLOW / MODIFY / BLOCK / HUMAN_REVIEW / REROUTE)
- Risk score and session risk band
- Per-detector scores sorted by risk descending
- Policy evidence (regulatory citations)
- Async job status

### Tab 2 — 🔬 Advanced Inspector
Slow-path LLM analysis of any prompt/response pair. Returns applicable policy, evidence references, detected risk level, required controls, and recommendation. Never enforces — the hot path enforces.

### Tab 3 — 📊 Platform Metrics (Full Analytics Dashboard)
Powered by the new `/v1/metrics/rich` endpoint with interactive Plotly charts:
- **KPI row**: Total Requests, Blocked, Modified, Human Review, Allowed, Avg Latency
- **Decision Distribution** — colour-coded donut chart
- **Risk Distribution** — histogram (Low / Medium / High buckets)
- **Latency Trend** — line chart of last 20 requests
- **Risk Score Trend** — line chart of last 20 requests
- **Detector Fire Rate** table — per-detector fires, totals, and rate
- **Blocked by Policy Rule** table — which rules are most active

### Tab 4 — 📜 Policy Rules
Loaded YAML policy rules in a structured interactive view:
- Action count summary badges at the top
- Per-rule expanders showing: action badge, target detector/condition, threshold, rationale, scope
- Raw condition JSON rendered inline (no nested expanders)
- Handles all condition formats: detector scores, risk thresholds, data classification, application scoping, signal/value operators

### Tab 5 — ⚖️ Review & Auto-Tuning
The most interactive tab — two sections:

**Self-Governing Threshold Auto-Tuner:**
- ASCII flow diagram explaining how reviewer overrides connect to YAML patches
- **🔍 Run Tuning Analysis (Dry Run)** — preview NUDGE / ESCALATE / HOLD decisions
- **⚡ Apply Decisions** — write threshold changes to policy YAML files
- **🧪 Seed Demo Review History** — populate 25 realistic review records showing all three patterns (results card stays visible permanently — no flash)
- Decisions table sorted: ESCALATE → NUDGE → HOLD → INSUFFICIENT DATA
- Reasoning cards for each actionable rule
- Tuning audit changelog from `tuning_history` DB table

**Human Review Queue:**
- All pending `HUMAN_REVIEW` items with risk score colour coding
- Per-item: request ID, policy rule, risk score, reason, timestamp
- Resolve controls: Reviewer ID, Action selector (BLOCK/ALLOW/MODIFY/REROUTE), Notes
- Resolving with a different action than the original automatically feeds the override into the auto-tuner

### Tab 6 — 🧠 Ask ControlPlane (RAG)
Compliance Q&A assistant with 4 quick-launch example chips:
- 📋 PII policy for HR
- ⚖️ GDPR Article 22
- 🛑 Recent BLOCK events
- 🏥 HIPAA data rules

Admin reindex function in a collapsed expander.

### Tab 7 — 🔁 RLHF Monitor
- **Stats banner**: Total Pairs, Labeled, Unlabeled, Label Coverage %, Sampling Rate
- **Label coverage progress bar**
- **⚡ Generate & Judge Preference Pair On-Demand**: Enter a prompt, select domain category, generate dual-model responses and LLM judgment instantly
- **Pairs Explorer**: Searchable/filterable table of all collected pairs (by prompt keyword, by category)
- Export-ready status indicator per pair

---

## 🧪 Testing & Evaluation Harness

### Full Automated Test Suite

```powershell
pytest -q
# Expected: 191 passed, 4 skipped across 19 test modules
```

| Test Module | What is Tested | Key Scenarios |
|---|---|---|
| `test_golden_path.py` | End-to-end HR salary violation + parallel hot-path + risk fusion | Full pipeline integration |
| `test_sensitive_data_coverage.py` | Credit card, CVV, PAN, Aadhaar, passport, medical, injection rephrasings | 19 detection scenarios |
| `test_agent_governance.py` | Tool governance: refund tiers, PII deletions, session risk carryover | 13 agentic scenarios |
| `test_audit_integrity.py` | Hash chains, Merkle proofs, naive & sophisticated tamper detection | 14 tamper cases |
| `test_rag.py` | Policy RAG, chunking, vector store fallback, grounding, Ask ControlPlane | 32 RAG cases |
| `test_session_accumulator.py` | Dual-signal math, entity reconstruction, contamination, Redis fallback, config loading | 25+ accumulator cases |
| `test_hallucination_bias.py` | Hot-path & async hallucination and counterfactual fairness engines | Deep analysis tests |
| `test_model_backend.py` | Lazy model loader, cache behaviour, grounding RAG integration | Model loading tests |
| `test_rlhf_integration.py` | Preference pair collection, category validation, DPO export, LLM judge | Full RLHF pipeline |
| `test_llm_client.py` | Groq client evidence-block injection hardening, citation verification, extractive fallback | LLM safety tests |
| `test_groq_llm_client.py` | Ask ControlPlane Groq integration, graceful degradation | Integration tests |
| `test_fast_lane.py` | 250ms timeout fail-open, webhook RETRACT, high-risk correction | Fast-lane tests |
| `test_accumulator_calibration_integrity.py` | Peak-dilution guarantee (calibration artifact or inline Option-B math) | Math correctness |
| `test_ml_pipeline.py` | ML training pipeline stubs & data utilities | ML pipeline tests |
| `test_governance.py` | PII detection, redaction escalation, injection defence, audit privacy | Core governance |
| `test_policy_engine.py` | Multi-file YAML policy precedence, priority resolution | Policy tests |
| `test_async_service.py` | Async analytics engine workflow and background consumers | Async pipeline |
| `test_gateway_api.py` | FastAPI HTTP client integration & redacted audit verification | API integration |
| `test_paraphrase_consistency.py` | Paraphrase attack detection consistency | Robustness tests |

### RAG Evaluation Harness

```powershell
python -m rag.evaluation
```

```
Policy RAG (retrieval relevance):         4/4 passed (100%)
Grounding RAG (status accuracy):          4/4 passed (100%)
Ask ControlPlane (answer / refusal):      4/4 passed (100%)
Total RAG Eval Score:                    12/12 passed (100%)
```

---

## 🎬 Runnable Demonstrations

```powershell
# 1. HR Golden Path — salary data access → automatic BLOCK
python scripts/run_golden_path.py

# 2. Agentic Tool-Call Governance — 8 scenarios: refund tiers, email, delete, session risk
python scripts/run_agent_governance_demo.py

# 3. Tamper-Evident Merkle Audit — 3 Acts: clean → naive tamper → rechaining attack
python scripts/run_audit_integrity_demo.py

# 4. Session Accumulator Demo — multi-turn risk memory
python demo_session_persistence.py

# 5. RAG Capability Evaluation
python -m rag.evaluation
```

---

## 📚 Research Grounding & Standards Compliance

| Methodology / Principle | Reference / Standard | System Implementation |
|---|---|---|
| **AI Risk Management Framework** | NIST AI RMF 1.0 (Govern, Map, Measure, Manage) | Multi-layer risk fusion, context enrichment, and complete audit trail |
| **AI Management System** | ISO/IEC 42001:2023 | Programmatic policy enforcement and human-in-the-loop review queues |
| **EU AI Act Transparency & High-Risk** | Regulation (EU) 2024/1689 (Articles 50 & Annex III) | Policy RAG corpus, hot-path bias tripwire, counterfactual fairness probe |
| **Faithfulness & NLI Grounding** | RAGAS; BEACON (arXiv:2606.07528); Vectara HHEM | Claim decomposition, DeBERTa cross-encoder NLI, SelfCheckGPT resampling |
| **Counterfactual Fairness** | Kusner et al. (NeurIPS 2017); LangFair (CVS Health) | Token perturbation and counterfactual flip-rate measurement |
| **Tamper-Evident Transparency** | RFC 6962 (Certificate Transparency); Sigstore Rekor | SHA-256 hash chains with domain-separated Merkle tree checkpoints |
| **Bayesian Signal Fusion** | Dempster-Shafer Evidence Theory; AWS Fraud Detector | Noisy-OR probability fusion preventing dilution of high-severity risks |
| **RLHF / DPO Alignment** | Rafailov et al. (arXiv:2305.18290); TRL library | Preference pair collection, LLM-as-judge labelling, per-category DPO fine-tuning |
| **Self-Governing Systems** | Control theory feedback loops; PID control principles | Override-rate-driven threshold auto-tuning with structural safety ceiling |
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
| `CONTROLPLANE_API_KEY` | `demo-key-001` | API key for gateway authentication |

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
| `GROQ_API_KEY` | — | Groq API key for Ask ControlPlane, Advanced Inspector, RLHF judging |
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
| `CONTROLPLANE_API_URL` | `http://127.0.0.1:8000` | Backend URL for Streamlit frontend |
| `CONTROLPLANE_API_KEY` | `demo-key-001` | API key for Streamlit → backend calls |

---

## 📄 License & Attribution

Built for the **Accenture Innovation Challenge 2026 (Track 1)**.  
All intellectual property and architecture designed for enterprise AI safety, transparency, and governance compliance.

> *"The same governance principle ControlPlane.ai applies to AI systems, it applies to itself."*
