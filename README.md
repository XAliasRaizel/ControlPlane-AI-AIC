# ControlPlane.ai — Enterprise AI Governance Control Plane

> **Accenture Innovation Challenge 2026 · Track 1**
> *A Production-Grade, Multi-Layered Governance Gateway for Generative AI & Autonomous Agent Systems.*

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.48.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-200%2B%20passing-brightgreen?logo=pytest)](https://pytest.org)
[![Ruff](https://img.shields.io/badge/Lint-ruff%20clean-006400?logo=python)](https://docs.astral.sh/ruff/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Proprietary-orange)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20openai%2Fgpt--oss--120b-f55036?logo=openai&logoColor=white)](https://console.groq.com)

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
- Continuously improved via a **production learning flywheel** that captures hot-path vs async disagreements as hard negative training examples

---

## Table of Contents

1. [Why ControlPlane.ai? — The Problem](#-why-controlplaneai--the-problem)
2. [Key Capabilities](#-key-capabilities)
3. [System Architecture](#-system-architecture)
4. [Repository Layout](#-repository-layout)
5. [Quick Start & Setup](#-quick-start--setup)
6. [API Reference](#-api-reference)
7. [Feature Deep Dives](#-feature-deep-dives)
   - [Hot-Path Detector Pipeline](#1-hot-path-detector-pipeline)
   - [Context-Aware RBAC (Two-Layer Permission Resolution)](#2-context-aware-rbac-two-layer-permission-resolution)
   - [Session Risk Accumulator](#3-session-risk-accumulator)
   - [Fast-Lane Post-Response Analysis](#4-fast-lane-post-response-analysis)
   - [Self-Governing Threshold Auto-Tuner](#5-self-governing-threshold-auto-tuner)
   - [Human Review Queue & Feedback Loop](#6-human-review-queue--feedback-loop)
   - [Policy RAG & Ask ControlPlane](#7-policy-rag--ask-controlplane)
   - [LLM-Powered Advanced Inspector](#8-llm-powered-advanced-inspector)
   - [Agentic Tool-Call Governance](#9-agentic-tool-call-governance)
   - [RLHF / DPO Preference Data Pipeline](#10-rlhf--dpo-preference-data-pipeline)
   - [Continuous Learning Flywheel](#11-continuous-learning-flywheel)
   - [Detector Training Pipeline (ML Scripts)](#12-detector-training-pipeline-ml-scripts)
   - [Tamper-Evident Merkle Audit Ledger](#13-tamper-evident-merkle-audit-ledger)
   - [Two-Layer Hallucination & Grounding Detection](#14-two-layer-hallucination--grounding-detection)
   - [Two-Layer Bias & Fairness Detection](#15-two-layer-bias--fairness-detection)
   - [Unified Sensitive Data Protection](#16-unified-sensitive-data-protection)
8. [Streamlit Dashboard — 7 Tabs](#-streamlit-dashboard--7-tabs)
9. [Testing & Evaluation Harness](#-testing--evaluation-harness)
10. [Runnable Demonstrations](#-runnable-demonstrations)
11. [Deployment](#-deployment)
12. [Research Grounding & Standards Compliance](#-research-grounding--standards-compliance)
13. [Environment Variables](#-environment-variables)
14. [CI/CD Pipeline](#-cicd-pipeline)

---

## Why ControlPlane.ai? — The Problem

Modern enterprises are deploying generative AI at scale — in HR copilots, finance chatbots, customer support agents, and internal knowledge assistants. Each deployment carries significant risk:

| Risk | Example | Without Governance |
|---|---|---|
| **PII Leakage** | Finance chatbot returns salary data to a contractor | GDPR breach, regulatory fines |
| **Prompt Injection** | Attacker overrides system prompt via user input | Data exfiltration, policy bypass |
| **Authorization Bypass** | Intern queries executive compensation data | Insider threat, compliance failure |
| **Hallucination** | LLM states incorrect regulatory facts in a patient context | Legal liability, patient harm |
| **Bias** | Loan decision cites applicant's ethnicity as a reason | ECOA/Title VII violation, reputational risk |
| **Session Evasion** | Attacker behaves well for 5 turns, exfiltrates on turn 6 | Undetected multi-turn attack |
| **Agentic Overreach** | Autonomous agent deletes customer records without authorization | Irreversible damage, audit failure |

**ControlPlane.ai** addresses all of these with a single, unified governance gateway that enforces policy on every AI interaction — before any content reaches users.

---

## Key Capabilities

| Layer / Feature | Description | Primary Location |
|---|---|---|
| **Hot-Path Gateway** | FastAPI ingress: API key auth, rate limiting, prompt-length guard, UUID-stamped lifecycle, SecurityHeaders | `backend/gateway/`, `backend/main.py` |
| **7 Parallel Detectors** | PII, Injection, Authorization, Safety, Hallucination, Bias, Sensitive-Query-Intent via `asyncio.gather` | `backend/detectors/` |
| **Noisy-OR Risk Engine** | Bayesian evidence fusion — prevents dilution of high-severity signals | `backend/risk/engine.py` |
| **Two-Layer RBAC** | Role-based + department-specific permission grants merged with OR logic | `backend/gateway/context_enrichment.py` |
| **Session Risk Accumulator** | Dual-signal cross-turn memory (EWMA + peak-decay), entity reconstruction, Redis support | `backend/risk/accumulator.py` |
| **Five Governance Decisions** | `ALLOW`, `MODIFY` (regex PII redaction), `REROUTE`, `HUMAN_REVIEW`, `BLOCK` | `backend/decision/engine.py` |
| **Hierarchical Policy Engine** | Multi-file YAML rules with Application > Department > Global precedence, hot-reloading | `backend/policy/` |
| **Self-Governing Auto-Tuner** | Override-rate-driven threshold nudging — applies same governance principle to itself | `backend/feedback/feedback_engine.py` |
| **Human Review Queue** | SQLite-backed queue; resolver propagates gold-label data to training flywheel | `backend/review/queue.py` |
| **Fast-Lane Post-Response** | Background async detectors (250ms timeout); high-risk trigger webhook RETRACT | `backend/main.py` |
| **Policy RAG Explainer** | Bounded (<40ms) semantic retrieval over GDPR, EU AI Act, HIPAA, internal policies | `rag/policy/policy_rag.py` |
| **Ask ControlPlane** | Compliance Q&A with hybrid retrieval + Groq citation synthesis | `rag/ask_controlplane/` |
| **Advanced Inspector** | Slow-path LLM-backed governance inspector (`POST /v1/inspect`) with injection-hardening | `backend/app/llm/` |
| **Agentic Tool Governance** | `ToolGovernor` intercepts agent tool calls before execution | `backend/agents/` |
| **RLHF / DPO Pipeline** | 1-in-N preference pair collection, LLM-judge labelling, DPO JSONL export, LoRA training | `rlhf/` |
| **Continuous Learning Flywheel** | Captures hot vs async disagreements (delta > 0.2) as training signals; gold labels from human review | `backend/async_pipeline/training_signal_collector.py` |
| **Detector Training Pipeline** | Production-grade ML scripts for SFT fine-tuning and threshold recalibration | `ml/scripts/` |
| **Merkle Audit Ledger** | SHA-256 hash chain + RFC 6962 Merkle checkpoints; detects naive and rechaining attacks | `backend/audit_integrity/` |
| **Two-Layer Hallucination** | Hot-path claim gate (<0.5ms) + async DeBERTa NLI + SelfCheckGPT | `backend/detectors/hallucination.py`, `backend/async_engines/grounding.py` |
| **Two-Layer Bias & Fairness** | Hot-path ECOA/Title-VII tripwire + async counterfactual flip-rate + LLM rubric | `backend/detectors/bias.py`, `backend/async_engines/fairness.py` |
| **Prometheus Metrics** | `controlplane_govern_requests_total`, `latency_seconds`, and more; Plotly dashboard | `backend/shared/metrics.py` |
| **OpenTelemetry Tracing** | Distributed traces via OTLP; zero-overhead when endpoint unset | `backend/shared/tracing.py` |

---

## System Architecture

```
AI Applications (chatbots, RAG pipelines, agents, copilots)
        |  prompt / candidate response / proposed tool-call
        v
============================== CONTROLPLANE.AI ==============================

  [1] GATEWAY (FastAPI + API Key Auth + Rate Limiting)
      auth . prompt-length guard . UUID request_id . SecurityHeaders
        |
        v
  [2] CONTEXT ENRICHMENT  (Two-Layer RBAC)
      role permissions --OR-- department permission grants
      -> can_access_salary, can_view_medical_records, can_access_banking...
        |
        v
  [3] HOT-PATH DETECTORS  (asyncio.gather, <50ms budget)
      +- PII & Sensitive Terms (6 categories + fail-cautious safety net)
      +- Prompt Injection & Jailbreak (DAN, instruction overrides, extraction)
      +- Authorization & RBAC (role entitlements vs. resource sensitivity)
      +- Safety & Harmful Content (violence, hacking, exploits)
      +- Hallucination Fast Gate (claim-level cross-reference, <0.5ms)
      +- Bias Fast Gate (protected-attribute causal tripwire)
      +- Sensitive Query Intent (detail-seeking language heuristic)
        |
        v
  [4] RISK ENGINE  (Noisy-OR Evidence Fusion)
      P(risk) = 1 - prod(1 - p_i) + context amplifier (1.35x for critical apps)
      + session_risk injection (EWMA / peak-decay when accumulator enabled)
        |
        v
  [5] POLICY ENGINE & POLICY RAG  (parallel)
      YAML rules: Application > Department > Global precedence
      +-> Policy RAG retrieves regulatory citations (~2ms warm)
        |
        v
  [6] DECISION ENGINE & SANITIZATION
      ALLOW . MODIFY (regex redaction) . REROUTE . BLOCK . HUMAN_REVIEW ---+
        |                                                                   |
        |  fire-and-forget background                                       v
        |                                                      [7] HUMAN REVIEW QUEUE
        |                                                          approve / reject / modify
        |                                                          -> gold-label training signal
        |                                                                   |
        |                                                                   v
        |                                                   [8] SELF-GOVERNING AUTO-TUNER
        |                                                       NUDGE / ESCALATE / HOLD
        |                                                       -> YAML threshold patches
        |<------------------------------------------------------------------+
        v
  Sanitized response returned to application (+ policy_evidence annotation)
        |
        +-- background: RLHF pair sampling (1-in-N)
        +-- background: fast-lane async detectors (250ms, webhook RETRACT)
        +-- background: deep async analytics pipeline
              +- Grounding RAG & Deep NLI Engine (DeBERTa + SelfCheckGPT)
              +- Deep Fairness Engine (Counterfactual probe + LLM judge)
              +- Performance, Cost, Privacy, Safety, Business engines
              +-> Training Signal Collector
                    captures hot vs async disagreements (delta > 0.2)
                    -> rlhf/data/detector_training/raw_signals_<DATE>.jsonl

  [9] TAMPER-EVIDENT AUDIT LEDGER
      HMAC-hashed audit context . SHA-256 hash chain . Merkle checkpoints

  [10] SESSION ACCUMULATOR (cross-turn, opt-in)
       EWMA score + peak-with-decay -> session_risk, session_band (1/2/3)
       PII entity-reconstruction detection across rolling fragment window
       Tool-chain contamination tracking (sticky per session TTL)
       Backends: InMemorySessionStore (default) | Redis (multi-worker)

  [11] RLHF / DPO PIPELINE (background, non-blocking)
       Category-validated preference pairs -> LLM judge labelling
       -> DPO JSONL export -> LoRA fine-tuning (rlhf/training/ stubs)

  [12] CONTINUOUS LEARNING FLYWHEEL
       Hot path disagreements -> training data -> SFT fine-tune -> recalibrate
       Human review resolutions -> gold labels -> highest priority training data
       Runs every 6 hours via recalibrate_thresholds.py (online, no downtime)

===========================================================================

  [13] AGENTIC GOVERNANCE (/agent/act)
       Agent proposes tool call -> ToolGovernor intercepts -> Risk + Policy
       -> ALLOW (executes) | HUMAN_REVIEW (held) | BLOCK (aborts)
```

---

## Repository Layout

```
controlplane-ai/
|-- Dockerfile                           # Multi-stage production container (non-root user)
|-- docker-compose.yml                   # Two-service stack: API + Streamlit UI
|-- requirements.txt                     # Core runtime dependencies
|-- pyproject.toml                       # pytest config + ruff lint rules
|-- start.ps1                            # PowerShell one-shot dev launcher
|
|-- backend/
|   |-- main.py                          # FastAPI entrypoint (40+ routes, lifespan hooks, async task queue)
|   |-- shared/
|   |   |-- schemas.py                   # Canonical Pydantic v2 contracts (single source of truth)
|   |   |-- sensitive_terms.py           # Unified taxonomy: financial, IDs, HR, medical, auth
|   |   |-- config.py                    # .env loader + Settings dataclass
|   |   |-- model_backend.py             # Lazy model loader (CONTROLPLANE_MODEL_<TASK>)
|   |   |-- circuit_breaker.py           # Thread-safe CircuitBreaker (CLOSED/OPEN/HALF_OPEN)
|   |   |-- db_pool.py                   # Thread-local SQLite pool (WAL mode)
|   |   |-- gpu_adapter.py               # Hardware inference interface
|   |   |-- llm_simulator.py             # Synthetic LLM response generator (dev/test)
|   |   |-- logging_config.py            # Structured JSON logging + request_id/trace_id context
|   |   |-- metrics.py                   # Prometheus counters and histograms
|   |   +-- tracing.py                   # OpenTelemetry OTLP tracing configuration
|   |-- gateway/
|   |   |-- auth.py                      # API key authentication dependency
|   |   +-- context_enrichment.py        # Two-layer RBAC: role perms + department grants (OR logic)
|   |-- detectors/
|   |   |-- base.py                      # BaseDetector ABC + self-registration DETECTOR_REGISTRY
|   |   |-- pii.py                       # Sensitive term & regex value scanner (+ Presidio opt-in)
|   |   |-- injection.py                 # Jailbreak & instruction-override scanner
|   |   |-- authorization.py             # Deterministic RBAC access check
|   |   |-- safety.py                    # Harmful content & exploit scanner
|   |   |-- hallucination.py             # Hot-path ungrounded-claim gate (<0.5ms)
|   |   |-- bias.py                      # Hot-path protected-attribute causal detector
|   |   |-- sensitive_query_intent.py    # Detail-seeking language heuristic
|   |   +-- async_analytics.py           # 7 background analytics engine detectors
|   |-- async_engines/
|   |   |-- grounding.py                 # Deep NLI entailment + LLM-judge + SelfCheckGPT
|   |   +-- fairness.py                  # Counterfactual probe + LLM bias rubric
|   |-- app/
|   |   +-- llm/
|   |       |-- client.py                # Groq-backed LLM client (injection-hardened evidence)
|   |       |-- prompt_registry.py       # Semver-versioned prompt template registry
|   |       |-- prompts.py               # Inspector system prompt + result parser
|   |       |-- schemas.py               # LLM request/response schemas
|   |       +-- token_budget.py          # tiktoken-based cost budget enforcement
|   |-- utils/
|   |   |-- claims.py                    # Lightweight claim decomposition utility
|   |   +-- llm_judge.py                 # Provider-agnostic AI-as-judge (Groq/OpenAI/Anthropic/Mock)
|   |-- risk/
|   |   |-- engine.py                    # Noisy-OR Bayesian risk fusion & session injection
|   |   |-- accumulator.py               # Dual-signal EWMA+peak accumulator & entity reconstruction
|   |   +-- session_store.py             # InMemorySessionStore + Redis SessionStore protocol
|   |-- policy/
|   |   |-- engine.py                    # Multi-scope hierarchical policy evaluator
|   |   +-- loader.py                    # Hot-reloading YAML policy loader & validator
|   |-- decision/
|   |   +-- engine.py                    # Decision resolution & pattern-based PII redaction
|   |-- review/
|   |   +-- queue.py                     # SQLite human review queue (enqueue / resolve + gold labels)
|   |-- async_pipeline/
|   |   |-- publisher.py                 # Fire-and-forget dispatcher with dead-letter SQLite store
|   |   |-- worker.py                    # Background job executor with smart sampling gate
|   |   |-- consumers.py                 # Analytics orchestrator (runs async-only detectors)
|   |   +-- training_signal_collector.py # Continuous learning: captures hot vs async disagreements
|   |-- agents/
|   |   |-- models.py                    # ToolCallContext, GovernanceDecision, PendingToolCall
|   |   |-- tools.py                     # Tool execution registry (send_email, issue_refund, delete_record)
|   |   |-- risk.py                      # Tool-call risk scoring (sensitivity + magnitude + reversibility)
|   |   |-- policy.py                    # YAML-driven agent policy evaluator (restricted _safe_eval)
|   |   |-- queue.py                     # In-memory pending tool-call review queue
|   |   |-- governance.py                # ToolGovernor: intercept -> score -> decide -> execute
|   |   +-- router.py                    # FastAPI routes (/agent/act, /agent/pending)
|   |-- audit/
|   |   +-- store.py                     # SQLite audit DB, HMAC privacy fingerprinting, richer_metrics()
|   |-- audit_integrity/
|   |   |-- models.py                    # AuditRecord, Checkpoint, VerificationResult
|   |   |-- hashing.py                   # Canonical JSON + SHA-256 + HMAC utilities
|   |   |-- merkle.py                    # RFC 6962 Merkle tree with inclusion proofs
|   |   |-- backends.py                  # SQLite record store & append-only anchor file
|   |   |-- ledger.py                    # TamperEvidentAuditLedger (append + seal)
|   |   +-- verifier.py                  # Independent chain & checkpoint verifier
|   +-- feedback/
|       |-- evaluator.py                 # FPR/FNR labelled error evaluator & RLHF pair generator
|       +-- feedback_engine.py           # Self-Governing Threshold Auto-Tuner (NUDGE/ESCALATE/HOLD)
|
|-- rag/
|   |-- config.py                        # RAG settings & latency budgets
|   |-- schemas.py                       # Chunk, Query, Document, RetrievalResult
|   |-- embeddings.py                    # Local TF-IDF/LSA + Sentence-Transformers embedder
|   |-- vector_store.py                  # ChromaDB + zero-dependency NumPy/JSON fallback
|   |-- chunking.py                      # Paragraph-aware text chunking
|   |-- retriever.py                     # Hybrid vector + lexical (BM25) retriever
|   |-- evaluation.py                    # End-to-end RAG evaluation harness (12 cases)
|   |-- tenant.py                        # Multi-tenant RAG corpus isolation
|   |-- corpus/
|   |   |-- regulatory/                  # GDPR, EU AI Act, HIPAA knowledge bases
|   |   +-- internal_kb/                 # Leave, IT security, expense, company policies
|   |-- ingestion/
|   |   |-- ingest.py                    # Corpus ingestion & index builder
|   |   |-- document_loader.py           # Text & Markdown loader
|   |   |-- policy_loader.py             # YAML policy -> prose converter
|   |   +-- audit_loader.py              # Privacy-safe audit record indexer
|   |-- policy/
|   |   +-- policy_rag.py                # Hot-path Policy RAG explainer (~2ms warm)
|   |-- grounding/
|   |   |-- claim_extractor.py           # Sentence-level checkable claim extractor
|   |   |-- entailment.py                # Lexical & number-penalty entailment checker
|   |   +-- grounding_checker.py         # Grounding verification orchestrator
|   +-- ask_controlplane/
|       |-- retrieval.py                 # Hybrid policy + audit retrieval
|       |-- chat.py                      # Q&A synthesizer with citation verification
|       +-- llm_client.py                # Groq LLM client with graceful fallback
|
|-- rlhf/
|   |-- config.py                        # Category enum, storage selector, sampling rate, daily caps
|   |-- schema.py                        # PreferencePair Pydantic model
|   |-- sampler.py                       # maybe_collect_pair -- 1-in-N fire-and-forget hook
|   |-- generators/
|   |   |-- api_vs_api.py                # Dual Groq API model generation
|   |   +-- local_vs_local.py            # Local model pair generation
|   |-- judges/
|   |   |-- llm_judge.py                 # LLM-as-Judge with position-bias control
|   |   +-- human_judge.py               # CLI-based human preference labelling
|   |-- storage/
|   |   |-- categorize.py                # Category validation (HR/FINANCIAL/GENERAL)
|   |   |-- json_store.py                # Active: append-only JSONL backend
|   |   +-- sqlite_store.py              # Drop-in SQLite backend
|   |-- export/
|   |   |-- dpo_export.py                # DPO JSONL export with label filtering
|   |   +-- filters.py                   # Filter pipeline: unlabelled, ties, errors, near-duplicates
|   +-- training/
|       |-- dataset.py                   # HuggingFace Dataset loader for DPO training
|       |-- dpo_config.py                # Per-category DPO + LoRA hyperparameters
|       |-- evaluate.py                  # Reward margin + human consistency evaluation
|       +-- train.py                     # DPO fine-tuning via TRL DPOTrainer + PEFT
|
|-- ml/
|   |-- train_detector.py                # Generic SFT fine-tuning entry point
|   |-- train_prompt_injection.py        # Injection-specific training pipeline
|   |-- common/                          # data_utils.py, eval_utils.py, lora_utils.py
|   |-- fairness/train.py                # Counterfactual fairness model training
|   |-- grounding/                       # NLI model evaluation scripts
|   |-- safety/                          # Safety classifier evaluation scripts
|   |-- prompt_injection/                # Injection classifier pipeline
|   |-- notebooks/train_detectors.py     # Orchestrated training notebook
|   +-- scripts/
|       |-- build_detector_dataset.py    # Build training datasets from prod signals (40/60 mix)
|       |-- train_from_production.py     # Master orchestrator: data -> SFT -> recalibrate
|       |-- recalibrate_thresholds.py    # Online threshold recalibration (every 6h, no downtime)
|       |-- train_all_detectors.py       # Train all 4 detector tasks in sequence
|       |-- train_scaled_detectors.py    # HuggingFace + Kaggle dataset training
|       |-- calibrate_sensitive_intent.py # Embedding-based intent threshold calibration
|       |-- calibrate_session_accumulator.py # EWMA/peak parameter calibration
|       |-- compare_detectors.py         # A/B comparison of model versions
|       |-- evaluate_model.py            # Per-task model evaluation suite
|       |-- export_onnx.py               # ONNX export for production inference
|       |-- download_pretrained.py       # Pre-trained model download scripts
|       +-- download_minilm.py           # MiniLM embedding model downloader
|
|-- policies/
|   |-- global.yaml                      # Universal fallthrough governance rules
|   |-- hr.yaml                          # HR-scoped policy (PII & authorization)
|   |-- finance.yaml                     # Finance-scoped policy (loan decisions, PII redaction)
|   |-- support.yaml                     # Support bot policy (injection & redaction)
|   |-- agent_tools.yaml                 # 7 rules governing tool calls (refunds, email, delete)
|   +-- hallucination_bias_rules.yaml    # Hallucination and bias threshold rules
|
|-- frontend/
|   |-- streamlit_app.py                 # Interactive Streamlit UI -- 7 tabs, ~2700 lines
|   |-- components.py                    # Shared Streamlit widget components
|   +-- theme.css                        # Custom Streamlit CSS theme
|
|-- tenants/
|   |-- default.yaml                     # Default tenant configuration
|   |-- acme_corp.yaml                   # ACME Corp tenant (demo)
|   +-- demo_tenant.yaml                 # Demo tenant
|
|-- scripts/
|   |-- run_golden_path.py               # HR golden-path end-to-end demo
|   |-- run_agent_governance_demo.py     # 8-scenario agent tool-call demo
|   |-- run_audit_integrity_demo.py      # 3-act tamper-evident ledger demo
|   +-- eval_rag_triad.py               # RAG triad CI evaluation script
|
|-- k8s/
|   |-- namespace.yaml                   # controlplane namespace
|   |-- deployment.yaml                  # API Deployment (2 replicas, resource limits)
|   |-- service.yaml                     # ClusterIP service
|   |-- hpa.yaml                         # HPA: scale 2->10 replicas on CPU > 70%
|   |-- pvc.yaml                         # PersistentVolumeClaim for audit DB
|   +-- secrets.example.yaml             # Secret manifest template
|
|-- docs/
|   |-- architecture.md                  # Detailed architecture documentation
|   |-- AGENT_GOVERNANCE_QUICKSTART.md   # Agentic governance quick start guide
|   |-- AUDIT_INTEGRITY_QUICKSTART.md    # Tamper-evident ledger quick start
|   |-- agent_tool_governance_spec.md    # Full agent tool governance specification
|   |-- audit_integrity_spec.md          # Merkle audit ledger specification
|   +-- pii_gap_diagnosis.md             # PII detection coverage analysis
|
+-- tests/                               # 27 test modules, 200+ tests
    |-- test_golden_path.py
    |-- test_governance.py
    |-- test_gateway_api.py
    |-- test_policy_engine.py
    |-- test_async_service.py
    |-- test_agent_governance.py
    |-- test_audit_integrity.py
    |-- test_hallucination_bias.py
    |-- test_sensitive_data_coverage.py
    |-- test_rag.py
    |-- test_session_accumulator.py
    |-- test_model_backend.py
    |-- test_rlhf_integration.py
    |-- test_llm_client.py
    |-- test_groq_llm_client.py
    |-- test_fast_lane.py
    |-- test_adversarial.py
    |-- test_multi_tenant.py
    |-- test_observability.py
    |-- test_performance.py
    |-- test_reliability.py
    |-- test_scalability.py
    |-- test_security.py
    |-- test_prompt_registry.py
    |-- test_ml_pipeline.py
    |-- test_paraphrase_consistency.py
    +-- test_accumulator_calibration_integrity.py
```

---

## Quick Start & Setup

### Prerequisites

- **Python 3.11+**
- A [Groq API key](https://console.groq.com/keys) for LLM-powered features
- Git

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
copy .env.example .env
```

Minimum settings for full functionality:

```env
GROQ_API_KEY=gsk_your-groq-key-here
GROQ_MODEL=openai/gpt-oss-120b
CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED=true
CP_JUDGE_PROVIDER=groq
```

### 3. Build RAG Indices

```powershell
python -m rag.ingestion.ingest
```

### 4. Run the Test Suite

```powershell
# Full suite
pytest -q

# Skip slow/performance tests (CI mode)
pytest -q -m "not slow"
```

### 5. Start the Application

**Local (two terminals):**

```powershell
# Terminal 1 -- Backend
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 -- Frontend
python -m streamlit run frontend/streamlit_app.py --server.port 8501
```

**One-command launcher:**

```powershell
.\start.ps1
```

**Docker Compose:**

```powershell
docker compose up --build
```

| Service | URL |
|---|---|
| **Streamlit Dashboard** | http://localhost:8501 |
| **API Swagger / OpenAPI** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |
| **Prometheus Metrics** | http://localhost:8000/metrics |

---

## API Reference

### Core Governance

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/govern` | API Key | **Core governance endpoint** -- full hot-path pipeline |
| `POST` | `/v1/chat` | API Key | Human-readable chat wrapper around `/v1/govern` |
| `POST` | `/v1/inspect` | API Key | Slow-path LLM-backed advanced inspector |
| `GET` | `/health` | None | Gateway health, registered detectors, policy summary |
| `GET` | `/v1/health/deep` | None | Deep probe: DB, circuit breaker, session store, ML thread pool |

**Example request:**

```json
POST /v1/govern
Authorization: demo-key-001

{
  "user_id": "emp-101",
  "user_role": "employee",
  "department": "HR",
  "application_id": "hr-copilot",
  "prompt": "Show me John Smith's salary and performance review",
  "data_classification": "HIGH",
  "session_id": "sess-abc123"
}
```

**Example response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "decision": {
    "action": "BLOCK",
    "reason": "Unauthorized access to PII-classified data",
    "policy_id": "hr-pii-unauthorized"
  },
  "risk": {
    "overall_risk": 0.87,
    "session_risk": 0.23,
    "session_band": 1
  },
  "detectors": [
    { "detector_name": "pii", "score": 0.85, "label": "PII_DETECTED" },
    { "detector_name": "authorization", "score": 0.92, "label": "UNAUTHORIZED_ACCESS" }
  ],
  "policy_evidence": { "citations": [], "regulatory_refs": [] },
  "async_job_id": "async-550e8400",
  "latency_ms": 23.4
}
```

### Metrics & Audit

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/metrics` | API Key | Aggregate request counts and decision distribution |
| `GET` | `/v1/metrics/rich` | API Key | Extended metrics: risk distribution, latency trend, detector fire rates |
| `GET` | `/v1/requests` | API Key | Recent request list (up to 200) |
| `GET` | `/v1/audits` | API Key | Recent audit records (up to 200) |
| `GET` | `/v1/audits/{request_id}` | API Key | Single audit record |
| `GET` | `/v1/audit/integrity` | API Key | Full Merkle + hash-chain tamper verification |

### Policy & Reviews

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/policies` | API Key | Loaded policy rules summary |
| `GET` | `/v1/reviews` | API Key | Pending human review queue |
| `POST` | `/v1/reviews/{id}/resolve` | API Key | Resolve review -- propagates to auto-tuner + training flywheel |
| `POST` | `/v1/feedback` | API Key | Submit reviewer feedback override |

### Self-Governing Auto-Tuner

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/feedback/tuning` | API Key | **Dry-run** -- preview NUDGE / ESCALATE / HOLD |
| `POST` | `/v1/feedback/tuning/apply` | API Key | **Apply** -- write threshold changes to YAML files |
| `POST` | `/v1/feedback/tuning/seed-demo` | API Key | Seed 25 realistic review records |
| `GET` | `/v1/feedback/tuning/history` | API Key | Full audit trail of threshold changes |

### Ask ControlPlane (RAG)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/ask-controlplane` | API Key | Compliance Q&A with citation-grounded answers |
| `POST` | `/v1/ask-controlplane/reindex` | Admin Key | Rebuild the audit index |

### Session & Async Jobs

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/session/{session_id}` | API Key | Live session accumulator state |
| `GET` | `/v1/jobs/{job_id}` | API Key | Async analytics job status |
| `GET` | `/v1/async/{request_id}` | API Key | Async job by governance request ID |

### RLHF

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/rlhf/status` | None | Preference pair collection statistics |
| `POST` | `/v1/rlhf/export` | None | Trigger DPO JSONL export |
| `GET` | `/v1/rlhf/export/latest` | None | Retrieve latest export content |

### Agent Governance

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/agent/act` | API Key | Agent tool-call governance endpoint |
| `GET` | `/agent/pending` | API Key | Pending agent actions awaiting human review |
| `POST` | `/agent/pending/{id}/resolve` | API Key | Resolve a pending agent action |

### Admin

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/admin/reload-models` | Admin Key | Hot-reload all ML model caches |
| `GET` | `/v1/admin/dead-letters` | API Key | List failed async events in dead-letter store |
| `POST` | `/v1/admin/dead-letters/{id}/retry` | API Key | Retry a failed async event |
| `DELETE` | `/v1/admin/dead-letters/{id}` | API Key | Delete a dead-letter entry |

---

## Feature Deep Dives

### 1. Hot-Path Detector Pipeline

Seven detectors are loaded via a self-registration `DETECTOR_REGISTRY` and executed concurrently via `asyncio.gather`. Each returns a `DetectorResult` with a `score` (0-1), `label`, `confidence`, and `evidence` list.

| Detector | What it catches |
|---|---|
| `pii` | 6 data categories: financial (card, CVV, UPI, IFSC), government IDs (PAN, Aadhaar, SSN, Passport, DL), HR records, medical history, credentials + fail-cautious safety net |
| `injection` | Instruction-override signatures, DAN jailbreaks, prompt extraction, role-manipulation patterns |
| `authorization` | RBAC: verifies the requesting user's role+department against the sensitivity of the data being accessed |
| `safety` | Violence, hacking, exploit, and harmful content patterns |
| `hallucination` | Extracts checkable claims (dates, numbers, named entities) and cross-references `retrieved_context` in <0.5ms |
| `bias` | Protected-attribute proximity detector -- flags age, gender, race, religion, disability cited as reasons in decision contexts (ECOA / Title VII / EU AI Act) |
| `sensitive_query_intent` | Heuristic for detail-seeking language about named third parties when explicit PII keywords are absent |

---

### 2. Context-Aware RBAC (Two-Layer Permission Resolution)

The `enrich_context()` function in `backend/gateway/context_enrichment.py` applies a two-layer permission model:

**Layer 1 -- Role-Based Permissions:**

| Role | can_access_salary | can_view_medical_records | can_access_banking | is_admin |
|---|---|---|---|---|
| `admin` | Yes | Yes | Yes | Yes |
| `hr-manager` | Yes | No | No | No |
| `finance-manager` | No | No | Yes | No |
| `doctor` | No | Yes | No | No |
| `employee` | No | No | No | No |
| `security_auditor` | No | No | No | No |

**Layer 2 -- Department-Specific Overrides (OR logic):**

| Department | Grants |
|---|---|
| `medical` | `can_view_medical_records = True` |
| `hr` | `can_access_salary = True` |
| `finance` / `payroll` | `can_access_banking = True` |
| `legal` | `can_view_medical_records = True` |
| `security` | `is_security_auditor = True` |

**Key example:** An `employee` in `department="HR"` gets `can_access_salary=True` from the department grant (they need it for their job). An `employee` in `department="Sales"` does not, so the `authorization` detector fires >= 0.5, triggering `hr-pii-unauthorized -> BLOCK`.

---

### 3. Session Risk Accumulator

Enabled via `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED=true`. Tracks risk across conversation turns using a dual-signal design:

- **EWMA score** (`alpha=0.01`) -- exponentially weighted per-turn average; responsive to sustained patterns
- **Peak-with-decay** (`peak_decay=0.99`) -- remembers the worst turn seen, prevents session-evasion attacks
- **Session risk** = `max(EWMA, peak)`, classified into Band 1 / 2 / 3

Additional capabilities:
- **Entity reconstruction**: Maintains a rolling PII fragment window. Concatenating fragments across turns that collectively trigger the PII regex -- even when no individual turn would -- is detected and flagged
- **Tool-chain contamination**: Once sensitive data touches a tool, that tool is marked contaminated for the session TTL
- **Fast-lane integration**: A fired fast-lane correction adds a risk spike of 1.0 on the next accumulator update
- **Storage backends**: `InMemorySessionStore` (default) or `RedisSessionStore` (multi-worker)

---

### 4. Fast-Lane Post-Response Analysis

After the synchronous governance decision is returned, any detectors marked `fast_async=True` run as a background task with a **250ms timeout**. On high-risk results:

- The SQLite job record is updated with `fast_lane_results`
- If the request included a `fast_lane_webhook` URL, a `POST` with `action=RETRACT` is sent -- allowing the application to retract an already-delivered response
- The fast-lane correction count feeds into the session accumulator for the next turn

---

### 5. Self-Governing Threshold Auto-Tuner

ControlPlane applies the same governance principle to **itself**.

```
Review Queue Resolutions --> Override Rate per Rule --> Tuning Decision
```

| Condition | Decision | Effect |
|---|---|---|
| < 5 resolved reviews | **INSUFFICIENT DATA** | No change |
| Override rate < 25% | **HOLD** | Rule is performing correctly |
| 25% <= rate < 50% | **NUDGE** | Detector threshold raised +0.05 in YAML |
| Override rate >= 50% | **ESCALATE** | Stop nudging -- rule needs human redesign |

**Structural safety guarantee:** Can only push thresholds **up**. Structurally cannot lower thresholds or make the system less safe.

**Audit trail:** Every applied NUDGE is logged to the `tuning_history` table with timestamp, rule ID, old/new threshold, override rate, and reasoning.

---

### 6. Human Review Queue & Feedback Loop

When the policy engine issues a `HUMAN_REVIEW` decision, the request is **held** in the review queue. Reviewers resolve via `POST /v1/reviews/{id}/resolve` with:
- `final_action` -- the actual disposition (`ALLOW`, `BLOCK`, `MODIFY`, `REROUTE`)
- `reviewer_id` -- audit identity
- `notes` -- free-text rationale

**Full feedback loop:**
1. Resolution feeds the Self-Governing Auto-Tuner (override rate tracking)
2. Resolution reconstructs the original `GovernanceRequest` from audit DB and writes a **gold-label training signal** (`label_confidence=1.0`) to the training collector

---

### 7. Policy RAG & Ask ControlPlane

**Policy RAG** (`rag/policy/policy_rag.py`) runs in parallel with the decision engine:
- Constructs a domain-specific retrieval query from context (role, department, matched rule, data classification, action)
- Retrieves matching clauses from GDPR, EU AI Act, HIPAA, and internal YAML policies
- Annotates the `GovernanceResponse` with a `policy_evidence` field. **Never blocks or alters decisions.**
- Warm path: ~2ms; cold path: <40ms budget enforced

**Ask ControlPlane** (`rag/ask_controlplane/`) is an interactive compliance Q&A assistant:
- Hybrid retrieval over the policy corpus and the SQLite audit database
- Synthesises answers via Groq with citation verification
- Falls back to extractive mode when no API key is configured

---

### 8. LLM-Powered Advanced Inspector

`POST /v1/inspect` runs a **separate slow-path** LLM analysis via Groq:
- Returns: `applicable_policy`, `evidence_refs`, `detected_risk`, `reason`, `required_controls`, `recommendation`, `citation_check`, `latency_ms`
- All evidence wrapped in injection-hardening delimiters (`<evidence>...</evidence>`)
- Verifies every `[N]` citation against the actual evidence count
- **Never enforces policy** -- the LLM describes; all enforcement stays in the hot path

---

### 9. Agentic Tool-Call Governance

Autonomous agents propose tool calls via `POST /agent/act`. The `ToolGovernor` intercepts before execution:

**Policy evaluation** (`policies/agent_tools.yaml`):
- Refunds < $500 -> `ALLOW`
- Refunds $500-$2000 -> `HUMAN_REVIEW`
- Refunds > $2000 -> `BLOCK`
- Deleting PII-flagged records without admin role -> `BLOCK`
- Sending PII to external email -> `BLOCK`
- Session risk > 0.6 -> `HUMAN_REVIEW` for any tool call

**Outcomes:** `ALLOW` (tool executes), `HUMAN_REVIEW` (held in queue), `BLOCK` (aborted with reason)

---

### 10. RLHF / DPO Preference Data Pipeline

A production-ready data flywheel for fine-tuning governance models:

1. **Sampling** (`rlhf/sampler.py`): `maybe_collect_pair()` fires on every `/v1/chat` -- 1-in-N sampling
2. **Generation** (`rlhf/generators/`): Two model responses generated concurrently
3. **Category validation** (`rlhf/storage/categorize.py`): Assigns `Category` (HR, FINANCIAL, GENERAL)
4. **Storage** (`rlhf/storage/json_store.py`): Append-only JSONL; SQLite drop-in via `RLHF_STORAGE_BACKEND=sqlite`
5. **Labelling** (`rlhf/judges/`): LLM-as-Judge with position-bias control (order swapped; ties on disagreement)
6. **Export** (`rlhf/export/dpo_export.py`): Filters and writes `{prompt, chosen, rejected}` DPO JSONL files
7. **Training** (`rlhf/training/`): Per-category `DPORunConfig` with LoRA r/alpha/modules, integrated with TRL `DPOTrainer`
8. **Evaluation** (`rlhf/training/evaluate.py`): Average reward margin + human-prompt consistency check

---

### 11. Continuous Learning Flywheel

The most significant architectural addition -- a closed-loop learning system that improves detectors from production traffic:

```
Every API request
  |
  +-- Hot path detectors run synchronously (<50ms)
  |
  +-- Async pipeline fires in background (worker.py)
        |
        +-- Runs deep LLM analytics engines
        |
        +-- Compares hot vs async scores per detector task
        |
        +-- If |delta| > 0.2 -> writes to rlhf/data/detector_training/
        |     Label priority:
        |       human override (confidence=1.0, gold)
        |       > high-conf async score (confidence=0.9, silver)
        |       > large disagreement async decision (confidence=0.7, bronze)
        |     PII redaction applied before disk persistence
        |     Context-prefix for authorization: [DEPT:...] [ROLE:...] [CLASS:...]
        |
        +-- Smart sampling gate:
              risk >= CONTROLPLANE_ASYNC_LLM_RISK_THRESHOLD (0.20) -> 100% LLM analysis
              risk < threshold -> CONTROLPLANE_ASYNC_LLM_SAMPLE_RATE (5%) sampled

Human review resolutions
  |
  +-- collect_human_override() writes gold-label training examples (confidence=1.0)

ml/scripts/recalibrate_thresholds.py (every 6h, online)
  |
  +-- Evaluates recent 500 production signals per task
  +-- Optimises thresholds at target FNR=0.03
  +-- Hot-writes calibration.json (live on next request, no restart)

ml/scripts/train_from_production.py (periodic, scheduled)
  |
  +-- Refreshes per-task datasets (40% public baseline + 60% production)
  +-- SFT fine-tunes injection, safety, pii, fairness classifiers
  +-- Threshold recalibration for embedding-based tasks
```

**Max collection rate:** 10,000 records/day (rotating daily JSONL)
**Location:** `rlhf/data/detector_training/raw_signals_YYYY-MM-DD.jsonl`

---

### 12. Detector Training Pipeline (ML Scripts)

| Script | Purpose |
|---|---|
| `build_detector_dataset.py` | Assembles 40% public baseline + 60% production signals; 3x oversampling for disagreements |
| `train_from_production.py` | Master orchestrator: fetch data -> SFT -> recalibrate all tasks |
| `recalibrate_thresholds.py` | Online threshold recalibration (no downtime, runs every 6h) |
| `train_all_detectors.py` | Train injection, safety, pii, fairness classifiers in sequence |
| `train_scaled_detectors.py` | Large-scale training from HuggingFace + Kaggle datasets |
| `calibrate_sensitive_intent.py` | Embedding-based intent threshold calibration |
| `calibrate_session_accumulator.py` | EWMA/peak parameter calibration |
| `compare_detectors.py` | A/B performance comparison between model versions |
| `evaluate_model.py` | Per-task evaluation (precision, recall, F1, AUC) |
| `export_onnx.py` | ONNX export for quantized production inference |
| `download_pretrained.py` | Download pretrained NLI/fairness/injection models |

---

### 13. Tamper-Evident Merkle Audit Ledger

Every governance decision is recorded with a cryptographic chain of custody:

- **Privacy-preserving hashing**: Audit contexts are HMAC-hashed -- raw prompts never appear in plaintext
- **SHA-256 hash chain**: `SHA-256(H_{i-1} || canonical_JSON(R_i))` -- any modification breaks the chain
- **RFC 6962 Merkle tree checkpoints**: Every 10 records, a Merkle tree is computed. Root is HMAC-signed and written to the independent anchor file (`controlplane.integrity.jsonl`)
- **Independent verifier**: Detects both naive (modify record content) and sophisticated (re-compute hash chain) attacks
- **Integrity endpoint**: `GET /v1/audit/integrity` returns `TAMPER_FREE` or `TAMPERED` with first broken sequence number

---

### 14. Two-Layer Hallucination & Grounding Detection

**Hot-path** (`detectors/hallucination.py`):
- Extracts specific checkable claims (dates, numbers, named entities)
- Cross-references against `retrieved_context` in <0.5ms

**Deep async** (`async_engines/grounding.py`):
- Ensembles **NLI cross-encoder entailment** (`cross-encoder/nli-deberta-v3-base`), **LLM-as-judge**, and **SelfCheckGPT self-consistency resampling**

**Grounding RAG** (`rag/grounding/`):
- Lexical and number-penalty entailment checker scoring claims against the internal knowledge base corpus

---

### 15. Two-Layer Bias & Fairness Detection

**Hot-path** (`detectors/bias.py`):
- Fast regex proximity detector catching protected attributes cited as explicit reasons in decision contexts
- Covers ECOA, Title VII, EU AI Act compliance

**Deep async** (`async_engines/fairness.py`):
- Generates counterfactual name/pronoun/attribute swaps and computes the **counterfactual flip rate**
- Applies an LLM-judge bias rubric across multiple fairness dimensions

---

### 16. Unified Sensitive Data Protection

**Single source of truth** (`backend/shared/sensitive_terms.py`): Imported by PII detector, authorization checker, and session accumulator.

| Category | Examples |
|---|---|
| Financial | Credit card numbers, CVV, UPI IDs, IFSC codes, bank account numbers |
| Government IDs | PAN, Aadhaar, Passport, Driver's License, SSN, GSTIN |
| HR Records | Salary, CTC, compensation, employment status, performance ratings |
| Medical | Diagnoses, prescriptions, health conditions, lab results |
| Credentials | Passwords, API keys, tokens, private keys |
| Fail-cautious Net | Third-party detail-seeking language ("give me / show me details about [person]") |

---

## Streamlit Dashboard -- 7 Tabs

The dashboard at **http://localhost:8501** provides a comprehensive operations interface.

### Sidebar
- **Backend Connected / Backend Offline** status pill (15-second cached health check)
- Configurable caller context: User ID, Role, Department, Application, Data Classification, API Key
- **Live Session Risk Monitor** -- polls `GET /v1/session/{session_id}` every 10 seconds

### Tab 1 -- Governance Chatbot
Interactive chat with real-time governance. Each response shows:
- Decision badge (ALLOW / MODIFY / BLOCK / HUMAN_REVIEW / REROUTE)
- Risk score and session risk band
- Per-detector scores sorted by risk descending
- Policy evidence (regulatory citations)
- Async job status

### Tab 2 -- Advanced Inspector
Slow-path LLM analysis. Returns applicable policy, evidence references, detected risk, required controls, and recommendation.

### Tab 3 -- Platform Metrics
Powered by `/v1/metrics/rich` with interactive Plotly charts:
- **KPI row**: Total Requests, Blocked, Modified, Human Review, Allowed, Avg Latency
- **Decision Distribution** -- colour-coded donut chart
- **Risk Distribution** -- histogram (Low / Medium / High)
- **Latency Trend** and **Risk Score Trend** -- line charts
- **Detector Fire Rate** and **Blocked by Policy Rule** tables

### Tab 4 -- Policy Rules
Loaded YAML policy rules in a structured interactive view with action count summary badges and per-rule expanders.

### Tab 5 -- Review & Auto-Tuning

**Self-Governing Threshold Auto-Tuner:**
- Run Tuning Analysis (Dry Run) -- preview NUDGE / ESCALATE / HOLD
- Apply Decisions -- write threshold changes to policy YAML files
- Seed Demo Review History -- populate 25 realistic review records
- Tuning audit changelog from `tuning_history` DB table

**Human Review Queue:**
- All pending HUMAN_REVIEW items with risk score colour coding
- Resolve controls: Reviewer ID, Action selector, Notes

### Tab 6 -- Ask ControlPlane (RAG)
Compliance Q&A assistant with 4 quick-launch chips:
- PII policy for HR
- GDPR Article 22
- Recent BLOCK events
- HIPAA data rules

### Tab 7 -- RLHF Monitor
- **Stats banner**: Total Pairs, Labeled, Unlabeled, Label Coverage, Sampling Rate
- **Generate & Judge Preference Pair On-Demand**: Enter a prompt, select domain, generate dual-model responses and LLM judgment
- **Pairs Explorer**: Searchable/filterable table

---

## Testing & Evaluation Harness

### Full Test Suite

```powershell
pytest -q
# Skip slow tests:
pytest -q -m "not slow"
```

| Test Module | What is Tested |
|---|---|
| `test_golden_path.py` | End-to-end HR salary violation + parallel hot-path |
| `test_governance.py` | PII detection, redaction escalation, injection defence |
| `test_sensitive_data_coverage.py` | 19 detection scenarios (card, CVV, PAN, Aadhaar, passport...) |
| `test_agent_governance.py` | Tool governance: refund tiers, PII deletions, session risk carryover |
| `test_audit_integrity.py` | Hash chains, Merkle proofs, naive & sophisticated tamper detection (14 cases) |
| `test_rag.py` | Policy RAG, chunking, vector store fallback, grounding, Ask ControlPlane (32 cases) |
| `test_session_accumulator.py` | Dual-signal math, entity reconstruction, contamination, Redis fallback (25+ cases) |
| `test_hallucination_bias.py` | Hot-path & async hallucination, counterfactual fairness |
| `test_model_backend.py` | Lazy model loader, cache behaviour |
| `test_rlhf_integration.py` | Preference pair collection, category validation, DPO export, LLM judge |
| `test_llm_client.py` | Groq client injection-hardening, citation verification, extractive fallback |
| `test_groq_llm_client.py` | Ask ControlPlane Groq integration, graceful degradation |
| `test_fast_lane.py` | 250ms timeout fail-open, webhook RETRACT, high-risk correction |
| `test_adversarial.py` | Prompt injection, session evasion, multi-turn attacks |
| `test_multi_tenant.py` | Multi-tenant corpus isolation |
| `test_observability.py` | Prometheus metrics, OTel tracing |
| `test_performance.py` | Sub-50ms hot-path latency benchmarks |
| `test_reliability.py` | Circuit breaker, retry, graceful degradation |
| `test_scalability.py` | Concurrent request handling |
| `test_security.py` | API key auth, rate limiting, security headers |
| `test_prompt_registry.py` | Semver-versioned prompt template registry |
| `test_ml_pipeline.py` | ML training pipeline stubs & data utilities |
| `test_policy_engine.py` | Multi-file YAML policy precedence, priority resolution |
| `test_async_service.py` | Async analytics engine workflow |
| `test_gateway_api.py` | FastAPI HTTP client integration |
| `test_paraphrase_consistency.py` | Paraphrase attack detection consistency |
| `test_accumulator_calibration_integrity.py` | Peak-dilution mathematical guarantee |

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

## Runnable Demonstrations

```powershell
# 1. HR Golden Path -- salary data access -> automatic BLOCK
python scripts/run_golden_path.py

# 2. Agentic Tool-Call Governance -- 8 scenarios: refund tiers, email, delete, session risk
python scripts/run_agent_governance_demo.py

# 3. Tamper-Evident Merkle Audit -- 3 Acts: clean -> naive tamper -> rechaining attack
python scripts/run_audit_integrity_demo.py

# 4. Session Accumulator Demo -- multi-turn risk memory
python demo_session_persistence.py

# 5. RAG Capability Evaluation
python -m rag.evaluation

# 6. RAG Triad CI Evaluation
python scripts/eval_rag_triad.py
```

---

## Deployment

### Option A: Local (PowerShell)

```powershell
$env:GROQ_API_KEY = "your-key"
$env:GROQ_MODEL = "openai/gpt-oss-120b"
$env:CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED = "true"
$env:CONTROLPLANE_API_KEYS = "your-api-key-here"

# Terminal 1 -- Backend
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2 -- Frontend
python -m streamlit run frontend/streamlit_app.py --server.port 8501 --server.headless true
```

Or use the one-command launcher: `.\start.ps1`

### Option B: Docker Compose

```powershell
docker compose up --build
```

The `docker-compose.yml` defines:
- `controlplane-api` (port 8000): FastAPI backend with SQLite volume mount
- `controlplane-ui` (port 8501): Streamlit UI connecting to the API service
- Named volume `controlplane_data` for persistent audit DB
- Health checks on both services (30s interval, 60s start period)

### Option C: Kubernetes

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

The HPA scales from 2 to 10 replicas when CPU exceeds 70%.

---

## Research Grounding & Standards Compliance

| Methodology / Principle | Reference / Standard | System Implementation |
|---|---|---|
| **AI Risk Management Framework** | NIST AI RMF 1.0 (Govern, Map, Measure, Manage) | Multi-layer risk fusion, context enrichment, complete audit trail |
| **AI Management System** | ISO/IEC 42001:2023 | Programmatic policy enforcement and human-in-the-loop review queues |
| **EU AI Act Transparency & High-Risk** | Regulation (EU) 2024/1689 (Articles 50 & Annex III) | Policy RAG corpus, hot-path bias tripwire, counterfactual fairness probe |
| **Faithfulness & NLI Grounding** | RAGAS; BEACON (arXiv:2606.07528); Vectara HHEM | Claim decomposition, DeBERTa cross-encoder NLI, SelfCheckGPT resampling |
| **Counterfactual Fairness** | Kusner et al. (NeurIPS 2017); LangFair (CVS Health) | Token perturbation and counterfactual flip-rate measurement |
| **Tamper-Evident Transparency** | RFC 6962 (Certificate Transparency); Sigstore Rekor | SHA-256 hash chains with domain-separated Merkle tree checkpoints |
| **Bayesian Signal Fusion** | Dempster-Shafer Evidence Theory; AWS Fraud Detector | Noisy-OR probability fusion preventing dilution of high-severity risks |
| **RLHF / DPO Alignment** | Rafailov et al. (arXiv:2305.18290); TRL library | Preference pair collection, LLM-as-judge labelling, per-category DPO fine-tuning |
| **Self-Governing Systems** | Control theory feedback loops; PID control principles | Override-rate-driven threshold auto-tuning with structural safety ceiling |
| **Session Risk** | EWMA + peak-with-decay dual-signal design | Cross-turn risk accumulation catching session evasion patterns |
| **Continuous Learning** | Online learning theory; hard negative mining | Production traffic -> training signal -> calibration without retraining cycles |
| **Context-Aware RBAC** | NIST SP 800-207 Zero Trust; ABAC standards | Two-layer role + department permission resolution with OR merge |

---

## Environment Variables

### Core Backend

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_DB_PATH` | `controlplane.db` | SQLite audit store path |
| `CONTROLPLANE_LOG_LEVEL` | `INFO` | Logging level |
| `CONTROLPLANE_ASYNC_DELAY_MS` | `50` | Artificial async delay (dev/test) |
| `CONTROLPLANE_POLICIES_DIR` | `policies/` | YAML policy files directory |
| `CONTROLPLANE_AUDIT_HASH_KEY` | `local-prototype-not-a-secret` | HMAC key -- **change in production** |
| `CONTROLPLANE_MAX_PROMPT_CHARS` | `12000` | Prompt length limit |
| `CONTROLPLANE_API_KEYS` | *(demo keys)* | Comma-separated valid API keys |
| `CONTROLPLANE_ADMIN_KEYS` | *(demo key)* | Comma-separated admin keys |
| `CONTROLPLANE_CORS_ORIGINS` | `http://localhost:8501` | Comma-separated CORS allowed origins |
| `CONTROLPLANE_RATE_LIMIT_GOVERN` | `60` | Requests/minute for `/v1/govern` and `/v1/chat` |
| `CONTROLPLANE_RATE_LIMIT_DEFAULT` | `120` | Requests/minute for all other endpoints |

### Session Accumulator

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED` | `false` | Enable cross-turn session risk tracking |
| `CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG` | -- | Path to `calibration.json` |
| `CONTROLPLANE_SESSION_STORE` | -- | Redis URL for multi-worker deployments |
| `CONTROLPLANE_SESSION_TTL_HOURS` | `24` | Session TTL before vacuum |

### Scalability

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_ASYNC_QUEUE_SIZE` | `500` | Max buffered background tasks |
| `CONTROLPLANE_METRICS_CACHE_TTL_S` | `60` | Seconds to cache metrics responses |

### Groq LLM

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | -- | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name |
| `GROQ_MAX_TOKENS` | `1024` | Max tokens per LLM completion |
| `GROQ_TEMPERATURE` | `0.3` | LLM temperature |

### Async LLM Judge

| Variable | Default | Purpose |
|---|---|---|
| `CP_JUDGE_PROVIDER` | `mock` | LLM judge provider: `groq`, `openai`, `anthropic`, `mock` |
| `CP_JUDGE_MODEL` | `llama3-8b-8192` | Model for LLM judge |
| `CONTROLPLANE_ASYNC_LLM_RISK_THRESHOLD` | `0.20` | Risk threshold for 100% async LLM analysis |
| `CONTROLPLANE_ASYNC_LLM_SAMPLE_RATE` | `0.05` | Fraction of benign requests sampled for drift monitoring |

### RAG

| Variable | Default | Purpose |
|---|---|---|
| `RAG_EMBEDDING_BACKEND` | `tfidf_lsa` | `tfidf_lsa` or `sentence_transformers` |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model name |
| `RAG_VECTOR_STORE_DIR` | `rag_store/` | ChromaDB / NumPy store location |
| `RAG_TOP_K` | `5` | Retrieved chunks per query |
| `RAG_POLICY_THRESHOLD` | `0.20` | Minimum similarity for policy retrieval |
| `RAG_GROUNDING_THRESHOLD` | `0.28` | Minimum similarity for grounding checks |
| `RAG_HOT_PATH_BUDGET_MS` | `40` | Max latency budget for Policy RAG on hot path |
| `RAG_GENERATION_ENABLED` | `true` | Enable Groq LLM synthesis in Ask ControlPlane |
| `GROQ_API_KEY` | -- | Groq API key for RAG generation |

### RLHF

| Variable | Default | Purpose |
|---|---|---|
| `RLHF_STORAGE_BACKEND` | `json` | `json` (active) or `sqlite` (drop-in ready) |
| `RLHF_SAMPLE_ALL` | `false` | Sample every request (not 1-in-N) |

### Observability

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_JSON_LOGS` | `false` | Structured JSON log output (for ELK/Datadog/GCP) |
| `CONTROLPLANE_OTLP_ENDPOINT` | -- | OTLP endpoint for OTel trace export |
| `CONTROLPLANE_OTEL_SERVICE_NAME` | `controlplane-api` | Service name in traces and logs |

### Frontend

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_API_URL` | `http://127.0.0.1:8000` | Backend URL for Streamlit frontend |
| `CONTROLPLANE_API_KEY` | `demo-key-001` | API key for Streamlit -> backend calls |

---

## CI/CD Pipeline

The GitHub Actions CI (`.github/workflows/ci.yml`) runs on every push and PR to `main`:

```
push/PR to main
  |
  +-- Job 1: Lint (ruff + mypy)
  |   +-- ruff check backend/ tests/ --output-format=github
  |   +-- mypy backend/ (advisory, does not fail pipeline)
  |
  +-- Job 2: Test (requires lint to pass)
  |   +-- Install requirements.txt
  |   +-- pytest tests/ -q --tb=short -m "not slow"
  |   +-- Upload junit XML artifact
  |
  +-- Job 3: Security Audit (parallel with Test)
  |   +-- pip-audit -r requirements.txt
  |   +-- Upload audit report (continue-on-error)
  |
  +-- Job 4: Docker Build (requires Test to pass)
      +-- docker/setup-buildx-action@v3
      +-- Build image (no push on PRs, push on main merges)
```

All test environment variables are set in the workflow:
```yaml
CONTROLPLANE_API_KEYS: "demo-key-001,demo-key-002,test-key,ci-test-key"
RAG_EMBEDDING_BACKEND: sentence_transformers
RAG_RERANK_ENABLED: "true"
RAG_NLI_ENABLED: "true"
RAG_BM25_ENABLED: "true"
```

---

## License & Attribution

Built for the **Accenture Innovation Challenge 2026 (Track 1)**.
All intellectual property and architecture designed for enterprise AI safety, transparency, and governance compliance.

> *"The same governance principle ControlPlane.ai applies to AI systems, it applies to itself."*
