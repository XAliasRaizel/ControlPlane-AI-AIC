# ControlPlane.ai — Enterprise AI Governance Control Plane

> **Accenture Innovation Challenge 2026 · Track 1**  
> *A Production-Grade, Multi-Layered Governance Gateway for Generative AI & Autonomous Agent Systems.*

---

ControlPlane.ai sits between AI applications (LLM chat, copilots, RAG pipelines, autonomous agents) and underlying foundation models/tools. It intercepts prompts, candidate responses, and agent actions, resolves rich business context, executes parallel hot-path detectors (<5ms), retrieves policy/regulatory evidence via **Policy RAG**, evaluates deterministic multi-scope policies, and renders enforceable decisions (**`ALLOW`**, **`MODIFY`**, **`REROUTE`**, **`HUMAN_REVIEW`**, **`BLOCK`**).

Decisions are recorded into a **cryptographically tamper-evident Merkle audit ledger**, enriched with **async deep analytics** (NLI grounding, counterfactual fairness, LLM-as-judge), and queryable via an interactive **Ask ControlPlane** compliance assistant.

```
                  ┌─────────────────────────────────────────────────┐
                  │          OBSERVE -> REASON -> ACT -> LEARN      │
                  │             ^_________________________|         │
                  └─────────────────────────────────────────────────┘
```

---

## 📑 Table of Contents

1. [Key Capabilities](#-key-capabilities)
2. [System Architecture](#-system-architecture)
3. [Repository Layout](#-repository-layout)
4. [Quick Start & Setup](#-quick-start--setup)
5. [Feature Deep Dives](#-feature-deep-dives)
   - [Policy RAG & Ask ControlPlane](#1-policy-rag--ask-controlplane-subsystem)
   - [Agentic Tool-Call Governance](#2-agentic-tool-call-governance-agent)
   - [Tamper-Evident Merkle Audit Ledger](#3-tamper-evident-merkle-audit-ledger)
   - [Two-Layer Hallucination & Grounding Detection](#4-two-layer-hallucination--grounding-detection)
   - [Two-Layer Bias & Fairness Detection](#5-two-layer-bias--fairness-detection)
   - [Unified Sensitive Data Protection & Safety Net](#6-unified-sensitive-data-protection--safety-net)
6. [Testing & Evaluation Harness](#-testing--evaluation-harness)
7. [Runnable Demonstrations](#-runnable-demonstrations)
8. [Research Grounding & Standards Compliance](#-research-grounding--standards-compliance)
9. [Environment Variables](#-environment-variables)

---

## 🌟 Key Capabilities

| Layer / Feature | Description | Primary Location |
|---|---|---|
| **Hot-Path Gateway** | FastAPI ingress with API key authentication, rate-limiting, context enrichment, and sub-5ms latency budget. | `backend/gateway/`, `backend/main.py` |
| **Unified Sensitive Data Scanner** | Single-source-of-truth scanner covering financial, government IDs, HR records, medical history, credentials, and fail-cautious safety nets. | `backend/shared/sensitive_terms.py`, `backend/detectors/pii.py` |
| **RBAC Authorization Check** | Deterministic access control verifying role-based entitlement against classified resources (salary, bank, medical). | `backend/detectors/authorization.py` |
| **Prompt Injection & Jailbreak Defense** | Regex signature detection for instruction overrides, prompt extraction, DAN attacks, and role manipulation. | `backend/detectors/injection.py` |
| **Policy RAG (Hot-Path Explainer)** | Bounded (<40ms budget, ~2ms warm) semantic retrieval over GDPR, EU AI Act, HIPAA, and internal YAML policies to explain decisions. | `rag/policy/policy_rag.py` |
| **Ask ControlPlane (RAG Assistant)** | Interactive compliance intelligence assistant querying indexed policy corpora and SQLite audit records with verifiable citations. | `rag/ask_controlplane/` |
| **Agentic Tool-Call Governance** | Pre-execution action interceptor (`ToolGovernor`) scoring risk on tool calls (`issue_refund`, `send_email`, `delete_record`) with session risk carryover. | `backend/agents/` |
| **Two-Layer Hallucination Detection** | Hot-path claim gate + async deep grounding (NLI cross-encoder entailment, LLM-as-judge, SelfCheckGPT consistency, and KB retrieval). | `backend/detectors/hallucination.py`, `backend/async_engines/grounding.py` |
| **Two-Layer Bias & Fairness Engine** | Hot-path causal tripwire + async counterfactual attribute perturbation and multi-category LLM-judge rubric. | `backend/detectors/bias.py`, `backend/async_engines/fairness.py` |
| **Tamper-Evident Merkle Audit Log** | Append-only SHA-256 hash chains + periodic RFC 6962 Merkle tree checkpoints anchored to external HMAC signatures. | `backend/audit_integrity/` |
| **Noisy-OR Risk Engine** | Bayesian evidence fusion combining independent detector probabilities with context escalation. | `backend/risk/engine.py` |
| **Human Review & Feedback Loop** | SQLite-backed review queue for ambiguous/high-risk requests + FPR/FNR feedback classification. | `backend/review/`, `backend/feedback/` |

---

## 🏛️ System Architecture

```
AI Applications (chatbots, RAG, agents, copilots, internal tools)
        │  prompt / candidate response / proposed tool-call
        ▼
================================ CONTROLPLANE.AI ================================

  [1] GATEWAY (FastAPI + API Key Auth)
      auth · rate limit · request envelope shaping · assigns UUID request_id
        │
        ▼
  [2] CONTEXT ENRICHMENT
      resolves user role · department · app criticality · data classification · RBAC
        │
        ▼
  [3] HOT PATH DETECTORS (Synchronous, asyncio.gather, <5ms)
      ┌─ PII & Sensitive Terms Scanner (6 categories + fail-cautious safety net)
      ├─ Prompt Injection & Jailbreak (instruction overrides, DAN signatures)
      ├─ Authorization & RBAC Check (resource entitlements)
      ├─ Safety & Harmful Content Filter (violence, hacking, exploit patterns)
      ├─ Hallucination Fast Gate (claim-level ungrounded fact verification)
      └─ Bias Fast Gate (protected-attribute-as-decision-reason compliance tripwire)
        │
        ▼
  [4] RISK ENGINE (Noisy-OR Evidence Fusion)
      P(risk) = 1 - ∏(1 - p_i) + context multiplier → overall_risk & confidence
        │
        ▼
  [5] POLICY ENGINE & POLICY RAG
      evaluates YAML rules (Application > Department > Global precedence)
      └─► Policy RAG retrieves matching regulatory/internal policy citations (~2ms)
        │
        ▼
  [6] DECISION ENGINE & SANITIZATION
      ALLOW · MODIFY (Redaction) · REROUTE · BLOCK · HUMAN_REVIEW ──┐
        │                                                           │
        │                                                           ▼
        │                                                [7] HUMAN REVIEW QUEUE
        │                                                    approve / reject / modify
        │◄──────────────────────────────────────────────────────────┘
        ▼
  Sanitized Response / Decision returned to Application
        │
        │  fire-and-forget background task
        ▼
  [8] ASYNC ANALYTICS PIPELINE
      ├─ Grounding RAG & Deep NLI Engine (DeBERTa cross-encoder + SelfCheckGPT)
      ├─ Deep Fairness Engine (Counterfactual probe + LLM judge rubric)
      └─ Performance, Cost, Privacy, Safety, Business compliance engines
        │
        ▼
  [9] TAMPER-EVIDENT AUDIT LEDGER
      privacy-preserving HMAC hashing · SHA-256 hash chains · Merkle checkpoints
        │
        ├──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼                              ▼
  [10] PERSISTENT AUDIT          [11] FEEDBACK & LEARNING      [12] ASK CONTROLPLANE
      SQLite / PostgreSQL            human overrides →              hybrid RAG over
      audit database                 FPR/FNR calibration            policies & audit log

===================================================================================

  [13] AGENTIC ACTION GOVERNANCE (/agent/act)
      Agent proposes tool call ──► ToolGovernor intercepts ──► Risk & Policy scoring
      ──► ALLOW (executes tool) | HUMAN_REVIEW (held in queue) | BLOCK (aborts)
```

---

## 📁 Repository Layout

```
controlplane_ai/
├── Dockerfile                           # Production container spec
├── docker-compose.yml                   # Multi-service stack (gateway + UI)
├── requirements.txt                     # Core dependencies
├── README.md                            # Complete system documentation
│
├── backend/
│   ├── main.py                          # FastAPI ingress, lifespan warm-up, and endpoints
│   ├── shared/
│   │   ├── schemas.py                   # Canonical Pydantic contracts (single source of truth)
│   │   ├── sensitive_terms.py           # Unified taxonomy: financial, IDs, HR, medical, auth
│   │   ├── config.py                    # Environment settings loader
│   │   ├── gpu_adapter.py               # Hardware inference interface
│   │   └── llm_simulator.py             # Synthetic LLM response generator
│   ├── gateway/
│   │   ├── auth.py                      # API key authentication dependency
│   │   └── context_enrichment.py        # RBAC and sensitivity resolver
│   ├── detectors/
│   │   ├── base.py                      # BaseDetector ABC + self-registration registry
│   │   ├── pii.py                       # Sensitive term & regex value scanner
│   │   ├── injection.py                 # Jailbreak & instruction override scanner
│   │   ├── authorization.py             # Deterministic RBAC access check
│   │   ├── safety.py                    # Harmful content & exploit scanner
│   │   ├── hallucination.py             # Hot-path ungrounded-claim gate
│   │   ├── bias.py                      # Hot-path protected attribute causal detector
│   │   └── async_analytics.py           # 7 background analytics engine detectors
│   ├── async_engines/
│   │   ├── grounding.py                 # Deep NLI entailment + LLM-judge + SelfCheckGPT
│   │   └── fairness.py                  # Counterfactual probe + LLM bias rubric
│   ├── utils/
│   │   ├── claims.py                    # Lightweight claim decomposition utility
│   │   └── llm_judge.py                 # Provider-agnostic AI-as-judge (OpenAI/Anthropic/Mock)
│   ├── risk/
│   │   └── engine.py                    # Noisy-OR Bayesian risk fusion & severity floor
│   ├── policy/
│   │   ├── engine.py                    # Multi-scope hierarchical policy evaluator
│   │   └── loader.py                    # Hot-reloading YAML policy loader & validator
│   ├── decision/
│   │   └── engine.py                    # Decision resolution & pattern-based redaction
│   ├── review/
│   │   └── queue.py                     # SQLite human review queue
│   ├── async_pipeline/
│   │   ├── publisher.py                 # Fire-and-forget async dispatcher
│   │   ├── worker.py                    # Background job executor
│   │   └── consumers.py                 # Analytics orchestrator
│   ├── agents/                          # Agentic Tool-Call Governance
│   │   ├── models.py                    # ToolCallContext, GovernanceDecision dataclasses
│   │   ├── tools.py                     # Tool execution layer & registry
│   │   ├── risk.py                      # Tool-call risk scoring
│   │   ├── policy.py                    # YAML-driven agent policy evaluator
│   │   ├── queue.py                     # Pending action human review queue
│   │   ├── governance.py                # ToolGovernor: intercept -> score -> decide -> execute
│   │   └── router.py                    # FastAPI endpoints (/agent/act, /agent/pending, ...)
│   ├── audit/
│   │   └── store.py                     # SQLite database & privacy-safe HMAC audit store
│   ├── audit_integrity/                 # Cryptographic Tamper-Evident Ledger
│   │   ├── models.py                    # AuditRecord, Checkpoint, VerificationResult
│   │   ├── hashing.py                   # Canonical JSON + SHA-256 + HMAC utilities
│   │   ├── merkle.py                    # RFC 6962 Merkle tree with inclusion proofs
│   │   ├── backends.py                  # SQLite record store & append-only anchor file
│   │   ├── ledger.py                    # TamperEvidentAuditLedger (append + seal)
│   │   └── verifier.py                  # Independent chain & checkpoint verification
│   └── feedback/
│       └── evaluator.py                 # FPR/FNR labeled error evaluator
│
├── rag/                                 # Retrieval-Augmented Generation Subsystem
│   ├── config.py                        # RAG settings & latency budgets
│   ├── schemas.py                       # Chunk, Query, Document, RetrievalResult schemas
│   ├── embeddings.py                    # Local TF-IDF/LSA + Sentence-Transformers embedder
│   ├── vector_store.py                  # ChromaDB + zero-dependency NumPy/JSON vector store
│   ├── chunking.py                      # Paragraph-aware text chunking
│   ├── retriever.py                     # Hybrid vector + lexical retriever
│   ├── evaluation.py                    # 12-case end-to-end RAG evaluation harness
│   ├── corpus/
│   │   ├── regulatory/                  # GDPR, EU AI Act, HIPAA knowledge bases
│   │   └── internal_kb/                 # Company leave, IT, security, expense policies
│   ├── ingestion/
│   │   ├── ingest.py                    # Corpus ingestion & index builder
│   │   ├── document_loader.py           # Text & Markdown loader
│   │   ├── policy_loader.py             # Programmatic YAML policy-to-prose loader
│   │   └── audit_loader.py              # Privacy-safe audit record indexer
│   ├── policy/
│   │   └── policy_rag.py                # Hot-path Policy RAG explainer (~2ms)
│   ├── grounding/
│   │   ├── claim_extractor.py           # Sentence-level checkable claim extractor
│   │   ├── entailment.py                # Lexical & number-penalty entailment checker
│   │   └── grounding_checker.py         # Grounding verification orchestrator
│   └── ask_controlplane/
│       ├── retrieval.py                 # Hybrid policy + audit retrieval
│       └── chat.py                      # Q&A synthesizer with citations
│
├── policies/
│   ├── global.yaml                      # Universal fallthrough governance rules
│   ├── hr.yaml                          # HR-scoped policy (PII & authorization rules)
│   ├── finance.yaml                     # Finance-scoped policy (loan decision rules)
│   ├── support.yaml                     # Support bot policy (injection & redaction)
│   ├── agent_tools.yaml                 # 7 rules governing refunds, emails, record deletions
│   └── hallucination_bias_rules.yaml    # Policy rules for hallucination and bias thresholds
│
├── frontend/
│   └── streamlit_app.py                 # Interactive Streamlit UI (Chatbot, Metrics, Inspector, RAG)
│
├── scripts/
│   ├── run_golden_path.py               # Section 14 HR golden-path demo
│   ├── run_agent_governance_demo.py     # 8-scenario agent tool-call governance demo
│   └── run_audit_integrity_demo.py      # 3-act tamper-evident ledger demonstration
│
└── tests/
    ├── test_golden_path.py              # End-to-end golden path tests (7)
    ├── test_governance.py               # Gateway & detector tests (5)
    ├── test_gateway_api.py              # HTTP integration tests (1)
    ├── test_policy_engine.py            # Hierarchical policy precedence tests (3)
    ├── test_async_service.py            # Async analytics pipeline tests (2)
    ├── test_agent_governance.py         # Agent tool-call governance tests (13)
    ├── test_audit_integrity.py          # Cryptographic Merkle audit tests (14)
    ├── test_hallucination_bias.py       # Hallucination & bias engine tests (6)
    ├── test_sensitive_data_coverage.py  # Comprehensive sensitive data coverage tests (19)
    └── test_rag.py                      # Policy RAG, Grounding RAG, Ask ControlPlane tests (32)
```

---

## 🚀 Quick Start & Setup

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt
```

### 2. Build RAG Indices
```powershell
# Ingest regulatory corpora, internal KB, and YAML policies into vector store
python -m rag.ingestion.ingest
```

### 3. Run Full Test Suite (102 Tests)
```powershell
pytest -v
```
*Expected result:* **102 passed** across all 10 test suites.

### 4. Start the Application

**Option A: Run Locally**
```powershell
# Terminal 1: Start FastAPI Gateway (Port 8000)
python -m uvicorn backend.main:app --port 8000

# Terminal 2: Start Streamlit Dashboard (Port 8501)
python -m streamlit run frontend/streamlit_app.py --server.headless true
```

**Option B: Docker Compose**
```powershell
docker compose up --build
```

- **Interactive UI**: [http://localhost:8501](http://localhost:8501)
- **API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🔍 Feature Deep Dives

### 1. Policy RAG & Ask ControlPlane Subsystem
- **Policy RAG (`rag/policy/policy_rag.py`)**: Executes in parallel with decision evaluation. It constructs a domain-specific query and retrieves relevant clauses from GDPR, EU AI Act, HIPAA, or company policies in **~2ms warm**. It attaches evidence to the response without blocking or altering the decision.
- **Ask ControlPlane (`rag/ask_controlplane/`)**: An interactive compliance copilot in Streamlit (Tab 6) and via `POST /v1/ask-controlplane`. It uses hybrid retrieval over both policy evidence and SQLite audit logs to answer auditor questions with exact citations, refusing to hallucinate when evidence is insufficient.

### 2. Agentic Tool-Call Governance (`/agent/*`)
Autonomous agents introduce compounding risk when executing real-world actions:
- **`ToolGovernor.invoke(...)`**: Intercepts actions before execution.
- **Risk Scoring**: Evaluates action sensitivity, parameters (e.g., refund amounts), user roles, and **session risk carryover**.
- **Declarative Policies (`policies/agent_tools.yaml`)**:
  - Small refunds (<$50) by support agents → `ALLOW`.
  - Mid refunds ($50–$500) → `HUMAN_REVIEW`.
  - High refunds (>$500) → `BLOCK`.
  - Deleting records with PII without admin role → `BLOCK`.
  - Elevated session risk carries over to escalate otherwise clean calls.

### 3. Tamper-Evident Merkle Audit Ledger
- **SHA-256 Hash Chain**: Every record's hash depends on the cryptographic hash of the preceding record ($H_i = \text{SHA-256}(H_{i-1} \parallel R_i)$).
- **RFC 6962 Merkle Tree Checkpoints**: Every $N$ records (default: 10), a Merkle tree is computed over the batch. The root is signed with HMAC and written to an independent, append-only anchor store.
- **Independent Verifier**: Detects naive data tampering (modifying record content) as well as sophisticated attacks (re-computing the hash chain) because the re-computed root will mismatch the externally signed checkpoint.

### 4. Two-Layer Hallucination & Grounding Detection
- **Hot-Path (`detectors/hallucination.py`)**: Extracts specific checkable claims (dates, numbers, entities) and verifies them against the request's `retrieved_context` in <0.5ms.
- **Deep Async (`async_engines/grounding.py`)**: Ensembles **NLI cross-encoder entailment** (`cross-encoder/nli-deberta-v3-base`), **LLM-as-judge**, and **SelfCheckGPT self-consistency resampling**.
- **Grounding RAG (`rag/grounding/`)**: Lexical entailment checker scoring claims against the company internal knowledge base.

### 5. Two-Layer Bias & Fairness Detection
- **Hot-Path (`detectors/bias.py`)**: Fast regex proximity detector catching protected attributes (age, gender, race, religion, disability) cited as explicit reasons in decision contexts (ECOA / Title VII / EU AI Act compliance).
- **Deep Async (`async_engines/fairness.py`)**: Generates counterfactual name/pronoun swaps, re-evaluates the prompt, and computes the **counterfactual flip rate** alongside an LLM-judge bias rubric.

### 6. Unified Sensitive Data Protection & Safety Net
- **Single Source of Truth (`backend/shared/sensitive_terms.py`)**: Shared taxonomy across `pii.py` and `authorization.py` covering Financial (cards, CVV, UPI, IFSC), Government IDs (PAN, Aadhaar, Passport, DL, SSN), HR records, Medical history, and Account Credentials.
- **Fail-Cautious Safety Net**: Requests naming third parties with detail-seeking language ("give me / show me details about Rahul") automatically score elevated risk even for terms not on explicit keyword lists.

---

## 🧪 Testing & Evaluation Harness

### Full Automated Test Suite (102 Tests)

| Test Module | Test Count | Description |
|---|:---:|---|
| `test_golden_path.py` | 7 | End-to-end HR salary violation scenario, parallel hot-path, risk fusion |
| `test_sensitive_data_coverage.py` | 19 | Credit card, CVV, PAN, Aadhaar, passport, medical, injection rephrasings |
| `test_agent_governance.py` | 13 | Tool governance: refund tiers, PII deletions, session risk carryover |
| `test_audit_integrity.py` | 14 | Hash chains, Merkle proofs, naive & sophisticated tamper detection |
| `test_rag.py` | 32 | Policy RAG, chunking, vector store fallback, grounding, Ask ControlPlane |
| `test_hallucination_bias.py` | 6 | Hot-path & async hallucination and counterfactual fairness engines |
| `test_governance.py` | 5 | PII detection, redaction escalation, injection defense, audit privacy |
| `test_policy_engine.py` | 3 | Multi-file YAML policy precedence, priority resolution |
| `test_async_service.py` | 2 | Async analytics engine workflow and background consumers |
| `test_gateway_api.py` | 1 | FastAPI HTTP client integration & redacted audit verification |
| **Total** | **102** | **100% Passing (0 failures, 0 warnings)** |

### RAG Evaluation Harness (`python -m rag.evaluation`)
Evaluates precision across all three RAG subsystems:
```
Policy RAG (retrieval relevance):        4/4 passed (100%)
Grounding RAG (status accuracy):         4/4 passed (100%)
Ask ControlPlane (answer / refusal):     4/4 passed (100%)
Total RAG Eval Score:                   12/12 passed (100%)
```

---

## 🎬 Runnable Demonstrations

Execute the interactive scripts to demonstrate each subsystem in action:

```powershell
# 1. Section 14 Golden Path Demo (HR salary + PII unauthorized access -> BLOCK)
python scripts/run_golden_path.py

# 2. Agentic Tool-Call Governance Demo (8 scenarios: refund tiers, email, delete, session risk)
python scripts/run_agent_governance_demo.py

# 3. Tamper-Evident Merkle Audit Demo (3 Acts: clean ledger, naive tamper, rechaining attack)
python scripts/run_audit_integrity_demo.py

# 4. RAG Capability Evaluation Harness
python -m rag.evaluation
```

---

## 📚 Research Grounding & Standards Compliance

ControlPlane.ai is engineered against established industry frameworks and academic literature:

| Methodology / Principle | Reference / Standard | System Implementation |
|---|---|---|
| **AI Risk Management Framework** | NIST AI RMF 1.0 (Govern, Map, Measure, Manage) | Multi-layer risk fusion, context enrichment, and audit trail |
| **AI Management System** | ISO/IEC 42001:2023 | Programmatic policy enforcement and human-in-the-loop review queues |
| **EU AI Act Transparency & High-Risk** | Regulation (EU) 2024/1689 (Articles 50 & Annex III) | Policy RAG corpus, hot-path bias tripwire, and counterfactual fairness probe |
| **Faithfulness & NLI Grounding** | RAGAS & Vectara HHEM; BEACON (arXiv:2606.07528) | Claim decomposition, cross-encoder NLI entailment, and SelfCheckGPT |
| **Counterfactual Fairness** | Kusner et al. (NeurIPS 2017); LangFair (CVS Health) | Token perturbation and counterfactual flip-rate measurement |
| **Tamper-Evident Transparency** | RFC 6962 (Certificate Transparency); Sigstore Rekor | SHA-256 hash chains with domain-separated Merkle tree checkpoints |
| **Bayesian Signal Fusion** | Dempster-Shafer Evidence Theory; AWS Fraud Detector | Noisy-OR probability fusion preventing dilution of high-severity risks |

---

## ⚙️ Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CONTROLPLANE_API_URL` | `http://127.0.0.1:8000` | Gateway URL for Streamlit frontend |
| `CP_API_KEY` | `demo-key-001` | Gateway API key authentication |
| `CP_JUDGE_PROVIDER` | `mock` | LLM-as-judge backend (`mock`, `openai`, `anthropic`) |
| `CP_JUDGE_MODEL` | Provider default | Specific model name for LLM judge |
| `OPENAI_API_KEY` | — | OpenAI API key (when provider is `openai`) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (when provider is `anthropic`) |
| `RAG_EMBEDDING_BACKEND` | `tfidf_lsa` | RAG embedding engine (`tfidf_lsa`, `sentence_transformers`) |
| `RAG_HOT_PATH_BUDGET_MS` | `40.0` | Maximum latency budget for Policy RAG on hot path |
| `CONTROLPLANE_DB_PATH` | `controlplane.db` | SQLite audit store file location |

---

## 📄 License & Attribution

Built for the **Accenture Innovation Challenge 2026 (Track 1)**.  
All intellectual property and architecture designed for enterprise AI safety, transparency, and governance compliance.
