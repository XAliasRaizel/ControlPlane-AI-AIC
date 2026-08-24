# ControlPlane.ai — Enterprise AI Governance Control Plane

ControlPlane.ai is a governance control plane that sits between AI applications and the models/tools they call. It intercepts every prompt and response, resolves the business context (who, what app, what data, how sensitive), runs risk detectors concurrently, evaluates policies, and decides whether to `ALLOW`, `MODIFY`, `REROUTE`, `HUMAN_REVIEW`, or `BLOCK` the interaction — then records the decision and learns from human corrections over time.

```
OBSERVE -> REASON -> ACT -> LEARN
   ^_________________________|
```

## Architecture

```
AI Applications (chatbots, RAG, agents, copilots, internal tools)
        │  prompt / response / tool-call
        ▼
================================ CONTROLPLANE.AI ================================

  [1] GATEWAY (FastAPI + API Key Auth)
      auth · rate limit · request shaping · assigns request_id
        │
        ▼
  [2] CONTEXT ENRICHMENT
      resolves user role · app criticality · data classification
        │
        ▼
  [3] HOT PATH (Synchronous, ~50ms Latency Budget)
      ┌─ PII detector ────────┐
      ├─ Injection detector   │  all run concurrently via asyncio.gather(...)
      ├─ Authorization check  │  registered via @register plugin decorator
      └─ Safety detector ─────┘
        │
        ▼
  [4] RISK ENGINE
      fuses detector scores + context → overall_risk, confidence
        │
        ▼
  [5] POLICY ENGINE
      evaluates YAML rules (Application > Department > Global precedence)
        │
        ▼
  [6] DECISION ENGINE
      ALLOW · MODIFY · REROUTE · BLOCK · HUMAN_REVIEW ──┐
        │                                               │
        │                                               ▼
        │                                    [7] HUMAN REVIEW QUEUE
        │                                        approve / reject / modify
        │◀───────────────────────────────────────────────┘
        ▼
  response (possibly modified/sanitized) returned to application
        │
        │  fire-and-forget event
        ▼
  [8] ASYNC PATH (Non-blocking Background Pipeline)
      Safety · Privacy · Fairness · Grounding · Cost · Performance
        │
        ▼
  [9] AUDIT LOG (Privacy-Preserving Audit Store)
      HMAC fingerprinting · allow-listed context · SQLite / PostgreSQL
        │
        ├──────────────────────────────┐
        ▼                              ▼
  [10] DATA LAYER                [11] FEEDBACK & LEARNING
      Postgres · Vector DB           human overrides → threshold &
      · event/metrics store          policy optimization

===================================================================================
```

## Repository Structure

```
controlplane/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── backend/
│   ├── main.py                      # FastAPI entrypoint
│   ├── shared/
│   │   ├── schemas.py               # Section 6 canonical contracts (single source of truth)
│   │   ├── config.py                # Environment configuration
│   │   └── gpu_adapter.py           # Optional hardware inference seam
│   ├── gateway/
│   │   ├── auth.py                  # API key validation dependency
│   │   └── context_enrichment.py    # Resolves role, criticality, data classification
│   ├── detectors/
│   │   ├── base.py                  # BaseDetector ABC + self-registering registry + hot path runner
│   │   ├── pii.py                   # PII scanner (regex + pattern detection)
│   │   ├── injection.py             # Prompt injection signature scanner
│   │   ├── authorization.py         # Deterministic RBAC access check
│   │   └── safety.py                # Harmful content rules
│   ├── risk/
│   │   └── engine.py                # Multi-dimensional signal fusion + confidence scoring
│   ├── policy/
│   │   ├── engine.py                # Multi-scope policy evaluator (App > Dept > Global)
│   │   └── loader.py                # Dynamic YAML loader & validator with hot reload
│   ├── decision/
│   │   └── engine.py                # Decision resolution & response redaction
│   ├── review/
│   │   └── queue.py                 # Human review queue & phase-1 fallback
│   ├── async_pipeline/
│   │   ├── publisher.py             # Fire-and-forget async dispatcher
│   │   ├── worker.py                # Background task executor
│   │   └── consumers.py             # Grounding, fairness, privacy, safety, cost engines
│   ├── audit/
│   │   └── store.py                 # SQLite database & privacy-safe HMAC audit store
│   └── feedback/
│       └── evaluator.py             # Labeled error classification (FPR/FNR)
│
├── policies/
│   ├── global.yaml                  # Universal fallthrough governance rules
│   ├── hr.yaml                      # HR-scoped policy (PII & authorization rules)
│   ├── finance.yaml                 # Finance-scoped policy (loan decision rules)
│   └── support.yaml                 # Support bot policy (injection prevention & redaction)
│
├── frontend/
│   └── streamlit_app.py             # Interactive governance dashboard & audit viewer
│
├── scripts/
│   └── run_golden_path.py           # Runnable Section 14 end-to-end demo scenario
│
└── tests/
    ├── test_golden_path.py          # Automated verification of Section 14 scenario
    ├── test_governance.py           # Unit and component governance tests
    ├── test_gateway_api.py          # FastAPI HTTP client integration tests
    ├── test_policy_engine.py        # Multi-file policy engine precedence tests
    └── test_async_service.py        # Async analysis engine pipeline tests
```

## Quick Start

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
pytest -v
```

### 3. Run the Section 14 Golden Path Demo
```powershell
python scripts/run_golden_path.py
```

### 4. Start the FastAPI Gateway
```powershell
uvicorn backend.main:app --reload --port 8000
```
Interactive OpenAPI documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 5. Launch the Streamlit Dashboard
```powershell
streamlit run frontend/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) to interact with the governance UI.

### 6. Docker Compose
```powershell
docker compose up --build
```

## Golden Path Demo Scenario

An employee asks the HR Copilot:
> *"Give me Rahul's salary and personal phone number."*

### Execution Trace:
- **Gateway**: Authenticated caller (`user=aryan`, `role=employee`, `app=hr-copilot`)
- **Context Enrichment**: `department=HR`, `data_classification=HIGH`, `criticality=high`
- **Hot Path Detectors (Parallel via `asyncio.gather`)**:
  - `pii` → score `0.85`, label `PII_DETECTED`
  - `authorization` → score `1.00`, label `DENIED` (caller unauthorized for salary data)
  - `injection` → score `0.00`, label `CLEAN`
  - `safety` → score `0.00`, label `CLEAN`
  - **Latency**: `< 5 ms` (well within the 50 ms budget)
- **Risk Engine**: `overall_risk=0.46`, `confidence=0.95`
- **Policy Engine**: Matches `hr.yaml` rule `hr-pii-unauthorized`
- **Decision Engine**: `BLOCK`, Reason: `Unauthorized access to PII-classified data`
- **Audit Store**: Privacy-safe `AuditRecord` stored with HMAC prompt fingerprint (no raw prompt/PII saved)
- **Async Path**: Background analytics (fairness, grounding, cost, safety) scheduled non-blockingly
