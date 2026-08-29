# Quickstart: Agentic Tool-Call Governance

## Drop-in

Copy these into your repo at the **exact same relative paths**:

```
backend/app/agents/__init__.py
backend/app/agents/models.py
backend/app/agents/tools.py
backend/app/agents/risk.py
backend/app/agents/policy.py
backend/app/agents/queue.py
backend/app/agents/governance.py
backend/app/agents/router.py
policies/agent_tools.yaml
scripts/run_agent_governance_demo.py
tests/test_agent_governance.py
```

If `backend/app/__init__.py` doesn't already exist in your repo, add an empty one — it's what makes `app` importable as a package (yours almost certainly already has this, since the rest of `backend/app/*` already works as a package today).

## Run it (no new dependencies — stdlib + PyYAML only, which you already have)

```bash
python3 scripts/run_agent_governance_demo.py
python3 -m unittest tests.test_agent_governance -v
# or, since it's plain unittest, your existing pytest suite will pick it up too:
pytest tests/test_agent_governance.py -v
```

Both were run and passed (13/13 tests, 8/8 demo scenarios matching their expected outcome label) before this was handed to you.

## Wire it into the API (needs fastapi + pydantic, which your project already has — this part wasn't testable in the sandbox this was built in)

In `backend/app/main.py`:

```python
from app.agents.router import router as agent_governance_router
app.include_router(agent_governance_router)
```

Then:

```bash
curl -X POST http://localhost:8000/agent/act \
  -H "Content-Type: application/json" \
  -d '{"tool": "issue_refund", "args": {"order_id": "o1", "amount": 1500}, "role": "support_agent", "application": "support-agent", "session_id": "demo-1"}'
```

should come back `HUMAN_REVIEW`, and:

```bash
curl -X POST http://localhost:8000/agent/pending/<pending_id_from_above>/resolve \
  -H "Content-Type: application/json" \
  -d '{"approve": true, "resolved_by": "manager_priya"}'
```

should then execute it.

## If you want to hand this to an agentic coding tool (Antigravity, Claude Code, etc.)

Point it at `docs/agent_tool_governance_spec.md` — that's written as a standalone task brief with the design rationale, the exact files, the integration TODOs (each one marked with a `[ ]`), and acceptance criteria, so an agent (or a teammate) can pick up from there without needing this whole conversation as context.
