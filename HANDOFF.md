# ControlPlane.ai — Fine-Tuning Work Handoff

> **Purpose of this file.** A self-contained brief so a *different* agent (or a
> fresh session) can pick up this work mid-stream if the current session hits an
> API/usage limit. It captures the task, the environment gotchas, the plan, and
> exactly what is done vs. remaining. Read it top-to-bottom, then continue from
> the **Progress checklist**.

---

## 1. Task & intent

Fine-tune the ControlPlane.ai governance pipeline (this repo). The user asked to
"look into what could be better and do it," informed by a per-detector fine-tuning
plan (reference images): injection → RoBERTa/DeBERTa-v3, safety/toxicity → pretrained
toxicity classifier, fairness → RoBERTa/DeBERTa + HateXplain rationale spans, grounding →
NLI cross-encoder scored per-claim vs retrieved chunks, PII → Presidio; trained with
LoRA/PEFT on a free Colab/Kaggle T4, using group-aware splits, temperature calibration,
and a threshold chosen for a target false-negative rate.

**Hard constraints (do not violate):**
- **Additive & default-OFF.** With no model artifact configured (the default, and the
  state of the local `.venv`), the pipeline must behave **byte-for-byte** as it does today.
- **Do not disturb others' work.** Leave `backend/risk/engine.py`, `backend/decision/engine.py`,
  `backend/policy/*`, `backend/feedback/evaluator.py`, `backend/async_pipeline/consumers.py`,
  the gateway, and `backend/shared/schemas.py` untouched.
- **No new required dependency** for the default install. Heavy ML libs stay in
  `ml/requirements-ml.txt`, imported lazily.

---

## 2. Environment & CRITICAL gotchas

- **OS/shell:** Windows, Git Bash (MINGW64). **Python:** 3.13.3.
- **venv:** `./.venv/Scripts/python.exe`. Has `pydantic`, `pytest`, `numpy`, `pyyaml`,
  `fastapi`. Does **NOT** have `torch`, `transformers`, `scikit-learn`, `datasets`,
  `peft`, `accelerate`. So all new *runtime* code runs on its fallback path here and
  must stay green without those libs. Training is validated out-of-band on Colab/Kaggle.
- **⚠️ Bash tool gotcha #1 — single quotes.** The sandbox wrapper wraps each command in
  single quotes, so a single-quote character *anywhere* in the command (including
  apostrophes in prose and Python `'literals'`) breaks it with "unexpected EOF". **Fix:**
  do file writes with the **Edit tool** (it does not go through the shell — apostrophes are
  fine), or use quote-free `cat`/`printf`. Verify Python with `py_compile <bare-filename>`.
- **⚠️ Bash tool gotcha #2 — length.** Very long single commands get truncated/mangled.
  Keep commands short; write files in modest chunks or via Edit.
- **Repo run cmds:** compile `./.venv/Scripts/python.exe -m py_compile <file>`;
  test `./.venv/Scripts/python.exe -m pytest -q`.

---

## 3. Architecture quick-reference (integration points)

- Detectors self-register via `@register` in `backend/detectors/base.py`; auto-imported by
  `backend/detectors/__init__.py`. Hot-path ones run in `run_hot_path` (asyncio.gather,
  ~50ms budget). Async ones set `hot_path=False`.
- Hot-path detectors that affect the enforced decision: `injection`, `safety`, `pii`,
  `authorization`. Async deep-analysis detectors live in `backend/detectors/async_analytics.py`
  (incl. `GroundingEngineDetector`, currently naive token-overlap).
- Risk fusion (`backend/risk/engine.py`) reads detector scores by name (noisy-OR). Async
  detectors do **not** feed the hot-path risk — safe place for the heavier NLI grounding.
- `GovernanceRequest.retrieved_context: list[str]` and `DetectorResult.evidence: list[str]`
  already exist (schemas.py) — grounding input and rationale-span output need **no** schema change.
- `GPUAdapter` is instantiated in `backend/main.py:55`, surfaced at health (`main.py:71`) and
  `GET /v1/gpu` (`main.py:314`). Its `status()` / `score_with_model()` signatures must stay stable.

---

## 4. The design — "optional learned-detector layer"

A dependency-free runtime **seam** that detectors consult only when an artifact is configured
via env `CONTROLPLANE_MODEL_<TASK>` (e.g. `CONTROLPLANE_MODEL_INJECTION=ml/artifacts/injection-v0/model`).
Every failure path resolves to `None`/no-op. Plus a generalized trainer that produces those
artifacts with calibration + a target-FNR threshold. Full plan mirror:
`C:\Users\Acer\.claude\plans\swift-riding-candy.md`.

---

## 5. Progress checklist

> **STATUS: COMPLETE** — every item below is done. `./.venv/Scripts/python.exe -m pytest -q`
> → **30 passed** (12 new in `tests/test_model_backend.py`); all files `py_compile` clean.

- [x] `ml/__init__.py` — lazy-import package doc.
- [x] `ml/common.py` (247 lines) — `load_jsonl_records`, `grouped_split` (sklearn + pure-Python
      fallback), `fit_temperature`, `apply_temperature`, `confusion_at_threshold`,
      `select_threshold_for_fnr`. Compiles.
- [x] `backend/shared/model_backend.py` — header + `CalibratedClassifier` + `GroundingScorer` +
      module getters (`artifact_dir_for`, `get_detector_model`, `get_grounding_scorer`,
      `consult`, `reset_cache`). Compiles.
- [x] `ml/train_detector.py` — generalized trainer (injection/toxicity/fairness; LoRA optional;
      group split; temperature calibration; threshold-for-FNR; saves model/ + calibration.json +
      evaluation.json; runnable as `python -m ml.train_detector`).
- [x] Detector wiring (guarded, ~2 lines each via `model_backend.consult`):
      `backend/detectors/injection.py`, `backend/detectors/safety.py`, and the async
      `GroundingEngineDetector` in `backend/detectors/async_analytics.py` (→ `get_grounding_scorer`).
- [x] `backend/shared/gpu_adapter.py` — API-stable delegation to `get_detector_model("injection")`.
- [x] `tests/test_model_backend.py` — fallback-returns-None, detector parity, `ml.common` unit tests,
      and a `sys.modules` assertion that importing the seam does not import torch.
- [x] `ml/requirements-ml.txt` — add `peft`, `sentencepiece`.
- [x] `ml/README.md` — document 4-detector mapping (+ PII/Presidio follow-up), trainer usage,
      calibration/threshold, `--lora`, the env-var seam, observe-first rollout warning.
- [x] `ml/artifacts/.gitkeep`.

---

## 6. Env vars (the seam)

| Var | Consumed by | Effect when unset (default) |
|-----|-------------|-----------------------------|
| `CONTROLPLANE_MODEL_INJECTION` | `injection` detector, `gpu_adapter` | regex-only, unchanged |
| `CONTROLPLANE_MODEL_SAFETY`    | `safety` detector | regex-only, unchanged |
| `CONTROLPLANE_MODEL_GROUNDING` | async `GroundingEngineDetector` | token-overlap heuristic |

Each points at a `<artifact>/model` dir with a sibling/nested `calibration.json`.

---

## 7. Verification (run after each step)

```bash
./.venv/Scripts/python.exe -m py_compile backend/shared/model_backend.py ml/common.py ml/train_detector.py
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m pytest tests/test_model_backend.py -q
```

Expectations: full suite stays green; the seam returns `None`/fallback everywhere (no ML deps,
env unset); `injection`/`safety` scores + labels are identical to pre-change when no
`CONTROLPLANE_MODEL_*` is set.

