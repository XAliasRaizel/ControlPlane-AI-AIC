# rlhf/ — RLHF Preference Data Collection & DPO Fine-Tuning Module

This self-contained package provides everything needed to collect human and
LLM-judged preference pairs for ControlPlane.ai's governance models, and to
run Direct Preference Optimization (DPO) fine-tuning per category.

---

## Folder Structure

```
rlhf/
├── __init__.py              # Re-exports Category, STORAGE_BACKEND, PreferencePair
├── README.md                # This file
├── config.py                # Category enum, storage backend selector, rate-limit caps, paths
├── schema.py                # PreferencePair pydantic model (single shared record shape)
│
├── generators/
│   ├── __init__.py
│   ├── api_vs_api.py        # Concurrent dual API-model generation → PreferencePair
│   └── local_vs_local.py    # Concurrent dual local-model generation → PreferencePair
│
├── judges/
│   ├── __init__.py
│   ├── llm_judge.py         # LLM-as-Judge with position-bias control (swap ordering)
│   └── human_judge.py       # CLI human labelling (injectable input_fn for UI use)
│
├── storage/
│   ├── __init__.py
│   ├── json_store.py        # ✅ ACTIVE: append-only JSONL with label reconciliation
│   ├── sqlite_store.py      # ⚠️  NOT ACTIVE YET: drop-in SQLite replacement
│   └── categorize.py        # assign_category() — the single category-validation gatekeeper
│
├── export/
│   ├── __init__.py
│   ├── filters.py           # filter_pairs() — drop unlabelled/tie/error/duplicate pairs
│   └── dpo_export.py        # export_for_dpo() → timestamped DPO JSONL file
│
├── training/
│   ├── __init__.py
│   ├── dataset.py           # load_dpo_dataset() → HuggingFace Dataset
│   ├── dpo_config.py        # get_dpo_config(category) → DPORunConfig with LoRA defaults
│   ├── train.py             # run_dpo_training() + CLI entry point
│   └── evaluate.py          # compute_reward_margin, human_prompt_consistency_check,
│                            #   run_full_evaluation, EvalResult
│
└── data/
    ├── raw/
    │   └── pairs.jsonl          # Append-only log — actively used
    ├── db/
    │   └── preferences.db       # Placeholder only — not written until SQLite is activated
    ├── exports/
    │   └── .gitkeep             # DPO JSONL exports land here
    └── checkpoints/
        └── .gitkeep             # LoRA adapter checkpoints land here
```

---

## End-to-End Flow

```
Sampled Prompt
     │
     ▼
[generators/] ── asyncio.gather ──► Two model calls (API or local)
     │                               Error-wrapped; daily cap enforced
     ▼
PreferencePair (unlabelled)
     │
     ├──► storage/json_store.write_pair()   ← category validated by assign_category()
     │
     ▼
[judges/] ── human_judge or llm_judge ──► chosen = "a" | "b" | "tie"
     │                                     LLM judge swaps response order to cancel
     │                                     position bias; ties on disagreement
     ▼
storage/json_store.update_label()  (appends label_update record)
     │
     ▼
[export/dpo_export.export_for_dpo()]
     │  reads all pairs via active backend
     │  applies filter_pairs() (drop unlabelled / tie / error / duplicate)
     │  reshapes to {prompt, chosen, rejected}
     ▼
data/exports/dpo_<CATEGORY>_<TIMESTAMP>.jsonl
     │
     ▼
[training/train.py — run_dpo_training()]
     │  loads DPORunConfig for category (beta, lr, LoRA r/alpha/modules …)
     │  loads base model + applies LoRA via peft
     │  loads dataset via training/dataset.py
     │  instantiates trl.DPOTrainer → .train()
     ▼
data/checkpoints/<CATEGORY>_<TIMESTAMP>/adapter/
     │
     ▼
[training/evaluate.py — run_full_evaluation()]
     │  Signal 1: avg reward margin across labeled_pairs (log-prob, no LLM calls)
     │  Signal 2: human-prompt consistency check (LLM judge, position-bias controlled)
     │            → flags prompts where fine-tuned model not consistently preferred
     ▼
Report: {average_reward_margin, consistency_results,
          flagged_for_human_review, pass: bool}
         ↑
         Human reviews flagged_for_human_review before any deployment decision.
```

---

## Storage Backend Switch

`STORAGE_BACKEND` in [`config.py`](config.py) is the **single switch** to flip
from JSON to SQLite:

```python
# config.py
STORAGE_BACKEND = "json"    # ← change to "sqlite" to activate SQLite backend
```

Or set the environment variable:
```bash
export RLHF_STORAGE_BACKEND=sqlite
```

The SQLite backend ([`storage/sqlite_store.py`](storage/sqlite_store.py)) already
exists with an identical public interface (`write_pair`, `update_label`,
`read_all_pairs`, `query`).  Call `sqlite_store.init_db()` once to create the
table, then flip the switch — nothing else needs to change.

---

## Category-Based Contamination Prevention

The `Category` enum in [`config.py`](config.py) is enforced at **write time**
via [`storage/categorize.py`](storage/categorize.py), not just at export time.
Both generator functions call `assign_category()` before returning a pair.
This prevents, for example, HR preference data from accidentally contaminating
the FINANCIAL fine-tuning run — the category is baked into the record before
it ever reaches a storage backend.

Export-time filtering (`export_for_dpo(category=Category.HR)`) is a second
layer of defence, but the primary enforcement is at generation time.

---

## Quick CLI Reference

```bash
# Export HR pairs to DPO format
python -c "from rlhf.export.dpo_export import export_for_dpo; from rlhf.config import Category; export_for_dpo(Category.HR)"

# Run DPO training for HR category
python -m rlhf.training.train \
    --category HR \
    --export-path rlhf/data/exports/dpo_HR_20260830_120000.jsonl \
    --base-model meta-llama/Llama-3.2-3B-Instruct
```

---

## Dependency Notes

The `rlhf` package is importable without ML dependencies (mirrors the
pattern in `ml/train_prompt_injection.py`).  ML packages are imported lazily
inside `training/` functions:

```
# Core (already in requirements.txt)
pydantic

# For training only (install separately)
pip install trl peft transformers accelerate bitsandbytes torch datasets
```

---

## Stubs to Wire Up

| File | Stub Function | Where to point it |
|------|--------------|-------------------|
| `generators/api_vs_api.py` | `call_api_model` | `backend/utils/llm_judge.py` (OpenAI/Anthropic provider) or `backend/shared/llm_simulator.py` (dev) |
| `generators/local_vs_local.py` | `call_local_model` | `backend/shared/gpu_adapter.py` (`GPUAdapter`) or `backend/shared/llm_simulator.py` |
| `judges/llm_judge.py` | `call_judge_llm` | `backend/utils/llm_judge.py` (`get_provider().complete(...)`) |
