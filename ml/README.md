# Prompt-injection model experiment

The default gateway intentionally uses deterministic, inspectable detectors so
it runs without a GPU or external downloads. This directory is the isolated
research path for the first learned engine described in the design brief:
RoBERTa binary classification of safe versus prompt-injection text.

Install the optional dependencies in a dedicated environment, then run:

```powershell
pip install -r ml/requirements-ml.txt
python ml/train_prompt_injection.py --data data/prompt_injection_sample.jsonl --output ml/artifacts/injection-v0
```

The sample data is only a pipeline test. A useful model needs a curated,
licensed dataset with attack families held together by `group_id`, a protected
test set, calibration, adversarial/multilingual evaluation, and a documented
false-negative target. Do not swap a resulting model into a blocking path
without that evaluation and a staged observe-only rollout.

---

# Optional learned-detector layer (generalized)

`train_prompt_injection.py` above is the original single-task experiment. The
generalized, config-driven trainer is `train_detector.py`, and the runtime half
that lets a trained model be consulted by the live detectors — **default-OFF** —
is `backend/shared/model_backend.py`. Nothing here is required by the default
gateway: with no artifact configured, every detector stays purely deterministic.

## Per-detector fine-tuning plan → what lives where

| Detector | Model family | Trainer task | Consumed by |
|----------|--------------|--------------|-------------|
| injection | RoBERTa / DeBERTa-v3 | `--task injection` | `backend/detectors/injection.py` (hot path) |
| safety / toxicity | pretrained toxicity classifier, fine-tuned | `--task toxicity` | `backend/detectors/safety.py` (hot path) |
| fairness / bias | RoBERTa / DeBERTa + HateXplain rationale spans | `--task fairness` | (rationale spans → `DetectorResult.evidence`) |
| grounding | NLI cross-encoder / HHEM, scored per-claim vs retrieved chunks | (NLI, trained separately) | async `GroundingEngineDetector` in `async_analytics.py` |
| PII | Presidio (regex + NER) | — (different shape; **follow-up**, not in this layer) | `backend/detectors/pii.py` |

Rationale spans and model scores need **no schema change** — `DetectorResult.evidence`
is a `list[str]`, and grounding reads the existing `GovernanceRequest.retrieved_context`.

## Train

```powershell
pip install -r ml/requirements-ml.txt
python -m ml.train_detector --task injection \
    --data data/injection.jsonl --output ml/artifacts/injection-v0 --lora
```

Dataset is JSONL with `{"text": ..., "label": 0|1, "group_id": ...}` per line.
`group_id` holds attack families / paraphrase clusters together so the group-aware
split (`ml/common.grouped_split`) prevents near-duplicate leakage across
train/valid/test. The trainer then:

1. **Calibrates** scores with temperature scaling on the validation split
   (`ml/common.fit_temperature`).
2. **Chooses the operating threshold for a target false-negative rate**
   (`--target-fnr`, default 0.05) via `ml/common.select_threshold_for_fnr` —
   never a naive 0.5.
3. **Evaluates on a protected test split** at that frozen threshold (confusion,
   FPR/FNR, ROC-AUC, AUPRC).
4. **`--lora`** does a parameter-efficient LoRA/PEFT fine-tune (fits a free
   Colab/Kaggle T4); falls back to a full fine-tune if `peft` is absent.

Output artifact (drop-in for the runtime seam):

```
ml/artifacts/injection-v0/
    model/            HF model + tokenizer
    calibration.json  temperature, threshold, positive_label, positive_index, max_length
    evaluation.json   split sizes, validation operating point, protected-test metrics
```

## Serve (the runtime seam)

Point an environment variable at the trained `model/` directory. Everything is
lazy and never raises — a missing var, missing artifact, or absent ML stack all
resolve to the deterministic fallback.

| Env var | Enables | Default when unset |
|---------|---------|--------------------|
| `CONTROLPLANE_MODEL_INJECTION` | injection detector model consult | regex-only |
| `CONTROLPLANE_MODEL_SAFETY` | safety/toxicity model consult | regex-only |
| `CONTROLPLANE_MODEL_GROUNDING` | NLI groundedness in async grounding engine | token-overlap heuristic |

```powershell
$env:CONTROLPLANE_MODEL_INJECTION = "ml/artifacts/injection-v0/model"
```

A model can only **raise** a detector's risk or promote its label; it never
lowers deterministic signal. **Staged rollout:** deploy observe-only first
(inspect the `model:<task>:<score>` evidence in the audit trail), confirm FPR/FNR
against a labeled set, and only then let it influence a blocking path.

