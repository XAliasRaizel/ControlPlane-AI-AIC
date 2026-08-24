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
