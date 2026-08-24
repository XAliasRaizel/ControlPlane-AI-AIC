# Prototype data

`prompt_injection_sample.jsonl` is a deliberately small, synthetic smoke-test
dataset. It is safe to commit and useful for exercising the detector and the
optional training script; it is **not** a benchmark and must not be used to
claim a security detection rate.

For a real experiment, maintain immutable raw data separately, remove exact
and near duplicates, group closely related variants by `group_id`, and split
at the group level before tokenization. Keep the final test set inaccessible
until model and thresholds have been selected.

Every record uses this schema:

```json
{
  "text": "Example prompt",
  "label": 0,
  "severity": "LOW",
  "group_id": "safe-001",
  "application": "support"
}
```

`label` is `0` for safe and `1` for prompt injection.
