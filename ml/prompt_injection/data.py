"""
ml/prompt_injection/data.py

Loads and merges public prompt-injection datasets from Hugging Face,
ready for the group-aware split in ml/common/data_utils.py.

This extends -- does not replace -- the existing ml/train_prompt_injection.py.
The goal is one validated dataset-loading path, not two.

NOT executed here (no network, no `datasets` library in this sandbox).
Written against the datasets library's standard load_dataset() API.
Run on Colab/Kaggle where you have real internet access.

IMPORTANT: Before trusting the rename/remap logic verbatim, confirm each
dataset's actual column names with a quick `ds["train"].features` print in
your real environment — public datasets do not always share a schema even
when they cover the same task.

Improvement over the original: added `rubend18/ChatGPT-jailbreak-prompts`
as a 4th source. This dataset specifically covers modern LLM jailbreaks
(DAN variants, roleplay-based bypasses, "grandma exploit") that the original
3 sources underrepresent.
"""
from __future__ import annotations

import pandas as pd

# (hf_dataset_name, text_column, label_column, label_map)
# label_map remaps source labels to 0=SAFE, 1=INJECTION; None means no remapping.
DATASET_SOURCES = [
    ("deepset/prompt-injections", "text", "label", None),
    ("xTRam1/safe-guard-prompt-injection", "text", "label", None),
    ("jayavibhav/prompt-injection", "text", "label", None),
    # Modern LLM jailbreaks — DAN variants, roleplay bypasses, "grandma exploit".
    # All rows in this dataset are injections (label=1); we assign label=1 for all.
    ("rubend18/ChatGPT-jailbreak-prompts", "Prompt", None, {"forced_label": 1}),
]


def _load_single(name: str, text_col: str, label_col, label_map: dict,
                 cache_dir: str) -> "pd.DataFrame":
    from datasets import load_dataset

    ds = load_dataset(name, cache_dir=cache_dir)
    split = ds["train"] if "train" in ds else list(ds.values())[0]
    df = split.to_pandas()

    # Handle the jailbreak dataset which has no label column
    forced = label_map.pop("forced_label", None) if isinstance(label_map, dict) else None
    if forced is not None:
        df = df.rename(columns={text_col: "text"}) if text_col != "text" else df
        df["label"] = forced
    else:
        if text_col != "text":
            df = df.rename(columns={text_col: "text"})
        if label_col and label_col != "label":
            df = df.rename(columns={label_col: "label"})
        if label_map:
            df["label"] = df["label"].map(label_map)

    df["source"] = name
    return df[["text", "label", "source"]].dropna(subset=["text", "label"])


def load_and_merge(cache_dir: str = "ml/data_cache") -> pd.DataFrame:
    """Load and merge all DATASET_SOURCES into a single DataFrame.

    Returns a deduplicated DataFrame with columns: text, label, source.
    Label encoding: 0=SAFE, 1=INJECTION.
    """
    frames = []
    for name, text_col, label_col, label_map in DATASET_SOURCES:
        lm = dict(label_map) if label_map else {}
        try:
            df = _load_single(name, text_col, label_col, lm, cache_dir)
            frames.append(df)
            print(f"  Loaded {len(df):,} rows from {name}")
        except Exception as exc:
            print(f"  WARNING: could not load {name}: {exc}")

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset="text").reset_index(drop=True)
    print(
        f"\nMerged {len(frames)} sources: {before:,} rows -> {len(merged):,} after exact-dedup"
    )
    print(merged["label"].value_counts().to_string())
    return merged


if __name__ == "__main__":
    df = load_and_merge()
    df.to_parquet("ml/data_cache/prompt_injection_merged.parquet")
    print(f"\nSaved {len(df):,} rows to ml/data_cache/prompt_injection_merged.parquet")
