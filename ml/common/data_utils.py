"""
ml/common/data_utils.py

Group-aware train/val/test splitting shared across all four fine-tuning tracks.
Near-identical rephrasings of the same attack cannot land in both train and test
(which would let a model memorize rather than generalize).

Improvements over the original ml/common.py grouped_split():
  1. Works with DataFrames directly — no JSONL round-trip needed.
  2. After splitting, checks for label skew: warns when any split is >90% one
     class, which makes recall/FPR undefined for a governance classifier.
  3. Raises immediately with a clear message when group count is too low.
"""
from __future__ import annotations

import hashlib
import re
import warnings
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def default_group_key(text: str) -> str:
    """Group key for near-identical texts (exact-match after normalization).

    Conservative: under-merges some paraphrases (safer than over-merging
    unrelated examples into one group and starving splits of diversity).
    For a stronger version, cluster on sentence embeddings.
    """
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def _check_label_skew(
    split_name: str, df: pd.DataFrame, label_col: str = "label"
) -> None:
    """Warn if any class dominates a split (>90% of rows).

    A 90%+ dominant class makes recall / FPR undefined or unreliable —
    exactly the metrics that matter for a governance detector.
    """
    if label_col not in df.columns:
        return
    counts = df[label_col].value_counts(normalize=True)
    for cls, frac in counts.items():
        if frac > 0.90:
            warnings.warn(
                f"Label skew in {split_name}: class {cls!r} is {frac:.0%} of the split "
                f"({len(df)} rows). Recall and FPR will be unreliable. "
                f"Consider adding more minority-class examples or adjusting split sizes.",
                UserWarning,
                stacklevel=3,
            )


def group_aware_split(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    group_col: Optional[str] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    check_skew: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df so no group appears in more than one of train/val/test.

    Falls back to default_group_key() (normalized-text hash) when group_col
    is not supplied. Raises if the resulting splits are not disjoint (should
    never happen given GroupShuffleSplit's contract, but this function is
    paranoid about data leakage and checks its own work).

    Parameters
    ----------
    df          Input DataFrame.
    text_col    Column used to derive group keys when group_col is None.
    label_col   Column checked for class skew after splitting.
    group_col   Pre-computed group identifier column, or None.
    test_size   Fraction of data reserved for the held-out test set.
    val_size    Fraction of data reserved for the validation set.
    seed        Random seed for reproducibility.
    check_skew  Warn when any split is >90% one class (default: True).

    Returns
    -------
    (train, val, test) DataFrames with no group overlap.
    """
    working = df.copy()
    added_group_col = False
    if group_col is None:
        working["_group"] = working[text_col].map(default_group_key)
        group_col = "_group"
        added_group_col = True

    n_groups = working[group_col].nunique()
    if n_groups < 3:
        raise ValueError(
            f"Only {n_groups} distinct group(s) found -- group-aware splitting "
            f"needs at least 3 distinct groups to separate train/val/test. "
            f"Check that group_col (or the default near-duplicate grouping) is not "
            f"collapsing all rows into one bucket."
        )

    splitter1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter1.split(working, groups=working[group_col]))
    train_val = working.iloc[train_val_idx]
    test = working.iloc[test_idx]

    relative_val_size = val_size / (1 - test_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx, val_idx = next(splitter2.split(train_val, groups=train_val[group_col]))
    train = train_val.iloc[train_idx]
    val = train_val.iloc[val_idx]

    train_groups = set(train[group_col])
    val_groups = set(val[group_col])
    test_groups = set(test[group_col])
    assert not (train_groups & val_groups), "leakage: a group appears in both train and val"
    assert not (train_groups & test_groups), "leakage: a group appears in both train and test"
    assert not (val_groups & test_groups), "leakage: a group appears in both val and test"

    if check_skew:
        _check_label_skew("train", train, label_col)
        _check_label_skew("val", val, label_col)
        _check_label_skew("test", test, label_col)

    drop_cols = ["_group"] if added_group_col else []
    return (
        train.drop(columns=drop_cols),
        val.drop(columns=drop_cols),
        test.drop(columns=drop_cols),
    )
