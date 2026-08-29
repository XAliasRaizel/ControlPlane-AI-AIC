"""Download sentence-transformers/all-MiniLM-L6-v2 for the sensitive-intent detector.

Track A (pretrained, no fine-tuning) — same pattern as download_pretrained.py.
Writes to ml/artifacts/sensitive-intent/model/

Usage:
    python ml/scripts/download_minilm.py
    python ml/scripts/download_minilm.py --out ml/artifacts/sensitive-intent/model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # ml/scripts -> ml -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUT = _REPO_ROOT / "ml" / "artifacts" / "sensitive-intent" / "model"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def download(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (idempotent)
    config_file = out_dir / "config.json"
    if config_file.exists():
        print(f"Model already present at {out_dir} — skipping download.")
        return

    print(f"Downloading {MODEL_NAME} -> {out_dir} ...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers is not installed.")
        print("Install it with:  pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer(MODEL_NAME)
    model.save(str(out_dir))
    print(f"Model saved to {out_dir}")

    # Verify it loads back cleanly
    _ = SentenceTransformer(str(out_dir))
    print("Verification load OK.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output directory for model files")
    args = parser.parse_args()
    download(Path(args.out))


if __name__ == "__main__":
    main()
