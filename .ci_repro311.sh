#!/usr/bin/env bash
set -eo pipefail
SRC="/mnt/d/tushar/random/aIC 2026 me/new_/ControlPlane-AI-AIC"
DST="$HOME/ccrepro311"
echo "== copy tracked source to ext4 (case-sensitive) =="
rm -rf "$DST"; mkdir -p "$DST"
cp -r "$SRC/backend" "$SRC/ml" "$SRC/tests" "$SRC/policies" "$SRC/frontend" "$SRC/requirements.txt" "$DST/"
cd "$DST"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260825/cpython-3.11.16%2B20260825-x86_64-unknown-linux-gnu-install_only.tar.gz"
echo "== fetch standalone CPython 3.11 =="
curl -sSL "$URL" -o /tmp/py311.tar.gz
tar -xzf /tmp/py311.tar.gz -C "$DST"
PY="$DST/python/bin/python3.11"
$PY --version
$PY -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
echo "== FULL requirements.txt (incl streamlit) on Linux py3.11 =="
.venv/bin/python -m pip install -r requirements.txt -q
echo "== compileall backend frontend tests =="
.venv/bin/python -m compileall -q backend frontend tests || echo "COMPILEALL_FAILED"
echo "===== PYTEST (Linux + py3.11 + full requirements = EXACT CI) ====="
.venv/bin/python -m pytest -q
