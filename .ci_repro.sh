#!/usr/bin/env bash
set -eo pipefail
SRC="/mnt/d/tushar/random/aIC 2026 me/new_/ControlPlane-AI-AIC"
DST="$HOME/ccrepro"
echo "== copy tracked source to ext4 (case-sensitive) =="
rm -rf "$DST"; mkdir -p "$DST"
cp -r "$SRC/backend" "$SRC/ml" "$SRC/tests" "$SRC/policies" "$SRC/requirements.txt" "$DST/"
cd "$DST"
python3 --version
export PATH="$HOME/.local/bin:$PATH"
export PIP_BREAK_SYSTEM_PACKAGES=1
if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "== bootstrapping pip via get-pip (user site) =="
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python3 /tmp/get-pip.py --user --break-system-packages -q
fi
python3 -m pip --version
grep -v streamlit requirements.txt > .req.txt
echo "== pip install --user -r requirements (minus streamlit) =="
python3 -m pip install --user --break-system-packages -r .req.txt -q
echo "== compileall =="
python3 -m compileall -q backend ml tests || true
echo "===== PYTEST (REAL LINUX ext4, py3.12) ====="
python3 -m pytest -q
