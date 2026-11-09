#!/usr/bin/env bash
# One-time setup on macOS / Linux:  bash setup.sh
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Done. Next time just run: source .venv/bin/activate && export PYTHONPATH=src"
