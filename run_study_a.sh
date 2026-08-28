#!/usr/bin/env bash
# Study A, end to end. Verifies the arithmetic before trusting it with real data.
set -euo pipefail

PY="${PY:-.venv/bin/python}"

if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]] && [[ ! -s "$HOME/.cache/huggingface/token" ]]; then
  echo "[note] No Hugging Face credentials found. WildChat-1M is public" >&2
  echo "       (ODC-BY), so this should still work. If downloads fail, set" >&2
  echo "       HF_TOKEN or run: hf auth login" >&2
fi

echo "==> 1/4  verifying the cost arithmetic"
"$PY" study_a/test_arithmetic.py

echo
echo "==> 2/4  sampling and filtering WildChat-1M"
"$PY" data/sample_study_a.py

echo
echo "==> 3/4  tokenizing and computing per-turn costs"
"$PY" study_a/compute_costs.py

echo
echo "==> 4/4  rendering charts and README"
"$PY" analysis/charts_study_a.py
"$PY" analysis/render_readme.py

echo
echo "Done. Tables in results/tables/, charts in results/figures/."
