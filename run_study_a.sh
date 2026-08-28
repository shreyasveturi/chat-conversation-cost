#!/usr/bin/env bash
# Study A, end to end. Verifies the arithmetic before trusting it with real data.
set -euo pipefail

PY="${PY:-.venv/bin/python}"

if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]] && [[ ! -s "$HOME/.cache/huggingface/token" ]]; then
  cat >&2 <<'MSG'
No Hugging Face credentials found.

WildChat-1M is a gated dataset. To run this study you need to:
  1. accept the terms at https://huggingface.co/datasets/allenai/WildChat-1M
  2. create a read token at https://huggingface.co/settings/tokens
  3. export HF_TOKEN=hf_...        (or run: hf auth login)
MSG
  exit 1
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
