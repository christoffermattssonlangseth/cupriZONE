#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/sc_py312/bin/python}
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMBA_NUM_THREADS=6
echo "start $(date)" > logs/merscope.status
$PY -u scripts/12_merscope_preprocess.py > logs/12_merscope_preprocess.log 2>&1 || { echo "preprocess FAILED $(date)" >> logs/merscope.status; exit 1; }
echo "preprocess ok $(date)" >> logs/merscope.status
$PY -u scripts/13_merscope_niche.py > logs/13_merscope_niche.log 2>&1 || { echo "niche FAILED $(date)" >> logs/merscope.status; exit 1; }
echo "niche ok $(date)" >> logs/merscope.status; echo "end $(date)" >> logs/merscope.status
