#!/usr/bin/env bash
# CellCharter niches, then re-run the k-means step (refactored) and the report.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/sc_py312/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMBA_NUM_THREADS=6
mkdir -p logs
echo "start $(date)" > logs/cellcharter.status
$PY -u scripts/05_cellcharter.py > logs/05_cellcharter.log 2>&1 || { echo "cellcharter FAILED $(date)" >> logs/cellcharter.status; exit 1; }
echo "cellcharter ok $(date)" >> logs/cellcharter.status
$PY -u scripts/03_niche.py > logs/03_niche_rerun.log 2>&1 || { echo "niche rerun FAILED $(date)" >> logs/cellcharter.status; exit 1; }
echo "niche rerun ok $(date)" >> logs/cellcharter.status
$PY -u scripts/04_report.py > logs/04_report.log 2>&1 || { echo "report FAILED $(date)" >> logs/cellcharter.status; exit 1; }
echo "report ok $(date)" >> logs/cellcharter.status
echo "end $(date)" >> logs/cellcharter.status
