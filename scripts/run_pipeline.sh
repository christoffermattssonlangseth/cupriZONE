#!/usr/bin/env bash
# Run preprocessing then niche analysis sequentially, logging to logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/sc_py312/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMBA_NUM_THREADS=6 NJOBS=6
mkdir -p logs
echo "start $(date)" > logs/pipeline.status
if [ ! -f results/xenium_all.h5ad ]; then
  $PY -u scripts/02_preprocess.py > logs/02_preprocess.log 2>&1 || { echo "preprocess FAILED $(date)" >> logs/pipeline.status; exit 1; }
  echo "preprocess ok $(date)" >> logs/pipeline.status
fi
$PY -u scripts/02b_figures.py > logs/02b_figures.log 2>&1 || { echo "figures FAILED $(date)" >> logs/pipeline.status; exit 1; }
echo "figures ok $(date)" >> logs/pipeline.status
$PY -u scripts/03_niche.py > logs/03_niche.log 2>&1 || { echo "niche FAILED $(date)" >> logs/pipeline.status; exit 1; }
echo "niche ok $(date)" >> logs/pipeline.status
NICHE_K=12 NICHE_TAG=_k12 $PY -u scripts/03_niche.py > logs/03_niche_k12.log 2>&1 || { echo "niche k12 FAILED $(date)" >> logs/pipeline.status; exit 1; }
echo "niche k12 ok $(date)" >> logs/pipeline.status
$PY -u scripts/04_report.py > logs/04_report.log 2>&1 || { echo "report FAILED $(date)" >> logs/pipeline.status; exit 1; }
echo "report ok $(date)" >> logs/pipeline.status
echo "end $(date)" >> logs/pipeline.status
