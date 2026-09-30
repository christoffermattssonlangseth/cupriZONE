#!/usr/bin/env bash
# Download the minimal Xenium outputs (cell x gene matrix, cell metadata, cell boundaries,
# run metrics) for the 19 Xenium sections in GSE266690. Skips transcripts CSVs (~2.5 GB each)
# and morphology OME-TIFFs (~8 GB each) which are not needed for cell-level niche analysis.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/../data/geo}"
mkdir -p "$OUT"
SUFFIXES=(matrix.mtx.gz barcodes.tsv.gz features.tsv.gz cells.csv.gz cell_boundaries.csv.gz metrics_summary.csv.gz experiment.xenium.txt.gz)
tail -n +2 "$HERE/samples.tsv" | while IFS=$'\t' read -r gsm prefix sample_id cond expt rep; do
  d="$OUT/$sample_id"; mkdir -p "$d"
  base="https://ftp.ncbi.nlm.nih.gov/geo/samples/${gsm:0:7}nnn/$gsm/suppl"
  for s in "${SUFFIXES[@]}"; do
    f="${gsm}_${prefix}_${s}"
    if [ -s "$d/$s" ]; then continue; fi
    echo "[$(date +%T)] $sample_id <- $f"
    curl -sSL --retry 5 --retry-delay 5 -o "$d/$s.part" "$base/$f" && mv "$d/$s.part" "$d/$s"
  done
done
echo "DONE $(date)"; du -sh "$OUT"
