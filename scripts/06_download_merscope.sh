#!/usr/bin/env bash
# Download the MERSCOPE (Vizgen) cell-level outputs for the 4 sections in GSE266690 (3-week-recovery time point).
# Only cell_by_gene + cell_metadata are needed to assemble an AnnData; detected_transcripts (0.25-1.7 GB each)
# and the full-run zips (7-20 GB each, images + .vzg) are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/../data/geo_merscope}"
mkdir -p "$OUT"
SUFFIXES=(cell_by_gene.csv.gz cell_metadata.csv.gz summary.png.gz)
tail -n +2 "$HERE/samples_merscope.tsv" | while IFS=$'\t' read -r gsm prefix sample_id cond expt rep; do
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
