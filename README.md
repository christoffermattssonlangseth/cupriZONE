# cupriZONE — Xenium niche analysis of GSE266690

Mouse forebrain Xenium In Situ data from GEO **GSE266690** (Nature Neuroscience 2025, PMID 39823226 /
40780964; "Single cell approaches define neural stem cell niches and identify microglial ligands that
enhance precursor-mediated remyelination"). 19 Xenium sections, mouse brain panel (248 predesigned +
100 custom genes; the custom set differs between the two batches, 307 genes common to all sections):

| experiment | groups (n sections) | GSMs |
|---|---|---|
| `recovery` | Control (2) vs CupRap (2) | GSM8253804–807 |
| `no_recovery` | Control (3) vs CupRap (3) | GSM8647384–389 |
| `infusion` | BSA_infused (6) vs OSM_infused (3) | GSM8647390–398 |

CupRap = 6 weeks cuprizone + rapamycin demyelination. Only cell-level outputs were downloaded
(~1.4 GB); the transcript tables (~2.5 GB/section) and morphology images (~8 GB/section) were skipped.
The MERSCOPE (GSM8253800–803) and scRNA-seq samples in the SuperSeries are not used.

## Layout

```
scripts/samples.tsv          sample sheet (GSM, GEO prefix, sample_id, condition, experiment)
scripts/01_download_geo.sh   cherry-pick files from GEO FTP into data/geo/<sample_id>/
scripts/02_preprocess.py     load, QC, normalise, PCA, kNN, Leiden, marker annotation, UMAP, figures
scripts/03_niche.py          spatial kNN graph, neighbourhood composition, k-means niches, enrichment,
                             abundance contrasts, squidpy neighbourhood enrichment, spatial maps
scripts/05_cellcharter.py    CellCharter niches: Delaunay graph, 3-layer neighbourhood aggregation of PCA, GMM,
                             k by ClusterAutoK stability (subsample), final fit on 500k cells, predict all
scripts/niche_downstream.py  shared niche outputs (enrichment, abundance/contrasts, spatial maps) for 03 and 05
scripts/run_pipeline.sh      runs 02 (if no checkpoint), 02b, 03 (k auto + k=12), 04
scripts/run_cellcharter.sh   runs 05, then 03 and 04 again
scripts/viz.py               shared palette / matplotlib chrome
```

Compute runs on the analysis Mac (`christoffer@100.115.223.38`, env `~/miniconda3/envs/sc_py312`) in
`~/work/karolinska/development/cupriZONE/`; results are synced back into `results/` here.

## Key parameters

- QC: ≥20 transcripts, ≥5 genes per cell, cell area within the 1st–99th percentile per section; genes in ≥10 cells.
- Normalisation: `normalize_total(target_sum=100)` + `log1p`; PCA 30 comps; kNN 15; Leiden (igraph) resolution 1.0.
- Annotation: cluster-level marker-set scores (mean log2FC of set genes vs all cells); pan-neuronal fallback.
- Niches: 15 spatial nearest neighbours per cell (within section) → cell-type composition → MiniBatchKMeans,
  k chosen by silhouette over 5–14 (override with `NICHE_K`).

## Outputs (`results/`)

- `REPORT.md` — full report: QC table, annotation, niches at k=6 (silhouette-selected) and k=12 (forced, finer), condition contrasts, composition tables.
- `figures/` — QC violins, UMAP (120k subsample), marker dotplot, composition bars, spatial cell-type maps.
- `niches_cellcharter/` — CellCharter GMM niches (k=8 by stability), same outputs plus `autok_stability.png`, `method.json`.
- `niches/` (k=6) and `niches_k12/` — enrichment heatmap, spatial niche maps, abundance per section, log2FC dot charts,
  differential abundance CSV, squidpy neighbourhood-enrichment z-scores (per section + condition means, k=6 dir only).
- On the remote only (too large for this disk): `results/xenium_all.h5ad` (2.36 M cells; obs has `leiden`, `cell_type`,
  `niche`, `niche_k12`, `niche_cc`; X = log-normalised, `layers["counts"]`, `obsm["X_pca"|"spatial"]`, `obsp["connectivities"]`),
  `results/cell_metadata_with_niches.csv.gz`, `results/niches*/neighbourhood_composition.parquet`.

## Pipeline notes

- `02_preprocess.py` writes a checkpoint h5ad right after annotation; `02b_figures.py` and `03_niche.py` read from it via h5py
  (metadata + coordinates only) because the shared analysis Mac killed processes above ~10 GB when its swap was full.
- `scripts/annotation_overrides.json` documents the two manual relabels (cluster 2 → Pericyte-VSMC; cluster 31 → lesion glia).
- Steps 02b/03/04 rerun in ~15 min in total; `run_pipeline.sh` skips step 02 when the checkpoint exists.
- Local `rsync` in `/usr/local/bin` is an x86 binary that no longer runs; use `/usr/bin/rsync` or `scp`.
