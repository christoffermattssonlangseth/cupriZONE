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
scripts/06_download_merscope.sh  MERSCOPE (Vizgen) cell_by_gene + cell_metadata for the 4 recovery sections (172 MB)
scripts/07_merscope_to_h5ad.py   assemble them into results/merscope/merscope_recovery_raw.h5ad (759k cells x 300 genes, raw)
scripts/08_striatal_lesion_check.py  what the white-matter/microglia niche spots inside the striatum are (results/striatal_lesion_check/)
scripts/viz.py               shared palette / matplotlib chrome
```

Compute runs on the analysis Mac (`christoffer@100.115.223.38`, env `~/miniconda3/envs/sc_py312`) in
`~/work/karolinska/development/cupriZONE/`; results are synced back into `results/` here.

## Groupings

Use `obs['group']` (Xenium: `NoRecov_Cntl`, `NoRecov_CupRap`, `Recov_Cntl`, `Recov_CupRap`, `Inf_BSA`, `Inf_OSM`;
MERSCOPE: `MERSCOPE_Recov_Cntl`, `MERSCOPE_Recov_CupRap`) for contrasts, not `condition` alone: `condition` = CupRap
spans two time points and two batches, and the two control groups differ in batch. `timepoint` and `batch` are also in obs
and in `scripts/samples*.tsv`.

## Time points

The cuprizone arm has two time points: `no_recovery` = 6 weeks CupRap, collected immediately (acute demyelination);
`recovery` = 6 weeks CupRap + 3 weeks recovery (early remyelination). They were run in different Xenium batches
(different custom panel, software and segmentation), so compare each to its own controls. The MERSCOPE sections are
the recovery time point only. The infusion arm (OSM vs BSA, 7 days, healthy mice, PFA-fixed) is a separate experiment.

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
- `results/xenium_all.h5ad` (2.36 M cells, 3.8 GB, remote + local copy, git-ignored; obs has `leiden`, `cell_type`,
  `niche`, `niche_k12`, `niche_cc`, `lesion_wm_niche`, `vsvz_wall`/`vsvz_niche`, `striatum_part`/`striatum_niche`,
  `oligo_subcluster`/`oligo_immune_score` (see `uns['obs_columns']`); obsp has the expression kNN plus the spatial 15-NN
  and Delaunay graphs (`uns['graphs']`); X = log-normalised, `layers["counts"]`, `obsm["X_pca"|"spatial"]`, `obsp["connectivities"]`),
  `results/cell_metadata_with_niches.csv.gz`, `results/niches*/neighbourhood_composition.parquet`.

## Pipeline notes

- `02_preprocess.py` writes a checkpoint h5ad right after annotation; `02b_figures.py` and `03_niche.py` read from it via h5py
  (metadata + coordinates only) because the shared analysis Mac killed processes above ~10 GB when its swap was full.
- `scripts/annotation_overrides.json` documents the two manual relabels (cluster 2 → Pericyte-VSMC; cluster 31 → lesion glia).
- Steps 02b/03/04 rerun in ~15 min in total; `run_pipeline.sh` skips step 02 when the checkpoint exists.
- Local `rsync` in `/usr/local/bin` is an x86 binary that no longer runs; use `/usr/bin/rsync` or `scp`.

## Region-focused analyses (added 2026-10-02)

Whole-section niches are too coarse for the structures the paper is about, so two focused analyses mimic its ROI:

- `scripts/14_vsvz_focus.py` → `results/vsvz_focus/`: V-SVZ = cells ≤75 µm from the lateral-ventricle ependymal lining
  (spatial clusters of ependymal cells away from the midline; choroid plexus excluded). Wall from tissue context within
  150 µm: white-matter glia (oligodendrocyte/OPC/microglia/lesion glia) → "dorsal" (roof under the corpus callosum and the
  white-matter-adjacent medial wall), striatal MSN → "lateral", otherwise "medial". Per-section composition, progenitor
  activation (*Mki67*, *Egfr*, *Ascl1*), microglia-state genes, focused k-means niches within the V-SVZ, zoom maps.
- `scripts/15_striatum_focus.py` → `results/striatum_focus/`: caudate putamen = ≥25 % striatal MSN within 75 µm, outside
  the V-SVZ band; fibre bundles = patches where white-matter glia are ≥50 % of cells within 40 µm, matrix = the rest.
  Same metrics and focused niches. Kept separate from the V-SVZ analysis.
- `scripts/focus_utils.py` holds the shared code.

MERSCOPE: `scripts/12_merscope_preprocess.py` (QC ≥15 transcripts / ≥5 genes / volume 1–99 %, Leiden, annotation) and
`scripts/13_merscope_niche.py` (k-means + CellCharter niches, cross-platform comparison) → `results/merscope/`.

## Interface ("barrier") niche test (`scripts/20_interface_artefact.py`, `21_interface_specificity.py` → `results/interface_artefact/`)

Does neighbourhood aggregation manufacture a niche at the white/grey-matter interface (cf. the "Ventral Rim OL" niche in RRMap)?
Acute batch, 6 sections. Geometric bands (±75 µm from the WM/GM border, independent of niches); CellCharter GMM at k = 12/24/36
with 0/1/2/3/5 aggregation layers; shuffle control (embeddings permuted within WM / GM / other per section, so geometry is kept
but any rim biology is destroyed); cell-intrinsic oligodendrocyte expression by band. Summary: `summary_interface_specificity.csv`.
