"""Assemble results/REPORT.md from the CSV outputs of 02_preprocess.py and 03_niche.py."""
import os, sys, json, re
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.environ.get("RES_DIR", f"{ROOT}/results"); N = f"{RES}/niches"
ital = lambda g: f"*{g}*"   # mouse gene symbols italicised

def to_md(df, index=True):
    """Minimal markdown table (avoids the tabulate dependency)."""
    df = df.copy()
    if index:
        df = df.reset_index()
    cells = df.astype(object).where(pd.notna(df), "")
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(map(str, cells.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(cells.columns)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join([head, sep] + body)
pd.DataFrame.to_markdown = lambda self, index=True, **kw: to_md(self, index=index)

qc = pd.read_csv(f"{RES}/qc_per_sample.csv", index_col=0)
ann = pd.read_csv(f"{RES}/cluster_annotation.csv", index_col=0)
ctf = pd.read_csv(f"{RES}/celltype_fractions_per_sample.csv", index_col=0)
ctc = pd.read_csv(f"{RES}/celltype_counts_per_sample.csv", index_col=0)
samples = pd.read_csv(f"{ROOT}/scripts/samples.tsv", sep="\t").set_index("sample_id")

L = []
w = L.append
w("# GSE266690 Xenium: preprocessing and niche analysis\n")
w(f"Sections: {len(qc)}. Cells after QC: {int(qc.n_cells_qc.sum()):,} of {int(qc.n_cells_raw.sum()):,} "
  f"({100*qc.n_cells_qc.sum()/qc.n_cells_raw.sum():.1f} %). Genes common to all sections: 307 "
  f"(the two panel batches differ in their 100-gene custom sets).\n")
w("Pipeline: `scripts/02_preprocess.py` (QC, normalisation, PCA, kNN, Leiden, marker-set annotation) → "
  "`scripts/03_niche.py` (spatial kNN graph, neighbourhood composition, k-means niches, enrichment, "
  "abundance contrasts, squidpy neighbourhood enrichment). Figures are in `results/figures/` and `results/niches/`.\n")

w("## 1. QC per section\n")
t = qc[["gsm", "experiment", "condition", "n_cells_raw", "n_cells_qc", "frac_removed", "median_transcripts_qc",
        "median_genes_qc", "median_cell_area_qc", "neg_probe_rate"]].copy()
t["frac_removed"] = (100 * t.frac_removed).round(1); t["neg_probe_rate"] = (100 * t.neg_probe_rate).round(2)
t.columns = ["GSM", "experiment", "condition", "cells raw", "cells QC", "% removed", "median transcripts", "median genes",
             "median area µm²", "neg-probe %"]
w(t.to_markdown()); w("")
w("Filters: ≥20 transcripts, ≥5 genes, cell area within the 1st–99th percentile of its section. "
  "Note the ~2× difference in median cell area between the 2023 batch (recovery experiment, Xenium analysis 1.4) "
  "and the 2024 batch (1.7): segmentation differs between batches, so cross-experiment comparisons of densities "
  "or per-cell counts should stay within experiment.\n")
w("![QC](figures/qc_violin_per_sample.png)\n")

w("## 2. Clustering and cell-type annotation\n")
w(f"Leiden (resolution 1.0, igraph) gave {len(ann)} clusters, annotated by marker-set scores (mean log2FC of the top-3 "
  "set genes vs all cells; pan-neuronal fallback = “Neuron (other)”). Overrides in `scripts/annotation_overrides.json`. "
  "Per-cluster scores are in `results/cluster_marker_scores.csv`; top DE genes per cluster in `results/cluster_annotation.csv`.\n")
tot = ctc.sum(1).sort_values(ascending=False)
t = pd.DataFrame({"cells": tot, "% of all": (100 * tot / tot.sum()).round(2)})
w(t.to_markdown()); w("")
low_ent = ann[ann.sample_entropy_norm < 0.8]
if len(low_ent):
    w("Clusters concentrated in few sections (normalised sample entropy < 0.8), i.e. condition- or batch-specific states:\n")
    tt = low_ent[["cell_type", "n_cells", "sample_entropy_norm", "top1", "top2", "top3", "top4"]].copy()
    for c in ["top1", "top2", "top3", "top4"]: tt[c] = tt[c].map(ital)
    w(tt.round(2).to_markdown()); w("")
w("![UMAP](figures/umap_celltype_sample.png)\n![dotplot](figures/dotplot_markers_by_celltype.png)\n"
  "![composition](figures/celltype_composition_per_sample.png)\n![spatial](figures/spatial_celltype_all_sections.png)\n")

import glob
niche_dirs = sorted(glob.glob(f"{RES}/niches*"))
sec = 3
for N in niche_dirs:
    tag = os.path.basename(N).replace("niches", "")
    sil = pd.read_csv(f"{N}/kmeans_silhouette.csv", index_col=0)
    enr = pd.read_csv(f"{N}/niche_celltype_log2enrichment.csv", index_col=0)
    frac = pd.read_csv(f"{N}/niche_celltype_fraction.csv", index_col=0)
    nfrac = pd.read_csv(f"{N}/niche_fraction_per_sample.csv", index_col=0)
    da = pd.read_csv(f"{N}/niche_differential_abundance.csv")
    k = len(enr)
    w(f"## {sec}. Niches (k = {k}{', forced' if tag else ', silhouette-selected'})\n"); sec += 1
    w(f"Neighbourhood composition = cell-type fractions among the 15 nearest neighbours within a section; "
      f"MiniBatchKMeans, **k = {k}**. Silhouette (30k-cell subsample) by k: "
      + ", ".join(f"k={i}: {v:.3f}" for i, v in sil.silhouette.items())
      + ". Niches are named “dominant cell type | most enriched cell type”.\n")
    rows = []
    for n, r in enr.iterrows():
        top = r.sort_values(ascending=False); fr = frac.loc[n].sort_values(ascending=False)
        rows.append({"niche": n, "cells": int((nfrac[n] * qc.loc[nfrac.index, "n_cells_qc"]).sum()),
                     "dominant (fraction)": ", ".join(f"{i} ({100*v:.0f} %)" for i, v in fr.head(3).items()),
                     "most enriched (log2)": ", ".join(f"{i} ({v:+.1f})" for i, v in top.head(3).items())})
    w(pd.DataFrame(rows).to_markdown(index=False)); w("")
    w(f"![niche enrichment]({os.path.basename(N)}/niche_celltype_enrichment_heatmap.png)\n![niche maps]({os.path.basename(N)}/spatial_niches_all_sections.png)\n")
    w(f"### Niche abundance by condition (k = {k})\n")
    w("Per-section niche fractions compared between groups within each experiment (Welch t-test and Mann–Whitney on "
      f"section-level fractions; n is small, so treat p-values as descriptive). Full table: `{os.path.basename(N)}/niche_differential_abundance.csv`.\n")
    for expt, g1, g2 in [("no_recovery", "CupRap", "Control"), ("recovery", "CupRap", "Control"), ("infusion", "OSM_infused", "BSA_infused")]:
        d = da[da.experiment == expt].copy()
        if d.empty: continue
        d = d.sort_values("p_welch")
        w(f"**{expt}: {g1} (n={int(d.n1.iloc[0])}) vs {g2} (n={int(d.n2.iloc[0])})**\n")
        t = d[["niche", "mean1", "mean2", "log2FC", "p_welch", "p_mwu"]].copy()
        t["mean1"] = (100 * t.mean1).round(2); t["mean2"] = (100 * t.mean2).round(2)
        t.columns = ["niche", f"% {g1}", f"% {g2}", "log2FC", "p Welch", "p MWU"]
        w(t.round(3).to_markdown(index=False)); w("")
    w(f"![niche abundance]({os.path.basename(N)}/niche_abundance_per_sample.png)\n![niche log2FC]({os.path.basename(N)}/niche_log2fc_by_experiment.png)\n")
    if not tag:
        w(f"### Cell-type neighbourhood enrichment\n")
        w("squidpy `nhood_enrichment` z-scores computed per section (15-NN graph, 200 permutations), averaged per condition; "
          f"the right column shows the difference between groups. Per-section matrices: `{os.path.basename(N)}/nhood_enrichment_z_<section>.csv`.\n")
        w(f"![nhood]({os.path.basename(N)}/nhood_enrichment_by_condition.png)\n")

w(f"## {sec}. Cell-type composition by condition\n")
comp = ctf.join(samples[["experiment", "condition"]])
for expt, g1, g2 in [("no_recovery", "CupRap", "Control"), ("recovery", "CupRap", "Control"), ("infusion", "OSM_infused", "BSA_infused")]:
    c = comp[comp.experiment == expt]
    if c.empty: continue
    m = c.groupby("condition")[ctf.columns].mean().T[[g1, g2]]   # explicit column order (groupby sorts alphabetically)
    m["log2FC"] = np.log2((m[g1] + 1e-4) / (m[g2] + 1e-4))
    m = (m.sort_values("log2FC", ascending=False))
    m[g1] = (100 * m[g1]).round(2); m[g2] = (100 * m[g2]).round(2)
    m.columns = [f"% {g1}", f"% {g2}", "log2FC"]
    w(f"### {expt}\n"); w(m.round(2).to_markdown()); w("")

open(f"{RES}/REPORT.md", "w").write("\n".join(L))
print(f"wrote {RES}/REPORT.md")
