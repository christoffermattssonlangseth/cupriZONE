"""MERSCOPE (4 recovery sections): QC, normalise, PCA, kNN, Leiden, marker annotation, UMAP, figures.
Mirrors 02_preprocess.py but for the Vizgen 300-gene panel. Writes results/merscope/merscope_processed.h5ad."""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, scipy.sparse as sp
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz
warnings.filterwarnings("ignore")
sc.settings.n_jobs = 6
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = f"{ROOT}/results/merscope"; FIG = f"{RES}/figures"; os.makedirs(FIG, exist_ok=True)
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
MIN_TX, MIN_GENES = 15, 5
samples = pd.read_csv(f"{ROOT}/scripts/samples_merscope.tsv", sep="\t"); order = samples.sample_id.tolist()

adata = ad.read_h5ad(f"{RES}/merscope_recovery_raw.h5ad")
adata.obs["sample_id"] = pd.Categorical(adata.obs.sample_id.astype(str), categories=order)
lo = adata.obs.groupby("sample_id", observed=True)["volume"].transform(lambda x: x.quantile(0.01))
hi = adata.obs.groupby("sample_id", observed=True)["volume"].transform(lambda x: x.quantile(0.99))
keep = (adata.obs.n_transcripts >= MIN_TX) & (adata.obs.n_genes >= MIN_GENES) & (adata.obs.volume >= lo) & (adata.obs.volume <= hi)
qc = adata.obs.groupby("sample_id", observed=True).agg(n_cells_raw=("n_transcripts", "size"), median_tx_raw=("n_transcripts", "median"))
qc["n_cells_qc"] = keep.groupby(adata.obs.sample_id, observed=True).sum(); qc["frac_removed"] = 1 - qc.n_cells_qc / qc.n_cells_raw
adata = adata[keep.values].copy()
qc["median_tx_qc"] = adata.obs.groupby("sample_id", observed=True)["n_transcripts"].median()
qc["median_genes_qc"] = adata.obs.groupby("sample_id", observed=True)["n_genes"].median()
qc["median_volume_qc"] = adata.obs.groupby("sample_id", observed=True)["volume"].median()
qc = qc.join(samples.set_index("sample_id")[["gsm", "condition"]]); qc.to_csv(f"{RES}/qc_per_sample.csv")
log("QC:\n" + qc.round(3).to_string())

fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
for ax, (col, lab) in zip(axes, [("n_transcripts", "log10 transcripts / cell"), ("volume", "log10 volume (µm³)")]):
    data = [np.log10(adata.obs.loc[adata.obs.sample_id == s, col].values + 1) for s in order]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc in parts["bodies"]: pc.set_facecolor(viz.CAT8[0]); pc.set_alpha(0.7); pc.set_edgecolor("none")
    for k in ["cbars", "cmins", "cmaxes", "cmedians"]: parts[k].set_color(viz.INK2); parts[k].set_linewidth(0.8)
    ax.set_ylabel(lab)
axes[-1].set_xticks(range(1, len(order) + 1)); axes[-1].set_xticklabels(order, rotation=45, ha="right")
fig.suptitle("MERSCOPE QC per section (after filtering)", x=0.02, ha="left"); fig.savefig(f"{FIG}/qc_violin_per_sample.png"); plt.close(fig)

sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, target_sum=100); sc.pp.log1p(adata)
adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
sc.pp.pca(adata, n_comps=30, zero_center=True, random_state=0)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=0); del adata.obsp["distances"]
sc.tl.leiden(adata, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, random_state=0)
log(f"leiden: {adata.obs.leiden.nunique()} clusters")

MARKERS = {
    "Oligodendrocyte": ["Mbp", "Mog", "Mag", "Sox10", "Enpp6"],
    "OPC": ["Pdgfra", "Cspg4", "Ptprz1", "Olig1", "Olig2", "Sox10"],
    "Astrocyte": ["Aqp4", "Gfap", "Aldoc", "Gja1", "S100b", "Gjb6", "Fam107a"],
    "Astrocyte Gfap-high / qNSC": ["Gfap", "Thbs4", "Nes", "Sox3", "Hes5", "Id3"],
    "NSC (activated)-TAP": ["Ascl1", "Egfr", "Mki67", "Rrm2", "Pclaf", "Hells", "Gsx2"],
    "Neuroblast": ["Dlx1", "Dlx2", "Sp8", "Sp9", "Cd24a", "Tubb3", "Stmn2", "Insm1", "Igfbpl1"],
    "Microglia": ["Cx3cr1", "Tmem119", "Csf1r", "Aif1"],
    "Microglia activated / DAM": ["Trem2", "Cd68", "Lgals3", "Cybb", "Lyz2", "H2-Eb1", "Igf1", "Apoc4"],
    "Endothelial": ["Pecam1", "Cdh5", "Esam", "Plvap", "Tek", "Tie1", "Acvrl1"],
    "Pericyte-VSMC": ["Acta2", "Myh11", "Mylk", "Des", "Notch3"],
    "VLMC-Fibroblast": ["Col2a1", "Cpz", "Fgf7", "Slc22a6", "Itih5", "Gjb2"],
    "Ependymal": ["Foxj1", "Ecrg4", "Hsd11b1"],
    "Choroid plexus": ["Ttr", "Kcnj13"],
    "Excitatory neuron": ["Satb2", "Cux2", "Fezf2", "Tbr1", "Neurod2", "Neurod1", "Rbfox3"],
    "Inhibitory neuron": ["Gad1", "Gad2", "Pvalb"],
    "Striatal MSN": ["Ebf1", "Isl1", "Gad1", "Gad2", "Rbfox3"],
    "T cell-lymphocyte": ["Il7r", "Cd52"],
}
MARKERS = {k: [g for g in v if g in adata.var_names] for k, v in MARKERS.items()}
MARKERS = {k: v for k, v in MARKERS.items() if len(v) >= 2}
json.dump(MARKERS, open(f"{RES}/markers_used.json", "w"), indent=1)
clusters = adata.obs.leiden.astype(str); cl_ids = sorted(clusters.unique(), key=int)
Xn = adata.X.copy(); Xn.data = np.expm1(Xn.data)
onehot = sp.csr_matrix(pd.get_dummies(clusters)[cl_ids].values.astype(np.float32))
means = pd.DataFrame(np.asarray((onehot.T @ Xn).todense()) / np.asarray(onehot.sum(0)).T, index=cl_ids, columns=adata.var_names)
overall = np.asarray(Xn.mean(0)).ravel(); del Xn
lfc = np.log2((means + 0.05) / (overall + 0.05))
scores = pd.DataFrame({k: lfc[v].apply(lambda r: np.sort(r.values)[-min(3, len(v)):].mean(), axis=1) for k, v in MARKERS.items()})
PAN = [g for g in ["Elavl2", "Elavl4", "Rbfox3", "Stmn2", "Gad1", "Gad2", "Satb2", "Fezf2", "Tbr1", "Nsg2", "Dpysl5"] if g in adata.var_names]
scores["Neuron (other)"] = lfc[PAN].apply(lambda r: np.sort(r.values)[-4:].mean(), axis=1)
scores.to_csv(f"{RES}/cluster_marker_scores.csv")
prim = scores.drop(columns="Neuron (other)"); top = prim.idxmax(1); srt = np.sort(prim.values, 1); margin = srt[:, -1] - srt[:, -2]
annot = {c: (top[c] if (srt[i, -1] > 0.75 and (margin[i] > 0.25 or srt[i, -1] > 2.0)) else
             ("Neuron (other)" if scores.loc[c, "Neuron (other)"] > 0.75 else "Unassigned")) for i, c in enumerate(cl_ids)}
adata.obs["cell_type"] = clusters.map(annot).astype("category")
de = sc.pp.subsample(adata, n_obs=min(200_000, adata.n_obs), random_state=0, copy=True)
sc.tl.rank_genes_groups(de, "leiden", method="wilcoxon", n_genes=8)
deg = pd.DataFrame({c: list(de.uns["rank_genes_groups"]["names"][c]) for c in cl_ids}).T; deg.columns = [f"top{i+1}" for i in range(deg.shape[1])]
ct = pd.crosstab(adata.obs.leiden, adata.obs.sample_id); p = ct.div(ct.sum(1), axis=0)
ann = pd.DataFrame({"leiden": cl_ids, "cell_type": [annot[c] for c in cl_ids], "top_set_score": srt[:, -1], "margin": margin,
                    "n_cells": [int((clusters == c).sum()) for c in cl_ids]}).set_index("leiden").join(deg)
ann["sample_entropy_norm"] = (-(p * np.log(p + 1e-12)).sum(1) / np.log(ct.shape[1])).values
ann["frac_CupRap"] = ct[[s for s in order if "CupRap" in s]].sum(1).div(ct.sum(1)).values
ann.to_csv(f"{RES}/cluster_annotation.csv")
log("annotation:\n" + ann[["cell_type", "n_cells", "top_set_score", "frac_CupRap", "top1", "top2", "top3", "top4"]].round(2).to_string())
pd.crosstab(adata.obs.sample_id, adata.obs.cell_type, normalize="index").to_csv(f"{RES}/celltype_fractions_per_sample.csv")
pd.crosstab(adata.obs.cell_type, adata.obs.sample_id).to_csv(f"{RES}/celltype_counts_per_sample.csv")

cmap_x = json.load(open(f"{ROOT}/results/celltype_colors.json"))   # reuse Xenium colours where names match
cats = list(adata.obs.cell_type.cat.categories); extra = viz.cmap_for([c for c in cats if c not in cmap_x])
cmap = {c: cmap_x.get(c, extra.get(c)) for c in cats}; json.dump(cmap, open(f"{RES}/celltype_colors.json", "w"), indent=1)
sc.tl.umap(de, random_state=0); xy = de.obsm["X_umap"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(xy[:, 0], xy[:, 1], s=0.3, c=de.obs.cell_type.map(cmap), lw=0, rasterized=True); viz.legend_outside(axes[0], cmap, "cell type")
scm = viz.cmap_for(order); axes[1].scatter(xy[:, 0], xy[:, 1], s=0.3, c=de.obs.sample_id.map(scm), lw=0, rasterized=True); viz.legend_outside(axes[1], scm, "section")
axes[0].set_title(f"MERSCOPE UMAP ({de.n_obs:,} cells), cell type"); axes[1].set_title("section")
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
fig.savefig(f"{FIG}/umap_celltype_sample.png"); plt.close(fig)
genes = list(dict.fromkeys(g for v in MARKERS.values() for g in v[:4]))
dp = sc.pl.dotplot(de, genes, groupby="cell_type", standard_scale="var", show=False, return_fig=True, cmap=viz.seq_cmap(), figsize=(max(10, len(genes) * 0.22), 6))
dp.savefig(f"{FIG}/dotplot_markers_by_celltype.png"); plt.close("all")
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), squeeze=False)
for ax, s in zip(axes.ravel(), order):
    m = (adata.obs.sample_id == s).values; sp_ = adata.obsm["spatial"][m]
    ax.scatter(sp_[:, 0], -sp_[:, 1], s=0.15, c=adata.obs.loc[m, "cell_type"].map(cmap), lw=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{s} (n={m.sum():,})", fontsize=8)
    for q in ax.spines.values(): q.set_visible(False)
viz.legend_outside(axes[0, -1], cmap, "cell type"); fig.suptitle("MERSCOPE: annotated cell types in space", x=0.02, ha="left")
fig.savefig(f"{FIG}/spatial_celltype_all_sections.png"); plt.close(fig)
adata.write_h5ad(f"{RES}/merscope_processed.h5ad", compression="lzf"); log(f"wrote merscope_processed.h5ad {adata.shape}")
