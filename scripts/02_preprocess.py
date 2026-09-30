"""Preprocess the 19 Xenium sections of GSE266690 (mouse forebrain; control vs cuprizone-rapamycin,
BSA vs OSM infusion): load 10x-style matrices + Xenium cell metadata, QC, normalise, PCA, kNN,
Leiden, marker-based cluster annotation, UMAP on a subsample. Writes results/xenium_all.h5ad."""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz

warnings.filterwarnings("ignore")
sc.settings.n_jobs = int(os.environ.get("NJOBS", "6"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA, RES, FIG = f"{ROOT}/data/geo", f"{ROOT}/results", f"{ROOT}/results/figures"
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

# QC thresholds (Xenium conventions: 10x / squidpy tutorials use >=10 transcripts; Salas et al. 2023
# recommend ~10-20). Area outliers are trimmed at the 1st/99th percentile per section.
MIN_TRANSCRIPTS, MIN_GENES = 20, 5
MIN_CELLS_PER_GENE = 10
LEIDEN_RES = float(os.environ.get("LEIDEN_RES", "1.0"))

samples = pd.read_csv(f"{ROOT}/scripts/samples.tsv", sep="\t")
if os.environ.get("LIMIT_SAMPLES"):  # smoke-test mode
    samples = samples.iloc[[int(i) for i in os.environ["LIMIT_SAMPLES"].split(",")]].reset_index(drop=True)
TEST_SUB = int(os.environ.get("TEST_SUBSAMPLE", "0"))
RES = os.environ.get("RES_DIR", RES); FIG = f"{RES}/figures"; os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------- load
adatas, qc_rows = [], []
for _, r in samples.iterrows():
    d = f"{DATA}/{r.sample_id}"
    a = sc.read_10x_mtx(d, var_names="gene_symbols", make_unique=True)  # matrix/barcodes/features
    a.var["feature_type"] = a.var["feature_types"] if "feature_types" in a.var else "Gene Expression"
    cells = pd.read_csv(f"{d}/cells.csv.gz", index_col="cell_id")
    a = a[cells.index.intersection(a.obs_names)].copy()
    if TEST_SUB and a.n_obs > TEST_SUB:
        sc.pp.subsample(a, n_obs=TEST_SUB, random_state=0)
    cells = cells.loc[a.obs_names]
    for c in cells.columns:
        a.obs[c] = cells[c].values
    a.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].to_numpy()
    for k in ["gsm", "sample_id", "condition", "experiment", "replicate"]:
        a.obs[k] = str(r[k])
    metrics = pd.read_csv(f"{d}/metrics_summary.csv.gz").iloc[0].to_dict()
    qc_rows.append({"sample_id": r.sample_id, "n_cells_raw": a.n_obs,
                    "n_genes_panel": int((a.var.feature_type == "Gene Expression").sum()),
                    "neg_probe_rate": metrics.get("adjusted_negative_control_probe_rate"),
                    "neg_codeword_rate": metrics.get("adjusted_negative_control_codeword_rate"),
                    "median_transcripts_per_cell": metrics.get("median_transcripts_per_cell"),
                    "median_genes_per_cell": metrics.get("median_genes_per_cell"),
                    "region_area_um2": metrics.get("region_area"), "panel": metrics.get("panel_name")})
    # keep only gene features (drop negative controls / unassigned codewords)
    a = a[:, a.var.feature_type == "Gene Expression"].copy()
    a.obs_names = [f"{r.sample_id}:{c}" for c in a.obs_names]
    adatas.append(a)
    log(f"loaded {r.sample_id}: {a.n_obs} cells x {a.n_vars} genes")

gene_sets = [set(a.var_names) for a in adatas]
common = sorted(set.intersection(*gene_sets))
union = set.union(*gene_sets)
log(f"genes: common={len(common)}, union={len(union)}; dropped per-sample: "
    + ", ".join(f"{s}:{len(set(a.var_names)-set(common))}" for s, a in zip(samples.sample_id, adatas)))
pd.Series(sorted(union - set(common))).to_csv(f"{RES}/genes_not_in_all_samples.csv", index=False, header=["gene"])
adatas = [a[:, common].copy() for a in adatas]
adata = ad.concat(adatas, join="inner", merge="first", index_unique=None)
adata.var["gene_ids"] = adatas[0].var.loc[common, "gene_ids"].values
del adatas
adata.obs["sample_id"] = pd.Categorical(adata.obs["sample_id"], categories=samples.sample_id.tolist())
log(f"concatenated: {adata.shape}")

# ----------------------------------------------------------------------------- QC
adata.layers["counts"] = adata.X.copy()
sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
adata.obs["n_transcripts"] = np.asarray(adata.X.sum(1)).ravel()
adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(1)).ravel()
lo = adata.obs.groupby("sample_id", observed=True)["cell_area"].transform(lambda x: x.quantile(0.01))
hi = adata.obs.groupby("sample_id", observed=True)["cell_area"].transform(lambda x: x.quantile(0.99))
keep = ((adata.obs.n_transcripts >= MIN_TRANSCRIPTS) & (adata.obs.n_genes >= MIN_GENES)
        & (adata.obs.cell_area >= lo) & (adata.obs.cell_area <= hi))
adata.obs["qc_pass"] = keep.values

qc = pd.DataFrame(qc_rows).set_index("sample_id")
qc["n_cells_qc"] = adata.obs.groupby("sample_id", observed=True)["qc_pass"].sum()
qc["frac_removed"] = 1 - qc.n_cells_qc / qc.n_cells_raw
qc["median_transcripts_qc"] = adata.obs[keep].groupby("sample_id", observed=True)["n_transcripts"].median()
qc["median_genes_qc"] = adata.obs[keep].groupby("sample_id", observed=True)["n_genes"].median()
qc["median_cell_area_qc"] = adata.obs[keep].groupby("sample_id", observed=True)["cell_area"].median()
qc = qc.join(samples.set_index("sample_id")[["gsm", "condition", "experiment", "replicate"]])
qc.to_csv(f"{RES}/qc_per_sample.csv")
log("QC table:\n" + qc[["n_cells_raw", "n_cells_qc", "frac_removed", "median_transcripts_qc", "neg_probe_rate"]].to_string())

# QC figure: per-sample distributions (violins are fine here: distributions, one hue = one metric)
fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
order = samples.sample_id.tolist()
for ax, (col, lab, ylog) in zip(axes, [("n_transcripts", "transcripts / cell", True),
                                       ("n_genes", "genes / cell", False), ("cell_area", "cell area (µm²)", True)]):
    data = [adata.obs.loc[adata.obs.sample_id == s, col].values for s in order]
    parts = ax.violinplot([np.log10(d + 1) if ylog else d for d in data], showmedians=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(viz.CAT8[0]); pc.set_alpha(0.7); pc.set_edgecolor("none")
    for k in ["cbars", "cmins", "cmaxes", "cmedians"]:
        parts[k].set_color(viz.INK2); parts[k].set_linewidth(0.8)
    ax.set_ylabel(("log10 " if ylog else "") + lab)
    if col == "n_transcripts":
        ax.axhline(np.log10(MIN_TRANSCRIPTS + 1), color=viz.CAT8[7], lw=1, ls="--")
axes[-1].set_xticks(range(1, len(order) + 1)); axes[-1].set_xticklabels(order, rotation=60, ha="right")
fig.suptitle("Xenium QC per section (before filtering; dashed = transcript threshold)", x=0.02, ha="left")
fig.savefig(f"{FIG}/qc_violin_per_sample.png"); plt.close(fig)

adata = adata[adata.obs.qc_pass].copy()
sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
log(f"after QC: {adata.shape}")

# ----------------------------------------------------------------------------- normalise / embed
sc.pp.normalize_total(adata, target_sum=100)   # Xenium: fixed small target keeps counts scale sane
sc.pp.log1p(adata)
adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
# PCA on the sparse log-normalised matrix with implicit centering (no dense scaled copy: memory)
sc.pp.pca(adata, n_comps=30, zero_center=True, random_state=0)
log("PCA done")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=0)
del adata.obsp["distances"]
log("neighbors done")
sc.tl.leiden(adata, resolution=LEIDEN_RES, flavor="igraph", n_iterations=2, directed=False,
             random_state=0, key_added="leiden")
log(f"leiden done: {adata.obs.leiden.nunique()} clusters")

# ----------------------------------------------------------------------------- annotation
MARKERS = {
    "Oligodendrocyte": ["Mbp", "Mog", "Mag", "Opalin", "Gjc3", "Plp1", "Mobp", "Cldn11", "Aspa"],
    "OPC": ["Pdgfra", "Cspg4", "Olig1", "Olig2", "Vcan", "Ptprz1"],
    "COP-NFOL": ["Enpp6", "Gpr17", "Bcas1", "Tcf7l2", "Fyn"],
    "Astrocyte": ["Aqp4", "Gfap", "Ntsr2", "Slc39a12", "Rfx4", "Slc1a2", "Slc1a3", "Aldoc", "Gjb6", "S100b", "Gja1"],
    "Microglia": ["Cx3cr1", "Tmem119", "Siglech", "Aif1", "Laptm5", "P2ry12", "Csf1r", "Hexb", "Spi1"],
    "Microglia activated / DAM": ["Trem2", "Cd68", "Lgals3", "Spp1", "Cybb", "Lyz2", "Igf1", "Cd63"],
    "Macrophage-BAM": ["H2-Eb1", "Mrc1", "Lyve1", "Cd74", "Pf4", "F13a1"],
    "Endothelial": ["Cldn5", "Pecam1", "Kdr", "Ly6a", "Emcn", "Adgrl4", "Cd93", "Ecscr", "Nostrin", "Sox17", "Acvrl1", "Flt1"],
    "Pericyte-VSMC": ["Acta2", "Carmn", "Pln", "Pdgfrb", "Vtn", "Kcnj8", "Rgs5", "Myh11"],
    "VLMC-Fibroblast": ["Dcn", "Col1a1", "Col6a1", "Fmod", "Fn1", "Igfbp6", "Cyp1b1", "Aldh1a2", "Lum", "Col1a2"],
    "Ependymal": ["Foxj1", "Trp73", "Spag16", "Ccdc153", "Dynlrb2", "Tmem212"],
    "Choroid plexus": ["Ttr", "Clic6", "Kcnj13", "Kl", "Folr1"],
    "Astrocyte Gfap-high / qNSC": ["Gfap", "Thbs4", "Nes", "Gli3", "Prom1", "Sox2", "Hopx", "Id3"],
    "NSC (activated)-TAP": ["Ascl1", "Egfr", "Mki67", "Rrm2", "Pclaf", "Hells", "Hat1", "Gsx2", "Top2a", "Cdk1"],
    "Neuroblast": ["Dlx2", "Dlx1", "Sox11", "Sp8", "Sp9", "Cd24a", "Tubb3", "Igfbpl1", "Insm1", "Dcx", "Stmn2"],
    "Excitatory neuron": ["Slc17a7", "Satb2", "Neurod6", "Cux2", "Rorb", "Fezf2", "Foxp2", "Tle4", "Slc17a6", "Neurod2", "Camk2a"],
    "Inhibitory neuron": ["Gad1", "Gad2", "Pvalb", "Sst", "Vip", "Cort", "Slc32a1", "Npy", "Lhx6"],
    "Striatal MSN": ["Ppp1r1b", "Penk", "Pdyn", "Drd1", "Drd2", "Adora2a", "Tac1"],
    "T cell-lymphocyte": ["Il7r", "Trbc2", "Ikzf1", "Cd3e", "Cd3g", "Ptprc", "Cd52"],
}
MARKERS = {k: [g for g in v if g in adata.var_names] for k, v in MARKERS.items()}
MARKERS = {k: v for k, v in MARKERS.items() if len(v) >= 2}
json.dump(MARKERS, open(f"{RES}/markers_used.json", "w"), indent=1)
log("marker sets (genes on panel): " + ", ".join(f"{k}:{len(v)}" for k, v in MARKERS.items()))

# cluster x gene mean expression (expm1 of log-normalised = normalised counts), compared with the
# overall mean: score(set) = mean over genes of log2((cluster mean + eps) / (overall mean + eps)).
clusters = adata.obs.leiden.astype(str)
cl_ids = sorted(clusters.unique(), key=int)
Xn = adata.X.copy(); Xn.data = np.expm1(Xn.data)          # normalised counts (sparse)
onehot = sp.csr_matrix(pd.get_dummies(clusters)[cl_ids].values.astype(np.float32))
means = np.asarray((onehot.T @ Xn).todense()) / np.asarray(onehot.sum(0)).T
means = pd.DataFrame(means, index=cl_ids, columns=adata.var_names)
overall = np.asarray(Xn.mean(0)).ravel()
del Xn, onehot
lfc = np.log2((means + 0.05) / (overall + 0.05))
# set score = mean of the 3 highest log2FCs in the set (marker sets mix pan- and subtype-specific genes)
scores = pd.DataFrame({k: lfc[v].apply(lambda r: np.sort(r.values)[-min(3, len(v)):].mean(), axis=1) for k, v in MARKERS.items()})
PAN_NEURON = [g for g in ["Elavl2", "Elavl4", "Meg3", "Nrn1", "Rab3b", "Slc17a6", "Slc17a7", "Gad1", "Gad2",
                          "Cpne4", "Nwd2", "Calb2", "Nr2f2", "Fezf2", "Neurod6", "Syt2", "Snap25", "Rbfox3"] if g in adata.var_names]
scores["Neuron (other)"] = lfc[PAN_NEURON].apply(lambda r: np.sort(r.values)[-4:].mean(), axis=1)  # top-4 pan-neuronal genes
scores.to_csv(f"{RES}/cluster_marker_scores.csv")
prim = scores.drop(columns="Neuron (other)")
top = prim.idxmax(1)
srt = np.sort(prim.values, 1)
margin = srt[:, -1] - srt[:, -2]
annot = {}
for i, c in enumerate(cl_ids):
    if srt[i, -1] > 0.75 and (margin[i] > 0.25 or srt[i, -1] > 2.0):
        annot[c] = top[c]
    elif scores.loc[c, "Neuron (other)"] > 0.75:
        annot[c] = "Neuron (other)"
    else:
        annot[c] = "Unassigned"
adata.obs["cell_type"] = clusters.map(annot).astype("category")
adata.obs["cell_type_cluster"] = (clusters.map(annot) + " (c" + clusters + ")").astype("category")
adata.write_h5ad(f"{RES}/xenium_all.h5ad", compression="lzf")   # checkpoint (figures below are re-creatable)
log("checkpoint written")
de_sub = sc.pp.subsample(adata, n_obs=min(300_000, adata.n_obs), random_state=0, copy=True)
sc.tl.rank_genes_groups(de_sub, "leiden", method="wilcoxon", n_genes=10)
deg = pd.DataFrame({c: [g for g in de_sub.uns["rank_genes_groups"]["names"][c]] for c in cl_ids}).T
del de_sub
deg.columns = [f"top{i+1}" for i in range(deg.shape[1])]
ann_tab = pd.DataFrame({"leiden": cl_ids, "cell_type": [annot[c] for c in cl_ids],
                        "top_set_score": srt[:, -1], "margin": margin,
                        "n_cells": [int((clusters == c).sum()) for c in cl_ids]}).set_index("leiden").join(deg)
ann_tab.to_csv(f"{RES}/cluster_annotation.csv")
log("cluster annotation:\n" + ann_tab[["cell_type", "n_cells", "top_set_score", "top1", "top2", "top3", "top4"]].to_string())

# batch mixing check: per cluster, fraction of cells from each sample (entropy vs. expected)
ct = pd.crosstab(adata.obs.leiden, adata.obs.sample_id)
p = ct.div(ct.sum(1), axis=0)
ent = -(p * np.log(p + 1e-12)).sum(1) / np.log(ct.shape[1])
ann_tab["sample_entropy_norm"] = ent
ann_tab.to_csv(f"{RES}/cluster_annotation.csv")
pd.crosstab(adata.obs.cell_type, adata.obs.sample_id).to_csv(f"{RES}/celltype_counts_per_sample.csv")

# ----------------------------------------------------------------------------- figures
sub = sc.pp.subsample(adata, n_obs=min(150_000, adata.n_obs), random_state=0, copy=True)
sc.tl.umap(sub, random_state=0)
adata.uns["umap_subsample_obs"] = list(sub.obs_names)
cmap = viz.cmap_for(adata.obs.cell_type.cat.categories)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
xy = sub.obsm["X_umap"]
axes[0].scatter(xy[:, 0], xy[:, 1], s=0.3, c=sub.obs.cell_type.map(cmap), lw=0, rasterized=True)
axes[0].set_title(f"UMAP of {sub.n_obs:,} subsampled cells, coloured by annotated cell type")
viz.legend_outside(axes[0], cmap, "cell type")
scmap = viz.cmap_for(samples.sample_id)
axes[1].scatter(xy[:, 0], xy[:, 1], s=0.3, c=sub.obs.sample_id.map(scmap), lw=0, rasterized=True)
axes[1].set_title("same cells, coloured by section")
viz.legend_outside(axes[1], scmap, "section", ncol=1)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
fig.savefig(f"{FIG}/umap_celltype_sample.png"); plt.close(fig)

# marker dotplot (mean expression per cell type)
genes = [g for v in MARKERS.values() for g in v[:4]]
genes = list(dict.fromkeys(genes))
dp = sc.pl.dotplot(sub, genes, groupby="cell_type", standard_scale="var", show=False,
                   return_fig=True, cmap=viz.seq_cmap(), figsize=(max(10, len(genes) * 0.22), 6))
dp.savefig(f"{FIG}/dotplot_markers_by_celltype.png"); plt.close("all")

# cell type composition per sample (stacked bars, grouped by experiment/condition)
comp = pd.crosstab(adata.obs.sample_id, adata.obs.cell_type, normalize="index").loc[order]
comp.to_csv(f"{RES}/celltype_fractions_per_sample.csv")
fig, ax = plt.subplots(figsize=(11, 4.5))
bottom = np.zeros(len(comp))
for ct_ in comp.columns:
    ax.bar(range(len(comp)), comp[ct_], bottom=bottom, color=cmap[ct_], width=0.8, edgecolor=viz.SURFACE, lw=1)
    bottom += comp[ct_].values
ax.set_xticks(range(len(comp))); ax.set_xticklabels(comp.index, rotation=60, ha="right")
ax.set_ylabel("fraction of cells"); ax.set_title("Cell-type composition per section")
viz.legend_outside(ax, cmap, "cell type")
fig.savefig(f"{FIG}/celltype_composition_per_sample.png"); plt.close(fig)

# spatial maps: one panel per section
def spatial_grid(adata, key, cmap, fname, title):
    n = len(order); ncol = 5; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.2), squeeze=False)
    for ax, s in zip(axes.ravel(), order):
        m = (adata.obs.sample_id == s).values
        xy = adata.obsm["spatial"][m]
        ax.scatter(xy[:, 0], -xy[:, 1], s=0.15, c=adata.obs.loc[m, key].map(cmap), lw=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{s} (n={m.sum():,})", fontsize=8)
        for sp_ in ax.spines.values(): sp_.set_visible(False)
    for ax in axes.ravel()[n:]: ax.axis("off")
    viz.legend_outside(axes[0, -1], cmap, key)
    fig.suptitle(title, x=0.02, ha="left")
    fig.savefig(fname); plt.close(fig)

spatial_grid(adata, "cell_type", cmap, f"{FIG}/spatial_celltype_all_sections.png",
             "Annotated cell types in space (one panel per section)")

adata.uns["celltype_colors"] = {"categories": list(cmap.keys()), "colors": list(cmap.values())}
adata.write_h5ad(f"{RES}/xenium_all.h5ad", compression="lzf")
log(f"wrote {RES}/xenium_all.h5ad {adata.shape}; done")
