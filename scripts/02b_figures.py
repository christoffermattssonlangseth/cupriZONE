"""Figures for the preprocessing step, computed from the checkpoint h5ad with a small memory footprint:
UMAP on a 120k-cell subsample, marker dotplot, cell-type composition, spatial cell-type maps."""
import os, sys, time, warnings, json
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common

warnings.filterwarnings("ignore")
ROOT = common.ROOT
RES = os.environ.get("RES_DIR", f"{ROOT}/results"); FIG = f"{RES}/figures"; os.makedirs(FIG, exist_ok=True)
H5 = f"{RES}/xenium_all.h5ad"
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
samples = pd.read_csv(f"{ROOT}/scripts/samples.tsv", sep="\t")

d = common.read_light(H5, keys=("obs", "var"), obsm=("spatial", "X_pca"))
obs, var, spatial, pca = d["obs"], d["var"], d["obsm"]["spatial"], d["obsm"]["X_pca"]
obs = common.apply_overrides(obs, log=log)
order = [s for s in samples.sample_id if s in set(obs.sample_id.astype(str))]
obs["sample_id"] = pd.Categorical(obs.sample_id.astype(str), categories=order)
MARKERS = json.load(open(f"{RES}/markers_used.json"))
log(f"obs loaded: {len(obs):,} cells")

pd.crosstab(obs.cell_type, obs.sample_id).to_csv(f"{RES}/celltype_counts_per_sample.csv")
comp = pd.crosstab(obs.sample_id, obs.cell_type, normalize="index").loc[order]
comp.to_csv(f"{RES}/celltype_fractions_per_sample.csv")

cmap = viz.cmap_for(obs.cell_type.cat.categories)
json.dump(cmap, open(f"{RES}/celltype_colors.json", "w"), indent=1)

# ---- subsample -> UMAP
rng = np.random.default_rng(0)
X_sub, idx = common.read_rows(H5, rng.choice(len(obs), size=min(120_000, len(obs)), replace=False))
sub = ad.AnnData(X=X_sub, obs=obs.iloc[idx].copy(), var=var, obsm={"X_pca": pca[idx]})
sc.pp.neighbors(sub, use_rep="X_pca", n_neighbors=15, random_state=0)
sc.tl.umap(sub, random_state=0)
log("UMAP done")
xy = sub.obsm["X_umap"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(xy[:, 0], xy[:, 1], s=0.3, c=sub.obs.cell_type.map(cmap), lw=0, rasterized=True)
axes[0].set_title(f"UMAP of {sub.n_obs:,} subsampled cells, coloured by annotated cell type")
viz.legend_outside(axes[0], cmap, "cell type")
scmap = viz.cmap_for(order)
axes[1].scatter(xy[:, 0], xy[:, 1], s=0.3, c=sub.obs.sample_id.map(scmap), lw=0, rasterized=True)
axes[1].set_title("same cells, coloured by section")
viz.legend_outside(axes[1], scmap, "section")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
fig.savefig(f"{FIG}/umap_celltype_sample.png"); plt.close(fig)
pd.DataFrame(xy, index=sub.obs_names, columns=["UMAP1", "UMAP2"]).to_csv(f"{RES}/umap_subsample_120k.csv.gz")

# ---- marker dotplot
genes = list(dict.fromkeys(g for v in MARKERS.values() for g in v[:4] if g in sub.var_names))
dp = sc.pl.dotplot(sub, genes, groupby="cell_type", standard_scale="var", show=False, return_fig=True,
                   cmap=viz.seq_cmap(), figsize=(max(10, len(genes) * 0.22), 6))
dp.savefig(f"{FIG}/dotplot_markers_by_celltype.png"); plt.close("all")
log("dotplot done")

# ---- composition per section
fig, ax = plt.subplots(figsize=(11, 4.5))
bottom = np.zeros(len(comp))
for ct_ in comp.columns:
    ax.bar(range(len(comp)), comp[ct_], bottom=bottom, color=cmap[ct_], width=0.8, edgecolor=viz.SURFACE, lw=1)
    bottom += comp[ct_].values
ax.set_xticks(range(len(comp))); ax.set_xticklabels(comp.index, rotation=60, ha="right")
ax.set_ylabel("fraction of cells"); ax.set_title("Cell-type composition per section")
viz.legend_outside(ax, cmap, "cell type")
fig.savefig(f"{FIG}/celltype_composition_per_sample.png"); plt.close(fig)

# ---- spatial maps
n = len(order); ncol = 5; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.2), squeeze=False)
for ax, s in zip(axes.ravel(), order):
    m = (obs.sample_id == s).values
    ax.scatter(spatial[m, 0], -spatial[m, 1], s=0.15, c=obs.loc[m, "cell_type"].map(cmap), lw=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{s} (n={m.sum():,})", fontsize=8)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
for ax in axes.ravel()[n:]: ax.axis("off")
viz.legend_outside(axes[0, -1], cmap, "cell type")
fig.suptitle("Annotated cell types in space (one panel per section)", x=0.02, ha="left")
fig.savefig(f"{FIG}/spatial_celltype_all_sections.png"); plt.close(fig)
log("done")
