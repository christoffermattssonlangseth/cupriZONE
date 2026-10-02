"""Quick query for one gene: detection rate and mean expression by cell type and by section/condition, plus a spatial map
of expressing cells. Usage: GENE=Crh python 10_gene_query.py"""
import os, sys, json
import numpy as np, pandas as pd, h5py
from anndata.io import sparse_dataset
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common
GENE = os.environ.get("GENE", "Crh")
ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/gene_queries"; os.makedirs(OUT, exist_ok=True)
H5 = f"{RES}/xenium_all.h5ad"
d = common.read_light(H5, keys=("obs", "var"), obsm=("spatial",))
obs, var, xy = d["obs"], d["var"], d["obsm"]["spatial"]
gi = var.index.get_loc(GENE)
expr = np.zeros(len(obs), np.float32); counts = np.zeros(len(obs), np.float32)
with h5py.File(H5) as f:
    X = sparse_dataset(f["X"]); C = sparse_dataset(f["layers/counts"])
    for s in range(0, len(obs), 200_000):
        expr[s:s+200_000] = X[s:s+200_000][:, gi].toarray().ravel()
        counts[s:s+200_000] = C[s:s+200_000][:, gi].toarray().ravel()
obs[f"{GENE}_counts"] = counts; obs[f"{GENE}_lognorm"] = expr; obs["pos"] = counts > 0
print(f"{GENE}: {int((counts>0).sum()):,} / {len(obs):,} cells with >=1 transcript ({100*(counts>0).mean():.2f} %); "
      f">=3 transcripts: {int((counts>=3).sum()):,}; total transcripts {int(counts.sum()):,}")
by_ct = obs.groupby("cell_type", observed=True).agg(n=("pos", "size"), pct_pos=("pos", lambda x: 100*x.mean()),
                                                     mean_counts=(f"{GENE}_counts", "mean"), total=(f"{GENE}_counts", "sum")).sort_values("pct_pos", ascending=False)
print("\nby cell type:\n" + by_ct.round(2).to_string())
by_s = obs.groupby(["experiment", "condition", "sample_id"], observed=True).agg(n=("pos", "size"), pct_pos=("pos", lambda x: 100*x.mean()),
                                                                                 n_pos3=(f"{GENE}_counts", lambda x: int((x>=3).sum())), total=(f"{GENE}_counts", "sum"))
print("\nby section:\n" + by_s.round(2).to_string())
by_ct.to_csv(f"{OUT}/{GENE}_by_celltype.csv"); by_s.to_csv(f"{OUT}/{GENE}_by_section.csv")
# high expressers: what are they, where
hi = obs[counts >= 3]
print(f"\ncells with >=3 {GENE} transcripts, by cell type:\n" + hi.cell_type.value_counts().head(8).to_string())
print(f"\n... by CellCharter niche:\n" + hi.niche_cc.value_counts().head(6).to_string())
# spatial map: grey all cells, coloured by counts for positives, one panel per section
order = list(obs.sample_id.cat.categories) if hasattr(obs.sample_id, "cat") else sorted(obs.sample_id.unique())
n = len(order); ncol = 5; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.2), squeeze=False)
for ax, s in zip(axes.ravel(), order):
    m = (obs.sample_id == s).values
    ax.scatter(xy[m, 0], -xy[m, 1], s=0.05, c="#e1e0d9", lw=0, rasterized=True)
    p = m & (counts >= 2)
    ax.scatter(xy[p, 0], -xy[p, 1], s=1.5, c=np.clip(counts[p], 2, 10), cmap=viz.seq_cmap(), vmin=2, vmax=10, lw=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{s} ({obs.condition[m].iloc[0]}): {int(p.sum())} cells", fontsize=8)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
for ax in axes.ravel()[n:]: ax.axis("off")
fig.suptitle(f"{GENE}: cells with >=2 transcripts (colour = count, 2 to 10+)", x=0.02, ha="left")
fig.savefig(f"{OUT}/{GENE}_spatial.png"); plt.close(fig)
