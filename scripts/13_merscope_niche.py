"""MERSCOPE niches: composition k-means (silhouette k) and CellCharter (stability k), both via niche_downstream;
then a cross-platform comparison with the Xenium recovery sections."""
import os, sys, time, json, warnings, logging
import numpy as np, pandas as pd, scanpy as sc, squidpy as sq, anndata as ad, scipy.sparse as sp
import cellcharter as cc
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common, niche_downstream
warnings.filterwarnings("ignore")
for n in ["lightning", "pytorch_lightning", "lightning.pytorch"]: logging.getLogger(n).setLevel(logging.ERROR)
ROOT = common.ROOT; RES = f"{ROOT}/results/merscope"
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
samples = pd.read_csv(f"{ROOT}/scripts/samples_merscope.tsv", sep="\t"); order = samples.sample_id.tolist()
CONTRASTS = [("recovery", "CupRap", "Control")]
adata = ad.read_h5ad(f"{RES}/merscope_processed.h5ad")
adata.obs["sample_id"] = pd.Categorical(adata.obs.sample_id.astype(str), categories=order)
CT = "cell_type"; cts = list(adata.obs[CT].cat.categories)

# ---- composition k-means
if "niche_cc" in adata.obs and not os.environ.get("FORCE_NICHES"):
    log("niches already in obs; skipping to cross-platform comparison")
else:
  if True:
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=15, library_key="sample_id", delaunay=False)
    A = adata.obsp["spatial_connectivities"].tocsr(); A.data[:] = 1
    oh = pd.get_dummies(adata.obs[CT]); comp = np.asarray((A @ sp.csr_matrix(oh.values.astype(np.float32))).todense())
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1; comp /= deg[:, None]
    rng = np.random.default_rng(0); sub = rng.choice(len(comp), min(200_000, len(comp)), replace=False); sil_idx = rng.choice(sub, 30_000, replace=False)
    sil = {}
    for k in range(5, 15):
        km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=8192, n_init=5).fit(comp[sub]); sil[k] = silhouette_score(comp[sil_idx], km.predict(comp[sil_idx]))
    OUT = f"{RES}/niches"; os.makedirs(OUT, exist_ok=True)
    pd.Series(sil, name="silhouette").rename_axis("k").to_csv(f"{OUT}/kmeans_silhouette.csv")
    k = int(os.environ.get("NICHE_K", max(sil, key=sil.get)))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=8192, n_init=10).fit(comp[sub]); lab = km.predict(comp)
    frac = pd.crosstab(lab, adata.obs[CT]).reindex(range(k)).fillna(0); frac = frac.div(frac.sum(1), axis=0)
    names = niche_downstream.name_niches(frac, adata.obs[CT].value_counts(normalize=True), prefix="N")
    adata.obs["niche"] = pd.Categorical([names[l] for l in lab], categories=[names[i] for i in range(k)])
    log(f"k-means niches k={k} (silhouette {sil[k]:.3f})")
    niche_downstream.downstream(adata, "niche", OUT, samples, log=log, CT_KEY=CT, method="k-means on neighbourhood composition", contrasts=CONTRASTS)
    del adata.obsp["spatial_connectivities"], adata.obsp["spatial_distances"]

    # ---- CellCharter
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, library_key="sample_id"); cc.gr.remove_long_links(adata)
    cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep="X_pca", out_key="X_cellcharter", sample_key="sample_id")
    TR = dict(accelerator="cpu", enable_progress_bar=False, enable_model_summary=False, logger=False)
    OUTC = f"{RES}/niches_cellcharter"; os.makedirs(OUTC, exist_ok=True)
    idx = np.sort(rng.choice(adata.n_obs, min(150_000, adata.n_obs), replace=False))
    autok = cc.tl.ClusterAutoK(n_clusters=(5, 15), max_runs=3, model_params=dict(random_state=0, trainer_params=TR))
    autok.fit(adata[idx].copy(), use_rep="X_cellcharter"); kc = int(os.environ.get("CC_K", autok.best_k))
    try:
        cc.pl.autok_stability(autok); plt.gcf().savefig(f"{OUTC}/autok_stability.png"); plt.close("all")
    except Exception as e: log(f"stability plot failed: {e}")
    gmm = cc.tl.Cluster(n_clusters=kc, random_state=0, trainer_params=TR)
    fit_idx = np.sort(rng.choice(adata.n_obs, min(400_000, adata.n_obs), replace=False))
    gmm.fit(adata[fit_idx].copy(), use_rep="X_cellcharter"); labc = np.asarray(gmm.predict(adata, use_rep="X_cellcharter")).astype(int)
    fracc = pd.crosstab(labc, adata.obs[CT]).reindex(range(kc)).fillna(0); fracc = fracc.div(fracc.sum(1), axis=0)
    namesc = niche_downstream.name_niches(fracc, adata.obs[CT].value_counts(normalize=True), prefix="C")
    adata.obs["niche_cc"] = pd.Categorical([namesc[l] for l in labc], categories=[namesc[i] for i in range(kc)])
    json.dump({"method": f"CellCharter: Delaunay, aggregate_neighbors n_layers=3 on 30 PCs, GMM; k by ClusterAutoK (5-15, 3 runs) on {len(idx):,} cells, fit on {len(fit_idx):,}", "k": kc}, open(f"{OUTC}/method.json", "w"), indent=1)
    log(f"CellCharter k={kc}")
    niche_downstream.downstream(adata, "niche_cc", OUTC, samples, log=log, CT_KEY=CT, method="CellCharter GMM on aggregated PCA neighbourhoods", contrasts=CONTRASTS)
    del adata.obsm["X_cellcharter"]
    adata.write_h5ad(f"{RES}/merscope_processed.h5ad", compression="lzf")

# ---- cross-platform comparison: Xenium recovery vs MERSCOPE recovery, cell-type fractions per condition
xen = pd.read_csv(f"{ROOT}/results/celltype_fractions_per_sample.csv", index_col=0)
xen = xen.loc[[s for s in xen.index if s.startswith("Recov_")]]
mer = pd.read_csv(f"{RES}/celltype_fractions_per_sample.csv", index_col=0)
def grp(df): return df.groupby(np.where(df.index.str.contains("CupRap"), "CupRap", "Control")).mean().T * 100
X, M = grp(xen), grp(mer); X.columns = [f"Xenium {c}" for c in X.columns]; M.columns = [f"MERSCOPE {c}" for c in M.columns]
cmp = X.join(M, how="outer").fillna(0)
cmp["Xenium log2FC"] = np.log2((cmp["Xenium CupRap"] + 0.01) / (cmp["Xenium Control"] + 0.01)); cmp["MERSCOPE log2FC"] = np.log2((cmp["MERSCOPE CupRap"] + 0.01) / (cmp["MERSCOPE Control"] + 0.01))
cmp.round(2).to_csv(f"{RES}/crossplatform_celltype_fractions.csv"); log("cross-platform cell-type % and log2FC:\n" + cmp.round(2).to_string())
fig, ax = plt.subplots(figsize=(6, 6)); both = cmp[(cmp["Xenium Control"] + cmp["Xenium CupRap"] > 0) & (cmp["MERSCOPE Control"] + cmp["MERSCOPE CupRap"] > 0)]
ax.axhline(0, color=viz.AXIS); ax.axvline(0, color=viz.AXIS)
ax.scatter(both["Xenium log2FC"], both["MERSCOPE log2FC"], s=40, color=viz.CAT8[0])
for n, r in both.iterrows(): ax.annotate(n, (r["Xenium log2FC"], r["MERSCOPE log2FC"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Xenium: log2(CupRap / Control) cell-type fraction"); ax.set_ylabel("MERSCOPE: log2(CupRap / Control)")
ax.set_title("Recovery time point: cuprizone effect on cell-type composition, two platforms"); fig.savefig(f"{RES}/crossplatform_celltype_log2fc.png"); plt.close(fig)
log("done")
