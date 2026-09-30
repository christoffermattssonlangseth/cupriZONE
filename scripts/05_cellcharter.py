"""CellCharter niches (Varrone et al. 2024): Delaunay spatial graph per section -> neighbourhood aggregation of the
PCA embedding over 3 layers -> Gaussian mixture; the number of clusters is chosen by CellCharter's stability
criterion (ClusterAutoK) on a subsample, the final GMM is fitted on a larger subsample and used to label all cells.
Outputs go to results/niches_cellcharter/ and obs['niche_cc'] in the h5ad."""
import os, sys, time, warnings, json, logging
import numpy as np, pandas as pd, scanpy as sc, squidpy as sq, anndata as ad
import cellcharter as cc
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common, niche_downstream

warnings.filterwarnings("ignore")
for name in ["lightning", "pytorch_lightning", "lightning.pytorch"]:
    logging.getLogger(name).setLevel(logging.ERROR)
ROOT = common.ROOT
RES = os.environ.get("RES_DIR", f"{ROOT}/results"); OUT = f"{RES}/niches_cellcharter"; os.makedirs(OUT, exist_ok=True)
H5 = f"{RES}/xenium_all.h5ad"
NK, NID = "niche_cc", "niche_cc_id"
CT_KEY = "cell_type"
K_RANGE = (int(os.environ.get("CC_KMIN", 5)), int(os.environ.get("CC_KMAX", 15)))
MAX_RUNS = int(os.environ.get("CC_MAX_RUNS", 3))
N_AUTOK = int(os.environ.get("CC_N_AUTOK", 200_000))   # cells used for the stability search
N_FIT = int(os.environ.get("CC_N_FIT", 500_000))       # cells used to fit the final GMM
FORCE_K = os.environ.get("NICHE_K")
N_LAYERS = 3
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
TRAINER = dict(accelerator="cpu", enable_progress_bar=False, enable_model_summary=False, logger=False)

d = common.read_light(H5, keys=("obs",), obsm=("spatial", "X_pca"))
obs = common.apply_overrides(d["obs"], log=log)
adata = ad.AnnData(obs=obs, obsm={"spatial": d["obsm"]["spatial"], "X_pca": d["obsm"]["X_pca"].astype(np.float32)})
del d
samples = pd.read_csv(f"{ROOT}/scripts/samples.tsv", sep="\t")
samples = samples[samples.sample_id.isin(adata.obs.sample_id.unique())].reset_index(drop=True)
order = samples.sample_id.tolist()
adata.obs["sample_id"] = pd.Categorical(adata.obs["sample_id"].astype(str), categories=order)
log(f"loaded {adata.shape}")

# ---- spatial graph + neighbourhood aggregation (CellCharter defaults)
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, library_key="sample_id")
cc.gr.remove_long_links(adata)                       # drop Delaunay edges above the 99th percentile length
cc.gr.aggregate_neighbors(adata, n_layers=N_LAYERS, use_rep="X_pca", out_key="X_cellcharter", sample_key="sample_id")
log(f"aggregated neighbourhoods: {adata.obsm['X_cellcharter'].shape}")
del adata.obsp["spatial_connectivities"], adata.obsp["spatial_distances"]

rng = np.random.default_rng(0)
idx_autok = np.sort(rng.choice(adata.n_obs, size=min(N_AUTOK, adata.n_obs), replace=False))
idx_fit = np.sort(rng.choice(adata.n_obs, size=min(N_FIT, adata.n_obs), replace=False))

# ---- number of clusters by stability
if FORCE_K:
    k_best = int(FORCE_K); stab = None
else:
    autok = cc.tl.ClusterAutoK(n_clusters=K_RANGE, max_runs=MAX_RUNS,
                               model_params=dict(random_state=0, trainer_params=TRAINER))
    autok.fit(adata[idx_autok].copy(), use_rep="X_cellcharter")
    k_best = int(autok.best_k)
    stab = {int(k): [float(v) for v in vals] for k, vals in autok.stability.items()} if isinstance(autok.stability, dict) else None
    try:
        fig = cc.pl.autok_stability(autok, return_fig=True) if "return_fig" in cc.pl.autok_stability.__code__.co_varnames else None
        if fig is None:
            cc.pl.autok_stability(autok); fig = plt.gcf()
        fig.savefig(f"{OUT}/autok_stability.png"); plt.close("all")
    except Exception as e:
        log(f"stability plot failed: {e}")
    log(f"ClusterAutoK best_k = {k_best}")
    try:
        pd.DataFrame(autok.stability).to_csv(f"{OUT}/autok_stability.csv")
    except Exception:
        pass

# ---- final GMM
gmm = cc.tl.Cluster(n_clusters=k_best, random_state=0, trainer_params=TRAINER)
gmm.fit(adata[idx_fit].copy(), use_rep="X_cellcharter")
labels = np.asarray(gmm.predict(adata, use_rep="X_cellcharter")).astype(int)
log(f"GMM fitted on {len(idx_fit):,} cells, predicted {adata.n_obs:,} cells")
del adata.obsm["X_cellcharter"]

# ---- naming: dominant | most enriched cell type among assigned cells
adata.obs[NID] = pd.Categorical([f"C{l}" for l in labels], categories=[f"C{i}" for i in range(k_best)])
ct_by = pd.crosstab(labels, adata.obs[CT_KEY]).reindex(range(k_best)).fillna(0)
frac = ct_by.div(ct_by.sum(1), axis=0)
names = niche_downstream.name_niches(frac, adata.obs[CT_KEY].value_counts(normalize=True), prefix="C")
adata.obs[NK] = pd.Categorical([names[l] for l in labels], categories=[names[i] for i in range(k_best)])
log(f"niches (k={k_best}):\n" + adata.obs[NK].value_counts().to_string())
json.dump({"method": "CellCharter 0.3.5: Delaunay graph (long links removed), aggregate_neighbors n_layers=3 on 30 PCs, "
                     f"GaussianMixture; k selected by ClusterAutoK stability over {K_RANGE} with max_runs={MAX_RUNS} on "
                     f"{len(idx_autok):,} cells; final GMM fitted on {len(idx_fit):,} cells and predicted on all",
           "k": k_best, "forced": bool(FORCE_K), "stability": stab}, open(f"{OUT}/method.json", "w"), indent=1)

niche_downstream.downstream(adata, NK, OUT, samples, log=log, CT_KEY=CT_KEY, method="CellCharter GMM on aggregated PCA neighbourhoods")

meta_out = adata.obs[[NID, NK]].copy()
meta_out.to_csv(f"{OUT}/cellcharter_labels.csv.gz")
common.write_obs(H5, adata.obs)
log("obs written back to h5ad; done")
