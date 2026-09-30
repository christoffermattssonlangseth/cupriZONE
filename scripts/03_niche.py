"""Niche analysis on the preprocessed Xenium object (results/xenium_all.h5ad).
1. Spatial kNN graph per section (squidpy), 2. neighbourhood cell-type composition per cell,
3. k-means niches on compositions (k chosen by silhouette), 4. niche x cell-type enrichment,
5. niche abundance per section + condition contrasts within each experiment,
6. cell-type neighbourhood enrichment (squidpy) per section, averaged per condition,
7. spatial niche maps. Writes results/niches/*.csv|png and results/xenium_all_niches.h5ad."""
import os, sys, time, warnings, itertools, json
import numpy as np, pandas as pd, scanpy as sc, squidpy as sq
import scipy.sparse as sp
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common, niche_downstream

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.environ.get("RES_DIR", f"{ROOT}/results"); TAG = os.environ.get("NICHE_TAG", ""); OUT = f"{RES}/niches{TAG}"
NK, NID = f"niche{TAG}", f"niche_id{TAG}"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
K_NEIGH = int(os.environ.get("K_NEIGH", "15"))       # spatial neighbours per cell
K_RANGE = range(5, 15)                                 # candidate niche numbers
FORCE_K = os.environ.get("NICHE_K")                    # override k
CT_KEY = "cell_type"

H5 = f"{RES}/xenium_all.h5ad"
d = common.read_light(H5, keys=("obs",), obsm=("spatial",))
obs = common.apply_overrides(d["obs"], log=log)
adata = sc.AnnData(obs=obs, obsm={"spatial": d["obsm"]["spatial"]})   # metadata + coordinates only (memory)
del d
samples = pd.read_csv(f"{ROOT}/scripts/samples.tsv", sep="\t")
samples = samples[samples.sample_id.isin(adata.obs.sample_id.unique())].reset_index(drop=True)
order = samples.sample_id.tolist()
adata.obs["sample_id"] = pd.Categorical(adata.obs["sample_id"].astype(str), categories=order)
cts = list(adata.obs[CT_KEY].cat.categories)
log(f"loaded {adata.shape}; {len(cts)} cell types")

# ------------------------------------------------------------------ 1-2 spatial graph + composition
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=K_NEIGH, library_key="sample_id", delaunay=False)
A = adata.obsp["spatial_connectivities"].tocsr()
A.data[:] = 1.0
onehot = sp.csr_matrix(pd.get_dummies(adata.obs[CT_KEY]).values.astype(np.float32))
comp = A @ onehot                                  # neighbour counts per type (self excluded)
deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
comp = np.asarray(comp.todense()) / deg[:, None]
comp_df = pd.DataFrame(comp, index=adata.obs_names, columns=list(pd.get_dummies(adata.obs[CT_KEY]).columns))
adata.obsm["neighbourhood_composition"] = comp_df.values
log("neighbourhood composition computed")

# ------------------------------------------------------------------ 3 k-means niches
rng = np.random.default_rng(0)
sub_idx = rng.choice(adata.n_obs, size=min(200_000, adata.n_obs), replace=False)
sil_idx = rng.choice(sub_idx, size=min(30_000, len(sub_idx)), replace=False)
sil = {}
for k in K_RANGE:
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=8192, n_init=5).fit(comp[sub_idx])
    sil[k] = silhouette_score(comp[sil_idx], km.predict(comp[sil_idx]))
    log(f"k={k}: silhouette={sil[k]:.3f}")
pd.Series(sil, name="silhouette").rename_axis("k").to_csv(f"{OUT}/kmeans_silhouette.csv")
k_best = int(FORCE_K) if FORCE_K else max(sil, key=sil.get)
km = MiniBatchKMeans(n_clusters=k_best, random_state=0, batch_size=8192, n_init=10).fit(comp[sub_idx])
labels = km.predict(comp)

# order niches by dominant cell type, name them by the two most enriched types
overall = comp_df.mean(0)
centers = pd.DataFrame(km.cluster_centers_, columns=comp_df.columns)
enr = np.log2((centers + 1e-3) / (overall.values + 1e-3))
names = {}
for i in range(k_best):
    dom = centers.iloc[i].idxmax()
    top_enr = [t for t in enr.iloc[i].sort_values(ascending=False).index if t != dom][0]
    names[i] = f"N{i}: {dom} | {top_enr}"
adata.obs[NID] = pd.Categorical([f"N{l}" for l in labels], categories=[f"N{i}" for i in range(k_best)])
adata.obs[NK] = pd.Categorical([names[l] for l in labels], categories=[names[i] for i in range(k_best)])
log(f"niches (k={k_best}):\n" + adata.obs[NK].value_counts().to_string())

meta = samples.set_index("sample_id")
contrasts = niche_downstream.CONTRASTS
niche_downstream.downstream(adata, NK, OUT, samples, log=log, CT_KEY=CT_KEY)

# ------------------------------------------------------------------ 6 cell-type neighbourhood enrichment per section
zs = {}
for s in ([] if TAG else order):
    a_s = adata[adata.obs.sample_id == s].copy()
    a_s.obs[CT_KEY] = a_s.obs[CT_KEY].cat.remove_unused_categories()
    sq.gr.spatial_neighbors(a_s, coord_type="generic", n_neighs=K_NEIGH, delaunay=False)
    sq.gr.nhood_enrichment(a_s, cluster_key=CT_KEY, n_perms=200, seed=0, show_progress_bar=False)
    z = pd.DataFrame(a_s.uns[f"{CT_KEY}_nhood_enrichment"]["zscore"], index=a_s.obs[CT_KEY].cat.categories,
                     columns=a_s.obs[CT_KEY].cat.categories).reindex(index=cts, columns=cts)
    zs[s] = z
    z.to_csv(f"{OUT}/nhood_enrichment_z_{s}.csv")
    log(f"nhood enrichment {s}")
def mean_z(sel):
    return pd.concat([zs[s] for s in sel]).groupby(level=0).mean().reindex(index=cts, columns=cts)
valid = [(e, g1, g2) for (e, g1, g2) in contrasts
         if zs and ((meta.experiment == e) & (meta.condition == g1)).any() and ((meta.experiment == e) & (meta.condition == g2)).any()]
fig, axes = plt.subplots(max(len(valid), 1), 3, figsize=(21, 6.5 * max(len(valid), 1)), squeeze=False, constrained_layout=True)
if not valid: plt.close(fig)
for row, (expt, g1, g2) in zip(axes, valid):
    m1 = mean_z(meta.index[(meta.experiment == expt) & (meta.condition == g1)])
    m2 = mean_z(meta.index[(meta.experiment == expt) & (meta.condition == g2)])
    m1.to_csv(f"{OUT}/nhood_enrichment_meanz_{expt}_{g1}.csv"); m2.to_csv(f"{OUT}/nhood_enrichment_meanz_{expt}_{g2}.csv")
    for ax, (m, t) in zip(row, [(m2, f"{expt}: {g2}"), (m1, f"{expt}: {g1}"), (m1 - m2, f"{expt}: {g1} minus {g2}")]):
        vmax = np.nanpercentile(np.abs(m.values), 98)
        im = ax.imshow(m.values, cmap=viz.div_cmap(), vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=90, fontsize=7)
        ax.set_yticks(range(len(cts))); ax.set_yticklabels(cts, fontsize=7)
        ax.set_title(t, fontsize=9); plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
if valid:
    fig.suptitle("Cell-type neighbourhood enrichment (squidpy z-score, mean over sections per condition)", x=0.02, ha="left")
    fig.savefig(f"{OUT}/nhood_enrichment_by_condition.png"); plt.close(fig)

comp_df.astype(np.float32).to_parquet(f"{OUT}/neighbourhood_composition.parquet")
meta_out = adata.obs.copy()
meta_out["x_centroid"], meta_out["y_centroid"] = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
meta_out.to_csv(f"{RES}/cell_metadata_with_niches.csv.gz")
common.write_obs(H5, adata.obs)          # niche labels + overrides written back into xenium_all.h5ad
log("obs written back to h5ad; done")
