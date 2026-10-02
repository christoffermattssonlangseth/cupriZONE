"""Fold the region-focused and subclustering labels into results/xenium_all.h5ad obs, and export the full obs table."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common
ROOT = common.ROOT; RES = f"{ROOT}/results"; H5 = f"{RES}/xenium_all.h5ad"
obs = common.read_light(H5, keys=("obs",), obsm=())["obs"]
def add(df, cols, fill):
    for c, new in cols.items():
        obs[new] = df[c].reindex(obs.index)
        if fill is not None: obs[new] = obs[new].fillna(fill)
v = pd.read_csv(f"{RES}/vsvz_focus/vsvz_cells.csv.gz", index_col=0)
add(v, {"vsvz_wall": "vsvz_wall", "svz_niche": "vsvz_niche", "dist_to_lining": "vsvz_dist_to_lining_um"}, None)
obs["vsvz_wall"] = obs.vsvz_wall.fillna("not V-SVZ"); obs["vsvz_niche"] = obs.vsvz_niche.fillna("not V-SVZ")
s = pd.read_csv(f"{RES}/striatum_focus/striatum_cells.csv.gz", index_col=0)
add(s, {"striatum_part": "striatum_part", "striatum_niche": "striatum_niche"}, "not striatum")
parts = []
for b in ["recovery_2023", "acute_infusion_2024"]:
    d = pd.read_csv(f"{RES}/oligo_subcluster/{b}_cells.csv.gz", index_col=0); d["oligo_subcluster"] = b.split("_")[0] + "_" + d["sub"].astype(str); parts.append(d)
o = pd.concat(parts)
add(o, {"oligo_subcluster": "oligo_subcluster", "immune_score": "oligo_immune_score"}, None)
obs["oligo_subcluster"] = obs.oligo_subcluster.fillna("not oligodendroglia")
lesion = obs.niche_cc.astype(str).str.startswith("C7") | obs.niche.astype(str).str.startswith("N5")
obs["lesion_wm_niche"] = lesion.values
for c in ["vsvz_wall", "vsvz_niche", "striatum_part", "striatum_niche", "oligo_subcluster"]: obs[c] = obs[c].astype("category")
common.write_obs(H5, obs)
import h5py
from anndata.io import write_elem
with h5py.File(H5, "r+") as f:
    if "obs_columns" in f["uns"]: del f["uns"]["obs_columns"]
    write_elem(f["uns"], "obs_columns", {
        "leiden, cell_type": "02_preprocess.py (+ overrides in annotation_overrides.json)",
        "niche, niche_id": "03_niche.py k-means on 15-NN cell-type composition, k=6 (silhouette)",
        "niche_k12, niche_id_k12": "03_niche.py, k=12 forced",
        "niche_cc, niche_cc_id": "05_cellcharter.py, CellCharter GMM k=8 (stability)",
        "lesion_wm_niche": "True if niche_cc C7 or niche N5 (white matter / lesion compartment)",
        "vsvz_wall, vsvz_niche, vsvz_dist_to_lining_um": "14_vsvz_focus.py: <=75 um from lateral-ventricle ependymal lining; dorsal = white-matter context, lateral = striatal context",
        "striatum_part, striatum_niche": "15_striatum_focus.py: caudate putamen (>=25% MSN within 75 um, outside V-SVZ); fibre bundles vs matrix",
        "oligo_subcluster, oligo_immune_score": "17_oligo_subcluster.py: per-batch Leiden of Oligodendrocyte+OPC+lesion glia with the batch's full panel; immune score = n detected immune genes (recovery: C4b,H2-Eb1,Lgals3,Il6; acute: H2-Eb1,Lgals3,Socs3,Osmr,Tnf,Cd52)"})
obs.to_csv(f"{RES}/cell_metadata_full.csv.gz")
print(obs.dtypes.to_string()); print("\n", obs[["vsvz_wall", "striatum_part", "oligo_subcluster"]].describe().to_string())
