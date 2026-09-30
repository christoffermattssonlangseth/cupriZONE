"""Shared helpers for the lean post-checkpoint steps (h5py-level reads; annotation overrides)."""
import os, json
import numpy as np, pandas as pd, h5py
from anndata.io import read_elem, write_elem, sparse_dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read_light(path, keys=("obs",), obsm=("spatial",)):
    """Read obs (+ selected obsm arrays, + var) from an h5ad without touching X/layers/obsp."""
    out = {}
    with h5py.File(path, "r") as f:
        for k in keys:
            out[k] = read_elem(f[k])
        out["obsm"] = {k: np.asarray(read_elem(f["obsm"][k])) for k in obsm if k in f["obsm"]}
    return out


def read_rows(path, idx):
    """Read a row subset of X (sparse) from an h5ad."""
    idx = np.sort(np.asarray(idx))
    with h5py.File(path, "r") as f:
        return sparse_dataset(f["X"])[idx], idx


def write_obs(path, obs):
    with h5py.File(path, "r+") as f:
        if "obs" in f:
            del f["obs"]
        write_elem(f, "obs", obs)


def apply_overrides(obs, key="cell_type", log=print):
    """Apply scripts/annotation_overrides.json (leiden cluster -> label). Returns the modified obs."""
    p = f"{ROOT}/scripts/annotation_overrides.json"
    if not os.path.exists(p):
        return obs
    ov = {k: v for k, v in json.load(open(p)).items() if not k.startswith("_")}
    ct = obs[key].astype(str)
    lei = obs["leiden"].astype(str)
    for cl, lab in ov.items():
        m = (lei == cl).values
        if m.any():
            log(f"override: leiden {cl} ({m.sum()} cells) {ct[m].iloc[0]} -> {lab}")
            ct[m] = lab
    obs[key] = pd.Categorical(ct)
    obs["cell_type_cluster"] = pd.Categorical(ct + " (c" + lei + ")")
    return obs
