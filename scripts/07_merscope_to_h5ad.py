"""Assemble the 4 MERSCOPE sections into one raw-count AnnData (results/merscope/merscope_recovery_raw.h5ad).
X = gene counts (sparse, blank barcodes removed); obs = Vizgen cell metadata + sample annotations + blank counts;
obsm['spatial'] = (center_x, center_y) in µm. No QC/normalisation here."""
import os, sys, time
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = f"{ROOT}/data/geo_merscope"; OUT = f"{ROOT}/results/merscope"; os.makedirs(OUT, exist_ok=True)
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
samples = pd.read_csv(f"{ROOT}/scripts/samples_merscope.tsv", sep="\t")

adatas, rows = [], []
for _, r in samples.iterrows():
    d = f"{DATA}/{r.sample_id}"
    cbg = pd.read_csv(f"{d}/cell_by_gene.csv.gz", index_col=0)
    meta = pd.read_csv(f"{d}/cell_metadata.csv.gz", index_col=0)
    cbg.index = cbg.index.astype(str); meta.index = meta.index.astype(str)
    common = cbg.index.intersection(meta.index)
    cbg, meta = cbg.loc[common], meta.loc[common]
    blank_cols = [c for c in cbg.columns if c.lower().startswith("blank")]
    genes = [c for c in cbg.columns if c not in blank_cols]
    a = ad.AnnData(X=sp.csr_matrix(cbg[genes].values.astype(np.float32)), obs=meta.copy(),
                   var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    a.obs["blank_counts"] = cbg[blank_cols].sum(1).values
    a.obs["n_transcripts"] = cbg[genes].sum(1).values
    a.obs["n_genes"] = (cbg[genes] > 0).sum(1).values
    for k in ["gsm", "sample_id", "condition", "experiment", "replicate"]:
        a.obs[k] = str(r[k])
    a.obs["platform"] = "MERSCOPE"
    a.obsm["spatial"] = meta[["center_x", "center_y"]].to_numpy()
    a.obs_names = [f"{r.sample_id}:{c}" for c in a.obs_names]
    a.uns["blank_barcodes"] = blank_cols
    adatas.append(a)
    rows.append({"sample_id": r.sample_id, "n_cells": a.n_obs, "n_genes": a.n_vars, "n_blanks": len(blank_cols),
                 "median_transcripts": float(np.median(a.obs.n_transcripts)), "median_genes": float(np.median(a.obs.n_genes)),
                 "median_volume": float(np.median(meta["volume"])) if "volume" in meta else np.nan,
                 "blank_rate": float(a.obs.blank_counts.sum() / (a.obs.blank_counts.sum() + a.obs.n_transcripts.sum())),
                 "metadata_columns": ";".join(meta.columns)})
    log(f"{r.sample_id}: {a.n_obs} cells x {a.n_vars} genes ({len(blank_cols)} blanks); meta cols: {list(meta.columns)}")

gene_sets = [set(a.var_names) for a in adatas]
common = sorted(set.intersection(*gene_sets))
log(f"genes common to all sections: {len(common)} / union {len(set.union(*gene_sets))}")
adata = ad.concat([a[:, common] for a in adatas], join="inner", merge="first", uns_merge="first", index_unique=None)
adata.obs["sample_id"] = pd.Categorical(adata.obs["sample_id"], categories=samples.sample_id.tolist())
adata.layers["counts"] = adata.X.copy()
pd.DataFrame(rows).set_index("sample_id").to_csv(f"{OUT}/merscope_summary_per_sample.csv")
pd.Series(common, name="gene").to_csv(f"{OUT}/merscope_genes.csv", index=False)
adata.write_h5ad(f"{OUT}/merscope_recovery_raw.h5ad", compression="lzf")
log(f"wrote {OUT}/merscope_recovery_raw.h5ad {adata.shape}")
