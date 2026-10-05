"""Store the spatial graphs in results/xenium_all.h5ad obsp: the 15-NN graph per section (used for the composition niches
and neighbourhood enrichment) and the Delaunay graph with long links removed (used by CellCharter)."""
import os, sys
import numpy as np, pandas as pd, anndata as ad, squidpy as sq, cellcharter as cc, h5py
from anndata.io import write_elem
sys.path.insert(0, os.path.dirname(__file__))
import common
RES = f"{common.ROOT}/results"; H5 = f"{RES}/xenium_all.h5ad"
d = common.read_light(H5, keys=("obs",), obsm=("spatial",))
a = ad.AnnData(obs=d["obs"][["sample_id"]].copy(), obsm={"spatial": d["obsm"]["spatial"]})
a.obs["sample_id"] = a.obs.sample_id.astype(str).astype("category")
sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=15, library_key="sample_id", delaunay=False)
knn_c, knn_d = a.obsp["spatial_connectivities"].tocsr(), a.obsp["spatial_distances"].tocsr()
sq.gr.spatial_neighbors(a, coord_type="generic", delaunay=True, library_key="sample_id"); cc.gr.remove_long_links(a)
del_c, del_d = a.obsp["spatial_connectivities"].tocsr(), a.obsp["spatial_distances"].tocsr()
print(f"knn15: {knn_c.nnz:,} edges; delaunay (long links removed): {del_c.nnz:,} edges; median edge length knn {np.median(knn_d.data):.1f} µm, delaunay {np.median(del_d.data):.1f} µm")
with h5py.File(H5, "r+") as f:
    for k, M in [("spatial_knn15_connectivities", knn_c), ("spatial_knn15_distances", knn_d), ("spatial_delaunay_connectivities", del_c), ("spatial_delaunay_distances", del_d)]:
        if k in f["obsp"]: del f["obsp"][k]
        write_elem(f["obsp"], k, M.astype(np.float32))
    note = dict(f["uns"]["obs_columns"].attrs) if False else {}
    if "graphs" in f["uns"]: del f["uns"]["graphs"]
    write_elem(f["uns"], "graphs", {"connectivities": "scanpy expression kNN (15 neighbours, 30 PCs) used for Leiden",
                                    "spatial_knn15_*": "squidpy 15 nearest neighbours in physical space, within section (used for k-means niches and nhood enrichment)",
                                    "spatial_delaunay_*": "squidpy Delaunay within section, edges above the 99th percentile length removed (CellCharter input); distances in µm"})
print("written")
