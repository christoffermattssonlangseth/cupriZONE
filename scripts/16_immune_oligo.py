"""Immune-like oligodendroglia? Score MHC-II (H2-Eb1, all sections), C4b (2023 recovery batch only) and Socs3/Osmr/Tnfrsf1a
(2024 acute+infusion batch only) in oligodendrocyte-lineage cells by condition and compartment, with a spill-over control
(distance to the nearest H2-Eb1+ microglia)."""
import os, sys
import numpy as np, pandas as pd, scanpy as sc
from scipy.spatial import cKDTree
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
import common, focus_utils as fu
ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/gene_queries"; H5 = f"{RES}/xenium_all.h5ad"
d = common.read_light(H5, keys=("obs", "var"), obsm=("spatial",)); obs, var, xy = d["obs"], d["var"], d["obsm"]["spatial"]
obs["sample_id"] = obs.sample_id.astype(str); ct = obs.cell_type.astype(str).values
cnt = fu.read_genes(H5, var, ["H2-Eb1", "Lgals3", "Il33", "Mbp", "Mog", "Pdgfra", "Trem2"], len(obs)); cnt.index = obs.index
# batch-specific genes straight from the per-sample 10x matrices
extra = pd.DataFrame(index=obs.index, columns=["C4b", "Socs3", "Osmr", "Tnfrsf1a", "Cd52", "Il6"], dtype=np.float32)
samples = pd.read_csv(f"{ROOT}/scripts/samples.tsv", sep="\t")
for _, r in samples.iterrows():
    a = sc.read_10x_mtx(f"{ROOT}/data/geo/{r.sample_id}", var_names="gene_symbols", make_unique=True)
    a.obs_names = [f"{r.sample_id}:{c}" for c in a.obs_names]; ids = obs.index[obs.sample_id == r.sample_id]
    a = a[a.obs_names.isin(ids)]
    for g in extra.columns:
        if g in a.var_names: extra.loc[a.obs_names, g] = a[:, g].X.toarray().ravel()
cnt = cnt.join(extra)
lineage = np.isin(ct, ["Oligodendrocyte", "OPC"])
lesion = obs.niche_cc.astype(str).str.startswith("C7").values | obs.niche.astype(str).str.startswith("N5").values
obs["compartment"] = np.where(lesion, "white matter / lesion", "grey matter")
grp = obs.experiment.astype(str) + ":" + obs.condition.astype(str)

def rate(gene, mask, by):
    return (cnt[gene][mask] >= 1).groupby(by[mask]).mean() * 100

print("% of cells with >=1 transcript, by group | compartment (rows) and cell type (cols)")
for gene in ["H2-Eb1", "C4b", "Socs3", "Osmr", "Tnfrsf1a", "Il33", "Lgals3"]:
    ok = cnt[gene].notna().values
    tab = pd.DataFrame({t: rate(gene, ok & (ct == t), grp + " | " + obs.compartment) for t in ["Oligodendrocyte", "OPC", "Microglia", "Astrocyte", "Striatal MSN"]})
    print(f"\n== {gene}\n" + tab.round(2).to_string())
    tab.round(3).to_csv(f"{OUT}/immuneOL_{gene.replace('-', '')}_rates.csv")

# per-section test: % H2-Eb1+ and C4b+ oligodendrocytes in the lesion compartment, CupRap vs control
rows = []
for gene in ["H2-Eb1", "C4b", "Socs3", "Il33"]:
    ok = cnt[gene].notna().values
    for comp in ["white matter / lesion", "grey matter"]:
        m = ok & (ct == "Oligodendrocyte") & (obs.compartment.values == comp)
        ps = (cnt[gene][m] >= 1).groupby(obs.sample_id[m]).mean() * 100
        for e, g1, g2 in fu.CONTRASTS:
            s1 = samples[(samples.experiment == e) & (samples.condition == g1)].sample_id; s2 = samples[(samples.experiment == e) & (samples.condition == g2)].sample_id
            a, b = ps.reindex(s1).dropna().values, ps.reindex(s2).dropna().values
            if len(a) > 1 and len(b) > 1:
                rows.append({"gene": gene, "compartment": comp, "experiment": e, f"mean_pct_{g1[:6]}": a.mean(), f"mean_pct_{g2[:6]}": b.mean(), "p_welch": stats.ttest_ind(a, b, equal_var=False)[1]})
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/immuneOL_oligodendrocyte_tests.csv", index=False)
print("\nOligodendrocytes: % positive per section, CupRap/OSM vs control/BSA:\n" + res.round(3).to_string(index=False))

# spill-over: H2-Eb1+ oligodendrocytes vs H2-Eb1- oligodendrocytes, distance to nearest H2-Eb1+ microglia (cuprizone sections)
rows = []
for s in obs.sample_id.unique():
    m = (obs.sample_id == s).values; mg = m & (ct == "Microglia") & (cnt["H2-Eb1"].values >= 1); ol = m & (ct == "Oligodendrocyte")
    if mg.sum() < 20 or ol.sum() < 100: continue
    dist, _ = cKDTree(xy[mg]).query(xy[ol]); pos = cnt["H2-Eb1"].values[ol] >= 1
    rows.append({"sample_id": s, "condition": obs.condition[m].iloc[0], "n_H2Eb1pos_oligo": int(pos.sum()), "pct_H2Eb1pos_oligo": 100 * pos.mean(),
                 "median_dist_pos": float(np.median(dist[pos])) if pos.sum() else np.nan, "median_dist_neg": float(np.median(dist[~pos])),
                 "pct_pos_within_15um": 100 * float((dist[pos] < 15).mean()) if pos.sum() else np.nan, "pct_neg_within_15um": 100 * float((dist[~pos] < 15).mean())})
sp_ = pd.DataFrame(rows).set_index("sample_id"); sp_.to_csv(f"{OUT}/immuneOL_H2Eb1_spillover.csv")
print("\nspill-over check, H2-Eb1 in oligodendrocytes (distance to nearest H2-Eb1+ microglia):\n" + sp_.round(1).to_string())

# do H2-Eb1+ oligodendrocytes look like oligodendrocytes? mean counts of core genes, lesion compartment, CupRap sections
m = (ct == "Oligodendrocyte") & lesion & obs.condition.isin(["CupRap"]).values
pos = cnt["H2-Eb1"].values >= 1
print("\nlesion oligodendrocytes in CupRap sections, mean raw counts (H2-Eb1- vs H2-Eb1+):\n" +
      cnt.loc[m, ["Mbp", "Mog", "Pdgfra", "Trem2", "Lgals3", "Il33"]].groupby(pos[m]).mean().round(2).to_string())
