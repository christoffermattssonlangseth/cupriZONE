"""Test: is Crh expressed in OPCs after cuprizone, in the caudate putamen (striatum)?
Per section x region x cell type: Crh detection; CupRap vs control; spill-over controls
(a) other glia in the same region, (b) distance of Crh+ OPCs to the nearest Crh+ neuron vs Crh- OPCs."""
import os, sys
import numpy as np, pandas as pd, h5py
from anndata.io import sparse_dataset
from scipy import stats
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common
ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/gene_queries"; os.makedirs(OUT, exist_ok=True)
H5 = f"{RES}/xenium_all.h5ad"
d = common.read_light(H5, keys=("obs", "var"), obsm=("spatial",))
obs, var, xy = d["obs"], d["var"], d["obsm"]["spatial"]
obs["sample_id"] = obs.sample_id.astype(str)
print("CRH-system genes on panel:", [g for g in var.index if g.lower().startswith(("crh", "ucn", "nr3c"))])
genes = ["Crh", "Pdgfra", "Cspg4", "Olig2", "Gpr17", "Enpp6"]
gi = [var.index.get_loc(g) for g in genes]
M = np.zeros((len(obs), len(genes)), np.float32)
with h5py.File(H5) as f:
    C = sparse_dataset(f["layers/counts"])
    for s in range(0, len(obs), 200_000):
        M[s:s+200_000] = C[s:s+200_000][:, gi].toarray()
cnt = pd.DataFrame(M, columns=genes, index=obs.index)
obs["crh"] = cnt.Crh.values; obs["crh_pos"] = cnt.Crh.values >= 1; obs["crh_pos2"] = cnt.Crh.values >= 2

# regions: striatum (caudate putamen) = >=25 % MSN within 75 µm; white matter = CellCharter C7 / k-means N5; cortex = C3/C4/C5
msn_frac = np.zeros(len(obs))
for s in obs.sample_id.unique():
    m = np.where((obs.sample_id == s).values)[0]
    is_msn = (obs.cell_type.values[m] == "Striatal MSN").astype(float)
    nb = cKDTree(xy[m]).query_ball_point(xy[m], r=75.0)
    msn_frac[m] = [is_msn[n].mean() if len(n) else 0 for n in nb]
cc = obs.niche_cc.astype(str).str[:2]
region = np.where(msn_frac >= 0.25, "striatum (CPu)",
         np.where(cc.isin(["C7"]).values | obs.niche.astype(str).str.startswith("N5").values, "white matter / lesion",
         np.where(cc.isin(["C3", "C4", "C5"]).values, "cortex", "other")))
obs["region"] = region

CT = ["OPC", "Lesion glia (Gfap+ Olig2+)", "Oligodendrocyte", "Astrocyte", "Microglia", "Striatal MSN", "Inhibitory neuron"]
sub = obs[obs.cell_type.isin(CT)].copy()
sub["cell_type"] = sub.cell_type.astype(str)
g = sub.groupby(["experiment", "condition", "region", "cell_type"], observed=True).agg(
    n=("crh_pos", "size"), pct_crh_pos=("crh_pos", lambda x: 100 * x.mean()), pct_crh_ge2=("crh_pos2", lambda x: 100 * x.mean()),
    mean_crh=("crh", "mean"))
g.to_csv(f"{OUT}/Crh_by_region_celltype_condition.csv")
show = g.reset_index(); show = show[show.cell_type.isin(["OPC", "Lesion glia (Gfap+ Olig2+)", "Oligodendrocyte", "Astrocyte"]) & show.region.isin(["striatum (CPu)", "white matter / lesion", "cortex"])]
print("\nCrh detection by experiment / condition / region / glial type:\n" +
      show.pivot_table(index=["experiment", "region", "cell_type"], columns="condition", values=["n", "pct_crh_pos"]).round(2).to_string())

# per-section OPC Crh rate in striatum: CupRap vs control (within experiment)
ps = sub[sub.cell_type == "OPC"].groupby(["experiment", "condition", "sample_id", "region"], observed=True).agg(
    n_opc=("crh_pos", "size"), n_crh_pos=("crh_pos", "sum"), pct=("crh_pos", lambda x: 100 * x.mean()))
ps.to_csv(f"{OUT}/Crh_in_OPC_per_section_region.csv")
print("\nOPCs per section (striatum):\n" + ps.reset_index().query("region == 'striatum (CPu)'").to_string(index=False))
rows = []
for expt, g1, g2 in [("no_recovery", "CupRap", "Control"), ("recovery", "CupRap", "Control"), ("infusion", "OSM_infused", "BSA_infused")]:
    for reg in ["striatum (CPu)", "white matter / lesion", "cortex"]:
        q = ps.reset_index().query("experiment == @expt and region == @reg")
        a, b = q[q.condition == g1].pct.values, q[q.condition == g2].pct.values
        if len(a) > 1 and len(b) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append({"experiment": expt, "region": reg, f"mean_pct_{g1}": a.mean(), f"mean_pct_{g2}": b.mean(), "p_welch": p})
print("\nOPC Crh+ rate, CupRap vs control (per-section Welch):\n" + pd.DataFrame(rows).round(3).to_string(index=False))
pd.DataFrame(rows).to_csv(f"{OUT}/Crh_in_OPC_tests.csv", index=False)

# spill-over control: distance from each OPC to the nearest Crh+ neuron (>=2 transcripts), Crh+ vs Crh- OPCs
neuron = obs.cell_type.isin(["Inhibitory neuron", "Excitatory neuron", "Neuron (other)", "Striatal MSN"]).values & obs.crh_pos2.values
res = []
for s in obs.sample_id.unique():
    m = (obs.sample_id == s).values
    opc = m & (obs.cell_type.values == "OPC"); nn = m & neuron
    if nn.sum() < 10: continue
    tree = cKDTree(xy[nn]); dist, _ = tree.query(xy[opc])
    cp = obs.crh_pos.values[opc]
    res.append({"sample_id": s, "condition": obs.condition[m].iloc[0], "experiment": obs.experiment[m].iloc[0],
                "n_opc": int(opc.sum()), "n_crh_pos_opc": int(cp.sum()),
                "median_dist_crhpos_opc_to_crh_neuron": float(np.median(dist[cp])) if cp.sum() else np.nan,
                "median_dist_crhneg_opc_to_crh_neuron": float(np.median(dist[~cp])),
                "pct_crhpos_opc_within_15um_of_crh_neuron": 100 * float((dist[cp] < 15).mean()) if cp.sum() else np.nan,
                "pct_crhneg_opc_within_15um_of_crh_neuron": 100 * float((dist[~cp] < 15).mean())})
res = pd.DataFrame(res).set_index("sample_id"); res.to_csv(f"{OUT}/Crh_OPC_spillover_check.csv")
print("\nspill-over check (distance to nearest Crh+ neuron):\n" + res.round(1).to_string())

# Crh+ OPCs: are they a distinct OPC state? mean counts of OPC/COP genes in Crh+ vs Crh- OPCs, by condition (cuprizone sections)
opc = obs.cell_type.values == "OPC"
cup = obs.experiment.isin(["no_recovery", "recovery"]).values
tab = cnt[opc & cup].assign(crh_pos=obs.crh_pos[opc & cup].values, condition=obs.condition[opc & cup].values).groupby(["condition", "crh_pos"]).mean()
print("\nmean raw counts in OPCs (cuprizone experiments), Crh- vs Crh+:\n" + tab.round(2).to_string())

# figure: striatal OPCs, Crh+ highlighted, one acute CupRap and one control section
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, s in zip(axes, ["NoRecov_Cntl1", "NoRecov_CupRap3"]):
    m = (obs.sample_id == s).values
    ax.scatter(xy[m, 0], -xy[m, 1], s=0.05, c="#e1e0d9", lw=0, rasterized=True)
    o = m & opc
    ax.scatter(xy[o, 0], -xy[o, 1], s=1, c="#86b6ef", lw=0, rasterized=True, label="OPC")
    p = o & obs.crh_pos.values
    ax.scatter(xy[p, 0], -xy[p, 1], s=6, c="#e34948", lw=0, label="Crh+ OPC")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{s}: {int(p.sum())} Crh+ of {int(o.sum())} OPCs")
    for sp_ in ax.spines.values(): sp_.set_visible(False)
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=3)
fig.savefig(f"{OUT}/Crh_in_OPC_spatial.png"); plt.close(fig)
