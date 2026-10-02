"""Focused striatum (caudate putamen) analysis (Xenium), kept separate from the V-SVZ analysis.
Striatum = cells with >=25 % striatal MSN within 75 µm, excluding the V-SVZ band (<=75 µm from the ventricle lining).
Sub-compartments from local context: fibre bundles (white-matter glia-rich patches) vs matrix.
Outputs: composition/marker metrics per section, CupRap vs control, focused niches, zoom maps."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common, focus_utils as fu
ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/striatum_focus"; os.makedirs(OUT, exist_ok=True)
H5 = f"{RES}/xenium_all.h5ad"
d = common.read_light(H5, keys=("obs", "var"), obsm=("spatial",)); obs, var, xy = d["obs"], d["var"], d["obsm"]["spatial"]
obs["sample_id"] = obs.sample_id.astype(str); ct = obs.cell_type.astype(str).values
cnt = fu.read_genes(H5, var, fu.STATE_GENES, len(obs)); cnt.index = obs.index

ctx75 = fu.context_fractions(obs, xy, {"msn": ["Striatal MSN"]}, r=75.0)
lining = fu.ependymal_lining(obs, xy); dist = np.full(len(obs), np.inf)
for s in obs.sample_id.unique():
    m = np.where((obs.sample_id == s).values)[0]; ep = m[lining[m]]
    if len(ep): dist[m], _ = cKDTree(xy[ep]).query(xy[m])
stri = (ctx75.msn.values >= 0.25) & (dist > 75.0)
# fibre bundles = patches where white-matter glia dominate locally (within 40 µm); matrix = the rest
ctx40 = fu.context_fractions(obs, xy, {"wm": ["Oligodendrocyte", "OPC", "Microglia", "Lesion glia (Gfap+ Olig2+)"], "msn": ["Striatal MSN"]}, r=40.0)
sub = np.where(~stri, "not striatum", np.where((ctx40.wm.values >= 0.5) & (ctx40.msn.values < 0.3), "fibre bundles", "matrix"))
obs["striatum_part"] = sub
print(pd.crosstab(obs.sample_id, obs.striatum_part).to_string())

TYPES = ["Striatal MSN", "Inhibitory neuron", "Astrocyte", "Astrocyte Gfap-high / qNSC", "OPC", "Oligodendrocyte", "Microglia", "Lesion glia (Gfap+ Olig2+)", "Endothelial", "Pericyte-VSMC", "Neuroblast"]
REG = ["matrix", "fibre bundles"]
tab = fu.metrics_table(obs, cnt, "striatum_part", REG, TYPES); tab.to_csv(f"{OUT}/striatum_composition_per_section.csv", index=False)
# whole-striatum table too
obs["striatum_all"] = np.where(stri, "striatum", "not striatum")
tab_all = fu.metrics_table(obs, cnt, "striatum_all", ["striatum"], TYPES); tab_all.to_csv(f"{OUT}/striatum_whole_composition_per_section.csv", index=False)
res = fu.contrasts_table(pd.concat([tab, tab_all]), REG + ["striatum"]); res.to_csv(f"{OUT}/striatum_contrasts.csv", index=False)
pd.set_option("display.width", 250)
for e, g1, g2 in fu.CONTRASTS:
    q = res[res.experiment == e]
    piv = q.pivot(index="metric", columns="region", values=["mean1", "mean2", "p_welch"]).round(2)
    piv.columns = [f"{a.replace('mean1', g1).replace('mean2', g2)} | {b}" for a, b in piv.columns]
    print(f"\n=== {e}: {g1} vs {g2} (per-section means)\n" + piv.to_string())
# bundle share of striatum per section
bs = obs[stri].groupby(["experiment", "condition", "sample_id"], observed=True).striatum_part.apply(lambda x: 100 * (x == "fibre bundles").mean()).rename("pct_striatum_in_bundles")
print("\n% of striatal cells in fibre-bundle patches:\n" + bs.round(2).to_string()); bs.to_csv(f"{OUT}/striatum_bundle_share_per_section.csv")

lab, frac, enr, sil = fu.focused_niches(obs, xy, stri, TYPES, r=40.0, prefix="T")
obs["striatum_niche"] = "not striatum"; obs.loc[obs.index[stri], "striatum_niche"] = lab
grp = obs.experiment.astype(str) + ":" + obs.condition.astype(str)
g = pd.crosstab(grp[stri], obs.striatum_niche[stri], normalize="index") * 100
print(f"\nfocused striatal niches k={len(frac)} (best silhouette {max(sil.values()):.2f}); % of striatal cells per group:\n" + g.round(1).T.to_string())
g.round(2).to_csv(f"{OUT}/striatum_focused_niche_fraction_by_group.csv"); frac.round(4).to_csv(f"{OUT}/striatum_focused_niche_composition.csv")
pd.crosstab([obs.sample_id[stri], obs.condition[stri]], obs.striatum_niche[stri], normalize="index").round(4).to_csv(f"{OUT}/striatum_focused_niche_fraction_per_section.csv")
obs[stri][["sample_id", "experiment", "condition", "cell_type", "striatum_part", "striatum_niche"]].assign(x=xy[stri, 0], y=xy[stri, 1]).to_csv(f"{OUT}/striatum_cells.csv.gz")

cmap = json.load(open(f"{RES}/celltype_colors.json")); pcol = {"matrix": viz.CAT8[0], "fibre bundles": viz.CAT8[1]}; ncol_ = viz.cmap_for(sorted(set(lab)))
reps = ["NoRecov_Cntl1", "NoRecov_CupRap1", "Recov_Cntl1", "Recov_CupRap1", "Inf_BSA1", "Inf_OSM1"]
fig, axes = plt.subplots(3, len(reps), figsize=(3.6 * len(reps), 11))
for j, s in enumerate(reps):
    m = (obs.sample_id == s).values; pts = xy[m & stri]
    if not len(pts): continue
    cx = np.median(xy[m, 0]); left = pts[pts[:, 0] < cx]
    x0, x1 = np.percentile(left[:, 0], [1, 99]); y0, y1 = np.percentile(left[:, 1], [1, 99])
    box = m & (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    axes[0, j].scatter(xy[box, 0], -xy[box, 1], s=1.2, c=[cmap[c] for c in ct[box]], lw=0, rasterized=True)
    for row, col, key in [(1, pcol, "striatum_part"), (2, ncol_, "striatum_niche")]:
        axes[row, j].scatter(xy[box, 0], -xy[box, 1], s=0.6, c="#e1e0d9", lw=0, rasterized=True); bw = box & stri
        axes[row, j].scatter(xy[bw, 0], -xy[bw, 1], s=1.2, c=[col[w] for w in obs[key].values[bw]], lw=0, rasterized=True)
    for ax in axes[:, j]:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for q in ax.spines.values(): q.set_visible(False)
    axes[0, j].set_title(f"{s} ({obs.condition[m].iloc[0]})", fontsize=9)
viz.legend_outside(axes[0, -1], cmap, "cell type"); viz.legend_outside(axes[1, -1], pcol, "compartment"); viz.legend_outside(axes[2, -1], ncol_, "focused niche")
fig.suptitle("Caudate putamen, one hemisphere: cell types / bundles vs matrix / focused striatal niches", x=0.02, ha="left")
fig.savefig(f"{OUT}/striatum_zoom_maps.png"); plt.close(fig)
fu.metric_figure(tab, REG, ["pct OPC", "pct Oligodendrocyte", "pct Microglia", "pct Lesion glia (Gfap+ Olig2+)", "pct Astrocyte Gfap-high / qNSC", "pct progenitors Mki67+", "pct OPC Gpr17/Enpp6+", "oligodendrocyte mean Mbp", "microglia mean Cybb", "microglia mean H2-Eb1", "microglia mean Spp1"],
                 f"{OUT}/striatum_metrics_matrix_bundles.png", "Caudate putamen: matrix vs fibre bundles, one dot per section")
print("done")
