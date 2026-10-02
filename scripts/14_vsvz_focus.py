"""Focused V-SVZ analysis (Xenium), mimicking the paper's ROI. Lining = large spatial clusters of ependymal cells away
from the midline (lateral ventricles). V-SVZ = cells within 75 µm of the lining (choroid plexus excluded). Wall from
tissue context within 150 µm: white-matter glia (oligodendrocyte/OPC/microglia/lesion glia) -> dorsal; striatal MSN -> lateral;
otherwise medial (septal)."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common, focus_utils as fu
ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/vsvz_focus"; os.makedirs(OUT, exist_ok=True)
H5 = f"{RES}/xenium_all.h5ad"; R_SVZ = 75.0
d = common.read_light(H5, keys=("obs", "var"), obsm=("spatial",)); obs, var, xy = d["obs"], d["var"], d["obsm"]["spatial"]
obs["sample_id"] = obs.sample_id.astype(str); ct = obs.cell_type.astype(str).values
cnt = fu.read_genes(H5, var, fu.STATE_GENES, len(obs)); cnt.index = obs.index

lining = fu.ependymal_lining(obs, xy)
dist = np.full(len(obs), np.inf)
for s in obs.sample_id.unique():
    m = np.where((obs.sample_id == s).values)[0]; ep = m[lining[m]]
    if len(ep): dist[m], _ = cKDTree(xy[ep]).query(xy[m])
svz = (dist <= R_SVZ) & (ct != "Choroid plexus")
ctx = fu.context_fractions(obs, xy, {"wm": ["Oligodendrocyte", "OPC", "Microglia", "Lesion glia (Gfap+ Olig2+)"], "msn": ["Striatal MSN"]}, r=150.0)
wall = np.where(~svz, "not V-SVZ", np.where((ctx.wm > ctx.msn) & (ctx.wm > 0.10), "dorsal V-SVZ", np.where(ctx.msn > 0.10, "lateral V-SVZ", "medial V-SVZ")))
obs["vsvz_wall"] = wall; obs["dist_to_lining"] = dist
print(pd.crosstab(obs.sample_id, obs.vsvz_wall).to_string())
# geometric sanity: dorsal should lie above (smaller y) lateral within each section
chk = obs[svz].assign(y=xy[svz, 1]).groupby(["sample_id", "vsvz_wall"], observed=True).y.median().unstack()
print("\nmedian y (smaller = more dorsal):\n" + chk.round(0).to_string())

TYPES = ["Ependymal", "Astrocyte Gfap-high / qNSC", "Astrocyte", "Neuroblast", "OPC", "Oligodendrocyte", "Microglia", "Lesion glia (Gfap+ Olig2+)", "Endothelial", "Striatal MSN"]
REG = ["dorsal V-SVZ", "lateral V-SVZ"]
tab = fu.metrics_table(obs, cnt, "vsvz_wall", REG, TYPES); tab.to_csv(f"{OUT}/vsvz_composition_per_section.csv", index=False)
res = fu.contrasts_table(tab, REG); res.to_csv(f"{OUT}/vsvz_contrasts.csv", index=False)
pd.set_option("display.width", 250)
for e, g1, g2 in fu.CONTRASTS:
    q = res[res.experiment == e]
    piv = q.pivot(index="metric", columns="region", values=["mean1", "mean2", "p_welch"]).round(2)
    piv.columns = [f"{a.replace('mean1', g1).replace('mean2', g2)} | {b.split()[0]}" for a, b in piv.columns]
    print(f"\n=== {e}: {g1} vs {g2} (per-section means)\n" + piv.to_string())

lab, frac, enr, sil = fu.focused_niches(obs, xy, svz, TYPES, r=40.0, prefix="S")
obs["svz_niche"] = "not V-SVZ"; obs.loc[obs.index[svz], "svz_niche"] = lab
grp = obs.experiment.astype(str) + ":" + obs.condition.astype(str)
g = pd.crosstab(grp[svz], obs.svz_niche[svz], normalize="index") * 100
print(f"\nfocused V-SVZ niches k={len(frac)} (best silhouette {max(sil.values()):.2f}); % of V-SVZ cells per group:\n" + g.round(1).T.to_string())
print("\nwall make-up of each focused niche (%):\n" + (pd.crosstab(obs.svz_niche[svz], obs.vsvz_wall[svz], normalize="index") * 100).round(1).to_string())
g.round(2).to_csv(f"{OUT}/vsvz_focused_niche_fraction_by_group.csv"); frac.round(4).to_csv(f"{OUT}/vsvz_focused_niche_composition.csv")
pd.crosstab([obs.sample_id[svz], obs.condition[svz]], obs.svz_niche[svz], normalize="index").round(4).to_csv(f"{OUT}/vsvz_focused_niche_fraction_per_section.csv")
obs[svz][["sample_id", "experiment", "condition", "cell_type", "vsvz_wall", "svz_niche", "dist_to_lining"]].assign(x=xy[svz, 0], y=xy[svz, 1]).to_csv(f"{OUT}/vsvz_cells.csv.gz")

cmap = json.load(open(f"{RES}/celltype_colors.json")); wcol = {"dorsal V-SVZ": viz.CAT8[0], "lateral V-SVZ": viz.CAT8[1], "medial V-SVZ": viz.CAT8[3]}
ncol_ = viz.cmap_for(sorted(set(lab)))
reps = ["NoRecov_Cntl1", "NoRecov_CupRap1", "Recov_Cntl1", "Recov_CupRap1", "Inf_BSA1", "Inf_OSM1"]
fig, axes = plt.subplots(3, len(reps), figsize=(3.6 * len(reps), 11))
for j, s in enumerate(reps):
    m = (obs.sample_id == s).values; pts = xy[m & svz]
    if not len(pts): continue
    cx = np.median(xy[m, 0]); left = pts[pts[:, 0] < cx]
    x0, x1 = left[:, 0].min() - 200, left[:, 0].max() + 200; y0, y1 = left[:, 1].min() - 200, left[:, 1].max() + 200
    box = m & (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    axes[0, j].scatter(xy[box, 0], -xy[box, 1], s=2, c=[cmap[c] for c in ct[box]], lw=0, rasterized=True)
    for row, col, key in [(1, wcol, "vsvz_wall"), (2, ncol_, "svz_niche")]:
        axes[row, j].scatter(xy[box, 0], -xy[box, 1], s=1, c="#e1e0d9", lw=0, rasterized=True); bw = box & svz
        axes[row, j].scatter(xy[bw, 0], -xy[bw, 1], s=3, c=[col[w] for w in obs[key].values[bw]], lw=0, rasterized=True)
    for ax in axes[:, j]:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for q in ax.spines.values(): q.set_visible(False)
    axes[0, j].set_title(f"{s} ({obs.condition[m].iloc[0]})", fontsize=9)
viz.legend_outside(axes[0, -1], cmap, "cell type"); viz.legend_outside(axes[1, -1], wcol, "V-SVZ wall"); viz.legend_outside(axes[2, -1], ncol_, "focused niche")
fig.suptitle("Lateral ventricle, one hemisphere: cell types / V-SVZ wall / focused V-SVZ niches", x=0.02, ha="left")
fig.savefig(f"{OUT}/vsvz_zoom_maps.png"); plt.close(fig)
fu.metric_figure(tab, REG, ["pct OPC", "pct Oligodendrocyte", "pct Microglia", "pct Neuroblast", "pct Astrocyte Gfap-high / qNSC", "pct progenitors Mki67+", "pct progenitors Egfr+", "pct progenitors Ascl1+", "microglia mean Cybb", "microglia mean H2-Eb1", "microglia mean Mki67"],
                 f"{OUT}/vsvz_metrics_dorsal_lateral.png", "V-SVZ (≤75 µm from the lateral-ventricle lining): dorsal vs lateral wall, one dot per section")
print("done")
