"""What are the niche-C7 / N5 spots inside the striatum of the recovery CupRap sections?
Defines striatal territory per section (>=25 % striatal MSN within 75 µm), then summarises the white-matter /
microglia niche cells inside it: counts per section, cell-type composition, marker expression vs corpus callosum."""
import os, sys, time
import numpy as np, pandas as pd, anndata as ad, h5py
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common
from anndata.io import read_elem

ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/striatal_lesion_check"; os.makedirs(OUT, exist_ok=True)
H5 = f"{RES}/xenium_all.h5ad"
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
d = common.read_light(H5, keys=("obs", "var"), obsm=("spatial",))
obs, var, xy = d["obs"], d["var"], d["obsm"]["spatial"]
obs["sample_id"] = obs["sample_id"].astype(str)
cup = obs.sample_id.str.startswith(("Recov_", "NoRecov_"))
R = 75.0

msn_frac = np.full(len(obs), np.nan)
for s in obs.sample_id[cup].unique():
    m = np.where((obs.sample_id == s).values)[0]
    tree = cKDTree(xy[m])
    is_msn = (obs.cell_type.values[m] == "Striatal MSN").astype(float)
    nb = tree.query_ball_point(xy[m], r=R)
    msn_frac[m] = [is_msn[n].mean() if len(n) else 0 for n in nb]
obs["striatal_territory"] = msn_frac >= 0.25
log("striatal territory defined")

# white-matter / lesion niche cells inside striatum vs corpus callosum (outside striatum)
wm_cc = obs.niche_cc.astype(str).str.startswith("C7")
les_k6 = obs.niche.astype(str).str.startswith("N5")
rows = []
for s in obs.sample_id[cup].unique():
    m = (obs.sample_id == s).values
    rows.append({"sample_id": s, "condition": obs.condition[m].iloc[0], "experiment": obs.experiment[m].iloc[0],
                 "n_striatal_cells": int(obs.striatal_territory[m].sum()),
                 "C7_in_striatum": int((wm_cc & obs.striatal_territory)[m].sum()),
                 "C7_in_striatum_pct": 100 * (wm_cc & obs.striatal_territory)[m].sum() / max(obs.striatal_territory[m].sum(), 1),
                 "N5_in_striatum": int((les_k6 & obs.striatal_territory)[m].sum()),
                 "N5_in_striatum_pct": 100 * (les_k6 & obs.striatal_territory)[m].sum() / max(obs.striatal_territory[m].sum(), 1),
                 "C7_outside_striatum": int((wm_cc & ~obs.striatal_territory)[m].sum())})
counts = pd.DataFrame(rows).set_index("sample_id"); counts.to_csv(f"{OUT}/wm_niche_in_striatum_per_section.csv")
log("\n" + counts.round(2).to_string())

# cell-type composition of C7 cells inside striatum, by group
grp = obs.experiment.astype(str) + ":" + obs.condition.astype(str)
sel = wm_cc & obs.striatal_territory & cup
comp = pd.crosstab(grp[sel], obs.cell_type[sel], normalize="index") * 100
comp_cc = pd.crosstab(grp[wm_cc & ~obs.striatal_territory & cup], obs.cell_type[wm_cc & ~obs.striatal_territory & cup], normalize="index") * 100
comp.to_csv(f"{OUT}/C7_in_striatum_celltype_composition.csv"); comp_cc.to_csv(f"{OUT}/C7_outside_striatum_celltype_composition.csv")
log("C7 cells INSIDE striatum, cell-type %:\n" + comp.round(1).T.to_string())
log("C7 cells OUTSIDE striatum (corpus callosum etc.), cell-type %:\n" + comp_cc.round(1).T.to_string())

# marker expression (mean normalised counts) in C7 striatal cells vs C7 corpus-callosum cells, per group
genes = [g for g in ["Mbp", "Mog", "Mag", "Opalin", "Enpp6", "Gpr17", "Pdgfra", "Olig2", "Sox10", "Cx3cr1", "Tmem119", "Trem2",
                     "Cd68", "Lgals3", "Spp1", "Cybb", "Igf1", "Gfap", "Aqp4", "Vim", "Ppp1r1b", "Penk"] if g in var.index]
gi = [var.index.get_loc(g) for g in genes]
groups = {}
for tag, mask in [("C7 striatum", wm_cc & obs.striatal_territory), ("C7 outside", wm_cc & ~obs.striatal_territory)]:
    for g in sorted(grp[cup].unique()):
        idx = np.where((mask & (grp == g)).values)[0]
        if len(idx) < 50: continue
        if len(idx) > 20000: idx = np.sort(np.random.default_rng(0).choice(idx, 20000, replace=False))
        X, _ = common.read_rows(H5, idx)
        X = X[:, gi].toarray(); X = np.expm1(X)
        groups[f"{tag} | {g} (n={len(idx)})"] = X.mean(0)
expr = pd.DataFrame(groups, index=genes).T
expr.to_csv(f"{OUT}/C7_marker_expression.csv")
log("mean normalised expression (per 100 counts):\n" + expr.round(2).to_string())

# zoom figure: one recovery CupRap section and one recovery control, striatum, coloured by cell type
cmap = dict(zip(*[json.load(open(f"{RES}/celltype_colors.json")).keys(), json.load(open(f"{RES}/celltype_colors.json")).values()])) if False else None
import json
cmap = json.load(open(f"{RES}/celltype_colors.json"))
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, s in zip(axes, ["Recov_Cntl1", "Recov_CupRap1"]):
    m = ((obs.sample_id == s) & obs.striatal_territory).values
    # bounding box of striatal territory, left hemisphere
    pts = xy[m]; cx = np.median(pts[:, 0])
    box = (pts[:, 0] < cx)
    x0, x1 = np.percentile(pts[box, 0], [1, 99]); y0, y1 = np.percentile(pts[box, 1], [1, 99])
    mm = ((obs.sample_id == s).values & (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
    ax.scatter(xy[mm, 0], -xy[mm, 1], s=1.2, c=obs.cell_type[mm].map(cmap), lw=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{s}: striatum (one hemisphere), cell types")
viz.legend_outside(axes[1], cmap, "cell type")
fig.savefig(f"{OUT}/striatum_zoom_celltypes.png"); plt.close(fig)
log("done")
