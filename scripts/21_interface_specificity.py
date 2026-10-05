"""Re-score the interface test: for every (k, aggregation setting) pick the niche with the highest interface specificity
(min(rim enrichment WM side, GM side) / deep-WM enrichment, niches >= 2000 cells) rather than the largest rim share; report its
size, how much of the interface it covers, and map it for k=24."""
import os, sys, glob
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common, focus_utils as fu
ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/interface_artefact"; RIM = 75.0
SECTIONS = ["NoRecov_Cntl1", "NoRecov_Cntl2", "NoRecov_Cntl3", "NoRecov_CupRap1", "NoRecov_CupRap2", "NoRecov_CupRap3"]
d = common.read_light(f"{RES}/xenium_all.h5ad", keys=("obs",), obsm=("spatial",))
keep = d["obs"].sample_id.astype(str).isin(SECTIONS).values; obs = d["obs"][keep].copy(); xy = d["obsm"]["spatial"][keep]; obs["sample_id"] = obs.sample_id.astype(str)
ctx = fu.context_fractions(obs, xy, {"wm": ["Oligodendrocyte", "OPC", "Microglia", "Lesion glia (Gfap+ Olig2+)"], "gm": ["Excitatory neuron", "Inhibitory neuron", "Striatal MSN", "Neuron (other)"]}, r=50.0)
terr = np.where(ctx.wm.values >= 0.40, "WM", np.where(ctx.gm.values >= 0.45, "GM", "other")); sdist = np.full(len(obs), np.nan)
for s in SECTIONS:   # keep only large contiguous white-matter sheets (corpus callosum / external capsule), not striatal bundles
    m = np.where(((obs.sample_id == s).values) & (terr == "WM"))[0]
    lab = DBSCAN(eps=40.0, min_samples=5).fit_predict(xy[m]); sizes = pd.Series(lab).value_counts()
    small = m[~np.isin(lab, sizes[sizes >= 2500].index) | (lab == -1)]; terr[small] = "other"
for s in SECTIONS:
    m = np.where((obs.sample_id == s).values)[0]; w = m[terr[m] == "WM"]; g = m[terr[m] == "GM"]
    dW, _ = cKDTree(xy[w]).query(xy[m]); dG, _ = cKDTree(xy[g]).query(xy[m]); sdist[m] = np.where(terr[m] == "WM", dG, np.where(terr[m] == "GM", -dW, np.nan))
band = np.where(np.isnan(sdist), "other", np.where(sdist > RIM, "WM deep", np.where(sdist > 0, "WM rim", np.where(sdist > -RIM, "GM rim", "GM deep"))))
print(pd.crosstab(obs.sample_id.values, band).to_string()); overall = pd.Series(band).value_counts(normalize=True); is_cup = obs.sample_id.str.contains("CupRap").values
rows, picks = [], {}
for f in sorted(glob.glob(f"{OUT}/k*/niche_labels_by_setting.csv.gz")):
    k = os.path.basename(os.path.dirname(f)); L = pd.read_csv(f, index_col=0).reindex(obs.index)
    for setting in L.columns:
        lab = L[setting].values; tab = pd.crosstab(lab, band); tab = tab[tab.sum(1) >= 2000]
        frac = tab.div(tab.sum(1), axis=0); enr = frac / overall.reindex(frac.columns)
        spec = enr[["WM rim", "GM rim"]].min(1) / enr["WM deep"].clip(lower=0.05); best = spec.idxmax()
        picks[(k, setting)] = (best, lab)
        rows.append({"k": k, "setting": setting, "niche": int(best), "n cells": int(tab.loc[best].sum()), "interface specificity": spec[best],
                     "enr WM rim": enr.loc[best, "WM rim"], "enr GM rim": enr.loc[best, "GM rim"], "enr WM deep": enr.loc[best, "WM deep"], "enr GM deep": enr.loc[best, "GM deep"],
                     "% of niche in rim bands": 100 * frac.loc[best, ["WM rim", "GM rim"]].sum(),
                     "% of WM-rim cells covered (controls)": 100 * ((lab == best) & (band == "WM rim") & ~is_cup).sum() / ((band == "WM rim") & ~is_cup).sum(),
                     "% of WM-rim cells covered (CupRap)": 100 * ((lab == best) & (band == "WM rim") & is_cup).sum() / ((band == "WM rim") & is_cup).sum(),
                     "% of niche in CupRap": 100 * is_cup[lab == best].mean(),
                     "median |dist to border| of niche cells (µm)": float(np.nanmedian(np.abs(sdist[(lab == best) & ~np.isnan(sdist)]))),
                     "overall median |dist| (µm)": float(np.nanmedian(np.abs(sdist)))})
res = pd.DataFrame(rows); pd.set_option("display.width", 260); print(res.round(2).to_string(index=False)); res.round(3).to_csv(f"{OUT}/summary_interface_specificity.csv", index=False)
# maps for k24: L0, L2, L3, shuffled, one control + one CupRap section
k = "k24"; sets = [s for s in ["L0 (no aggregation)", "L2", "L3", "L3 shuffled within territory"] if (k, s) in picks]
fig, axes = plt.subplots(2, len(sets), figsize=(5 * len(sets), 9), squeeze=False)
for i, s in enumerate(["NoRecov_Cntl1", "NoRecov_CupRap1"]):
    m = (obs.sample_id == s).values
    for j, st in enumerate(sets):
        best, lab = picks[(k, st)]; ax = axes[i, j]
        ax.scatter(xy[m, 0], -xy[m, 1], s=0.3, c=np.where(np.isin(band[m], ["WM rim", "WM deep"]), "#b7d3f6", "#e1e0d9"), lw=0, rasterized=True)
        mm = m & (lab == best); ax.scatter(xy[mm, 0], -xy[mm, 1], s=0.8, c=viz.CAT8[7], lw=0, rasterized=True)
        ax.set_title(f"{s}\n{k} {st}: most interface-specific niche ({(lab[m] == best).mean()*100:.1f} % of cells)", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for q in ax.spines.values(): q.set_visible(False)
fig.suptitle("Most interface-specific niche per setting (red) over white matter (blue) — k=24", x=0.02, ha="left")
fig.savefig(f"{OUT}/interface_specific_niche_maps_k24.png"); plt.close(fig); print("done")
