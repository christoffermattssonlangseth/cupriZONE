"""Clearer views of the interface test: (a) share of cells in the interface niche as a function of signed distance to the
white/grey border, per aggregation setting; (b) zoomed windows around the corpus callosum with the border drawn."""
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
ct = obs.cell_type.astype(str).values
ctx = fu.context_fractions(obs, xy, {"wm": ["Oligodendrocyte", "OPC", "Microglia", "Lesion glia (Gfap+ Olig2+)"], "gm": ["Excitatory neuron", "Inhibitory neuron", "Striatal MSN", "Neuron (other)"]}, r=50.0)
terr = np.where(ctx.wm.values >= 0.40, "WM", np.where(ctx.gm.values >= 0.45, "GM", "other")); sdist = np.full(len(obs), np.nan)
for s in SECTIONS:
    m = np.where((obs.sample_id == s).values)[0]; w = m[terr[m] == "WM"]; g = m[terr[m] == "GM"]
    dW, _ = cKDTree(xy[w]).query(xy[m]); dG, _ = cKDTree(xy[g]).query(xy[m]); sdist[m] = np.where(terr[m] == "WM", dG, np.where(terr[m] == "GM", -dW, np.nan))
band = np.where(np.isnan(sdist), "other", np.where(sdist > RIM, "WM deep", np.where(sdist > 0, "WM rim", np.where(sdist > -RIM, "GM rim", "GM deep"))))
pd.DataFrame({"sample_id": obs.sample_id.values, "territory": terr, "signed_dist_um": sdist, "band": band, "x": xy[:, 0], "y": xy[:, 1]}, index=obs.index).to_csv(f"{OUT}/interface_bands_cells.csv.gz")
spec = pd.read_csv(f"{OUT}/summary_interface_specificity.csv")
order = ["L0 (no aggregation)", "L1", "L2", "L3", "L5", "L3 shuffled within territory"]
is_cup = obs.sample_id.str.contains("CupRap").values
edges = np.arange(-300, 301, 20); centers = (edges[:-1] + edges[1:]) / 2
prof_rows = []
fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
for row, grp_name, gmask in [(0, "control sections", ~is_cup), (1, "CupRap sections", is_cup)]:
    for col, k in enumerate(["k12", "k24", "k36"]):
        ax = axes[row, col]; L = pd.read_csv(f"{OUT}/{k}/niche_labels_by_setting.csv.gz", index_col=0).reindex(obs.index)
        for st, colr, ls in zip(order, viz.CAT8[:5] + [viz.INK2], ["-", "-", "-", "-", "-", "--"]):
            best = int(spec[(spec.k == k) & (spec.setting == st)].niche.iloc[0]); lab = L[st].values
            ok = gmask & ~np.isnan(sdist); bins = np.digitize(sdist[ok], edges) - 1
            share = np.array([(lab[ok][bins == i] == best).mean() * 100 if (bins == i).sum() >= 100 else np.nan for i in range(len(centers))])
            ax.plot(centers, share, ls, color=colr, lw=2 if st != order[-1] else 1.5, label=st.replace(" within territory", ""))
            for c_, v in zip(centers, share): prof_rows.append({"k": k, "setting": st, "group": grp_name, "dist_um": c_, "pct_in_interface_niche": v})
        ax.axvline(0, color=viz.AXIS, lw=1); ax.axvspan(-RIM, RIM, color=viz.GRID, alpha=0.4, lw=0)
        ax.set_title(f"{k}, {grp_name}", fontsize=10); ax.grid(axis="y", color=viz.GRID)
        if row == 1: ax.set_xlabel("signed distance to WM/GM border (µm)   ← grey matter | white matter →")
        if col == 0: ax.set_ylabel("% of cells in the interface niche")
axes[0, -1].legend(fontsize=8, title="aggregation")
fig.suptitle("Interface niche occupancy vs distance from the border (shaded = ±75 µm rim bands). Aggregation creates a peak at the border; it is absent at L0", x=0.02, ha="left")
fig.savefig(f"{OUT}/interface_niche_distance_profiles.png"); plt.close(fig)
pd.DataFrame(prof_rows).to_csv(f"{OUT}/interface_niche_distance_profiles.csv", index=False)

# zoomed windows: corpus callosum / cortex border, 1.6 mm wide, one control and one CupRap section; k=36 L0 vs L3 vs shuffled
k = "k36"; L = pd.read_csv(f"{OUT}/{k}/niche_labels_by_setting.csv.gz", index_col=0).reindex(obs.index)
def window(s):
    m = (obs.sample_id == s).values; wm = m & (terr == "WM") & (band == "WM rim")
    # centre on the lateral corpus callosum: WM-rim cells in the dorsal third, left hemisphere
    p = xy[wm]; cx = np.median(xy[m, 0]); sel = p[(p[:, 0] < cx) & (p[:, 1] < np.percentile(xy[m, 1], 40))]
    c = np.median(sel, 0); return m, c
bcol = {"WM deep": "#104281", "WM rim": "#3987e5", "GM rim": "#f4a98a", "GM deep": "#9a3a12", "other": "#e1e0d9"}
sets = ["L0 (no aggregation)", "L2", "L3", "L3 shuffled within territory"]
fig, axes = plt.subplots(2, len(sets) + 1, figsize=(5.2 * (len(sets) + 1), 10))
for i, s in enumerate(["NoRecov_Cntl1", "NoRecov_CupRap1"]):
    m, c = window(s); half = 800
    box = m & (np.abs(xy[:, 0] - c[0]) < half) & (np.abs(xy[:, 1] - c[1]) < half)
    border = box & (np.abs(sdist) < 12)   # cells right at the border, drawn as the line
    ax = axes[i, 0]; ax.scatter(xy[box, 0], -xy[box, 1], s=6, c=[bcol[b] for b in band[box]], lw=0)
    ax.set_title(f"{s}: bands (±{RIM:.0f} µm)", fontsize=10)
    for j, st in enumerate(sets):
        best = int(spec[(spec.k == k) & (spec.setting == st)].niche.iloc[0]); lab = L[st].values; ax = axes[i, j + 1]
        ax.scatter(xy[box, 0], -xy[box, 1], s=6, c=np.where(terr[box] == "WM", "#b7d3f6", np.where(terr[box] == "GM", "#f1efe8", "#e1e0d9")), lw=0)
        mm = box & (lab == best); ax.scatter(xy[mm, 0], -xy[mm, 1], s=9, c=viz.CAT8[7], lw=0)
        ax.scatter(xy[border, 0], -xy[border, 1], s=2, c=viz.INK, lw=0)
        ax.set_title(f"{k} {st.replace(' within territory', '')}\\ninterface niche (red), border (black), WM (blue)", fontsize=9)
    for ax in axes[i]:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(-c[1] - half, -c[1] + half)
        for q in ax.spines.values(): q.set_visible(False)
    axes[i, 0].plot([c[0] - half + 60, c[0] - half + 260], [-c[1] - half + 60] * 2, color=viz.INK, lw=2); axes[i, 0].text(c[0] - half + 60, -c[1] - half + 90, "200 µm", fontsize=8)
viz.legend_outside(axes[0, 0], bcol, "band")
fig.suptitle("1.6 mm windows on the lateral corpus callosum / cortex border: where the interface niche sits (k = 36)", x=0.02, ha="left")
fig.savefig(f"{OUT}/interface_zoom_k36.png"); plt.close(fig)
# thickness: FWHM-like width of the control profile at L1..L5 (k36), relative to L0
prof = pd.DataFrame(prof_rows); rows = []
for k_ in ["k12", "k24", "k36"]:
    for st in order:
        p = prof[(prof.k == k_) & (prof.setting == st) & (prof.group == "control sections")].dropna()
        if p.empty: continue
        peak = p.pct_in_interface_niche.max(); half = peak / 2; above = p[p.pct_in_interface_niche >= half]
        rows.append({"k": k_, "setting": st, "peak % at border": peak, "peak position (µm)": float(p.loc[p.pct_in_interface_niche.idxmax(), "dist_um"]), "width at half max (µm)": float(above.dist_um.max() - above.dist_um.min() + 20)})
w = pd.DataFrame(rows); w.to_csv(f"{OUT}/interface_niche_profile_width.csv", index=False); print(w.round(1).to_string(index=False)); print("done")
