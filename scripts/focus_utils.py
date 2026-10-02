"""Shared helpers for region-focused analyses (V-SVZ, striatum): gene reads, per-section composition / marker
metrics, condition contrasts, focused niche clustering, metric figure."""
import numpy as np, pandas as pd, h5py
from anndata.io import sparse_dataset
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import viz

CONTRASTS = [("no_recovery", "CupRap", "Control"), ("recovery", "CupRap", "Control"), ("infusion", "OSM_infused", "BSA_infused")]
STATE_GENES = ["Cybb", "H2-Eb1", "Lgals3", "Trem2", "Spp1", "Tmem119", "Igf1", "Osm", "Mki67", "Egfr", "Ascl1", "Gpr17", "Enpp6", "Pdgfra", "Mbp", "Gfap", "Thbs4", "Dlx2", "Cd24a"]


def read_genes(h5, var, genes, n):
    genes = [g for g in genes if g in var.index]; gi = [var.index.get_loc(g) for g in genes]
    M = np.zeros((n, len(genes)), np.float32)
    with h5py.File(h5) as f:
        C = sparse_dataset(f["layers/counts"])
        for s in range(0, n, 200_000): M[s:s+200_000] = C[s:s+200_000][:, gi].toarray()
    return pd.DataFrame(M, columns=genes)


def context_fractions(obs, xy, groups, r=150.0):
    """Per cell, fraction of neighbours (within r µm, same section) belonging to each named cell-type group."""
    ct = obs.cell_type.astype(str).values; out = {k: np.zeros(len(obs)) for k in groups}
    for s in obs.sample_id.unique():
        m = np.where((obs.sample_id == s).values)[0]
        nb = cKDTree(xy[m]).query_ball_point(xy[m], r=r)
        for k, types in groups.items():
            ind = np.isin(ct[m], types).astype(float)
            out[k][m] = [ind[n].mean() if len(n) else 0 for n in nb]
    return pd.DataFrame(out, index=obs.index)


def ependymal_lining(obs, xy, min_cells=80, eps=60.0, midline_excl=250.0):
    """Spatial clusters of ependymal cells = ventricle linings; drop small clusters (mis-annotated singles) and
    clusters at the section midline (third ventricle). Returns boolean mask of lining cells."""
    ct = obs.cell_type.astype(str).values; keep = np.zeros(len(obs), bool)
    for s in obs.sample_id.unique():
        m = np.where((obs.sample_id == s).values)[0]; mid = np.median(xy[m, 0])
        ep = m[(ct[m] == "Ependymal") & (np.abs(xy[m, 0] - mid) > midline_excl)]   # drop third-ventricle / midline cells first
        if len(ep) < min_cells: continue
        lab = DBSCAN(eps=eps, min_samples=5).fit_predict(xy[ep])
        for l in set(lab) - {-1}:
            cl = ep[lab == l]
            if len(cl) >= min_cells:
                keep[cl] = True
    return keep


def metrics_table(obs, cnt, region_col, regions, types, min_cells=50):
    ct = obs.cell_type.astype(str).values; rows = []
    prog_types = ["Astrocyte Gfap-high / qNSC", "Neuroblast", "OPC", "Astrocyte"]
    for s in obs.sample_id.unique():
        for w in regions:
            m = (obs.sample_id == s).values & (obs[region_col].values == w)
            if m.sum() < min_cells: continue
            r = {"sample_id": s, "experiment": obs.experiment[m].iloc[0], "condition": obs.condition[m].iloc[0], "region": w, "n_cells": int(m.sum())}
            for t in types: r[f"pct {t}"] = 100 * (ct[m] == t).mean()
            mg = m & (ct == "Microglia")
            for g in ["Cybb", "H2-Eb1", "Lgals3", "Spp1", "Igf1", "Osm", "Mki67"]:
                if g in cnt: r[f"microglia mean {g}"] = cnt.loc[mg, g].mean() if mg.sum() >= 10 else np.nan
            prog = m & np.isin(ct, prog_types)
            for g in ["Mki67", "Egfr", "Ascl1"]:
                if g in cnt: r[f"pct progenitors {g}+"] = 100 * (cnt.loc[prog, g] >= 1).mean() if prog.sum() >= 10 else np.nan
            opc = m & (ct == "OPC")
            if "Gpr17" in cnt: r["pct OPC Gpr17/Enpp6+"] = 100 * ((cnt.loc[opc, "Gpr17"] + cnt.loc[opc, "Enpp6"]) >= 1).mean() if opc.sum() >= 10 else np.nan
            ol = m & (ct == "Oligodendrocyte")
            if "Mbp" in cnt: r["oligodendrocyte mean Mbp"] = cnt.loc[ol, "Mbp"].mean() if ol.sum() >= 10 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def contrasts_table(tab, regions, contrasts=CONTRASTS):
    metrics = [c for c in tab.columns if c.startswith(("pct", "microglia", "oligodendrocyte"))]; res = []
    for e, g1, g2 in contrasts:
        for w in regions:
            q = tab[(tab.experiment == e) & (tab.region == w)]
            for mtr in metrics:
                a, b = q[q.condition == g1][mtr].dropna().values, q[q.condition == g2][mtr].dropna().values
                if len(a) < 2 or len(b) < 2: continue
                t, p = stats.ttest_ind(a, b, equal_var=False)
                res.append({"experiment": e, "region": w, "metric": mtr, "group1": g1, "group2": g2, "mean1": a.mean(), "mean2": b.mean(),
                            "log2_ratio": np.log2((a.mean() + 1e-3) / (b.mean() + 1e-3)), "p_welch": p})
    return pd.DataFrame(res)


def focused_niches(obs, xy, mask, types, r=40.0, k_range=range(3, 9), prefix="F"):
    """k-means on local (r µm) cell-type composition, restricted to cells in mask. Returns labels (str) for mask cells + tables."""
    ct = obs.cell_type.astype(str).values; idx = np.where(mask)[0]; comp = np.zeros((len(idx), len(types)))
    for s in obs.sample_id.unique():
        m_all = np.where((obs.sample_id == s).values)[0]; m_in = np.where((obs.sample_id == s).values & mask)[0]
        if not len(m_in): continue
        nb = cKDTree(xy[m_all]).query_ball_point(xy[m_in], r=r); loc = np.searchsorted(idx, m_in)
        for i, n in zip(loc, nb):
            c = ct[m_all][n]; comp[i] = [(c == t).mean() for t in types]
    step = max(1, len(idx) // 30_000)
    sil = {k: silhouette_score(comp[::step], KMeans(k, n_init=5, random_state=0).fit_predict(comp[::step])) for k in k_range}
    kb = max(sil, key=sil.get); lab = KMeans(kb, n_init=10, random_state=0).fit_predict(comp)
    frac = pd.DataFrame([[(ct[idx][lab == i] == t).mean() for t in types] for i in range(kb)], columns=types)
    overall = pd.Series([(ct[idx] == t).mean() for t in types], index=types)
    enr = np.log2((frac + 1e-3) / (overall + 1e-3))
    names = {i: f"{prefix}{i}: {frac.loc[i].idxmax()} | {[t for t in enr.loc[i].sort_values(ascending=False).index if t != frac.loc[i].idxmax()][0]}" for i in range(kb)}
    return np.array([names[l] for l in lab]), frac.rename(index=names), enr.rename(index=names), sil


def metric_figure(tab, regions, keys, fname, title, contrasts=CONTRASTS):
    fig, axes = plt.subplots(len(contrasts), len(keys), figsize=(2.1 * len(keys), 2.6 * len(contrasts)), squeeze=False)
    for i, (e, g1, g2) in enumerate(contrasts):
        for j, kname in enumerate(keys):
            ax = axes[i, j]; q = tab[tab.experiment == e]
            for wi, w in enumerate(regions):
                for ci, cond in enumerate([g2, g1]):
                    v = q[(q.region == w) & (q.condition == cond)][kname].dropna().values if kname in q else np.array([]); x = wi * 2.2 + ci
                    ax.scatter(np.full(len(v), x) + np.linspace(-0.12, 0.12, len(v)), v, s=18, color=viz.CAT8[ci], zorder=3, edgecolor=viz.SURFACE, lw=0.5)
                    if len(v): ax.hlines(v.mean(), x - 0.3, x + 0.3, color=viz.CAT8[ci], lw=2)
            ax.set_xticks([wi * 2.2 + 0.5 for wi in range(len(regions))]); ax.set_xticklabels([r.split(" ")[0] for r in regions], fontsize=7); ax.tick_params(axis="y", labelsize=7)
            if i == 0: ax.set_title(kname.replace("pct ", "% ").replace("microglia mean ", "µglia ").replace("oligodendrocyte mean ", "oligo "), fontsize=8)
            if j == 0: ax.set_ylabel(f"{e}\n{g2} (blue) vs {g1} (orange)", fontsize=8)
            ax.grid(axis="y", color=viz.GRID)
    fig.suptitle(title, x=0.02, ha="left"); fig.tight_layout(); fig.savefig(fname); plt.close(fig)
