"""Downstream niche outputs shared by the k-means (03_niche.py) and CellCharter (05_cellcharter.py) steps:
niche x cell-type enrichment, abundance per section + condition contrasts, spatial maps, composition by condition."""
import itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import viz

CONTRASTS = [("no_recovery", "CupRap", "Control"), ("recovery", "CupRap", "Control"), ("infusion", "OSM_infused", "BSA_infused")]


def name_niches(frac, overall, prefix="N"):
    """Name niche i as '<prefix>i: <dominant type> | <most enriched type>' from a niche x cell-type fraction table."""
    enr = np.log2((frac + 1e-4) / (overall.reindex(frac.columns) + 1e-4))
    names = {}
    for n in frac.index:
        dom = frac.loc[n].idxmax()
        top_enr = [t for t in enr.loc[n].sort_values(ascending=False).index if t != dom][0]
        names[n] = f"{prefix}{n}: {dom} | {top_enr}"
    return names


def downstream(adata, NK, OUT, samples, log=print, CT_KEY="cell_type", method="k-means on neighbourhood composition", contrasts=None):
    order = samples.sample_id.tolist()
    meta = samples.set_index("sample_id")
    contrasts = contrasts or CONTRASTS
    cts = list(adata.obs[CT_KEY].cat.categories)
    k_best = len(adata.obs[NK].cat.categories)
    # ------------------------------------------------------------------ 4 niche x cell-type enrichment
    ct_by_niche = pd.crosstab(adata.obs[NK], adata.obs[CT_KEY])
    frac = ct_by_niche.div(ct_by_niche.sum(1), axis=0)
    enrich = np.log2((frac + 1e-4) / (adata.obs[CT_KEY].value_counts(normalize=True).reindex(frac.columns) + 1e-4))
    enrich.to_csv(f"{OUT}/niche_celltype_log2enrichment.csv"); frac.to_csv(f"{OUT}/niche_celltype_fraction.csv")
    fig, ax = plt.subplots(figsize=(0.55 * len(cts) + 3, 0.45 * k_best + 2))
    v = np.nanmax(np.abs(enrich.values.clip(-4, 4)))
    im = ax.imshow(enrich.values.clip(-4, 4), cmap=viz.div_cmap(), vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(enrich.columns))); ax.set_xticklabels(enrich.columns, rotation=60, ha="right")
    ax.set_yticks(range(k_best)); ax.set_yticklabels(enrich.index)
    for i, j in itertools.product(range(k_best), range(len(enrich.columns))):
        if abs(enrich.values[i, j]) >= 1:
            ax.text(j, i, f"{enrich.values[i, j]:.1f}", ha="center", va="center", fontsize=6, color=viz.INK)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="log2(niche fraction / overall fraction)")
    ax.set_title("Cell-type enrichment per niche (composition of the cells assigned to each niche)")
    fig.savefig(f"{OUT}/niche_celltype_enrichment_heatmap.png"); plt.close(fig)

    # ------------------------------------------------------------------ 5 niche abundance per section + tests
    ab = pd.crosstab(adata.obs.sample_id, adata.obs[NK], normalize="index").reindex(order)
    ab.to_csv(f"{OUT}/niche_fraction_per_sample.csv")
    rows = []
    for expt, g1, g2 in contrasts:
        s1 = meta.index[(meta.experiment == expt) & (meta.condition == g1)]
        s2 = meta.index[(meta.experiment == expt) & (meta.condition == g2)]
        for n in ab.columns:
            a, b = ab.loc[s1, n].values, ab.loc[s2, n].values
            t, p_t = stats.ttest_ind(a, b, equal_var=False) if min(len(a), len(b)) > 1 else (np.nan, np.nan)
            u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic") if min(len(a), len(b)) > 1 else (np.nan, np.nan)
            rows.append({"experiment": expt, "group1": g1, "group2": g2, "niche": n, "n1": len(a), "n2": len(b),
                         "mean1": a.mean(), "mean2": b.mean(), "log2FC": np.log2((a.mean() + 1e-4) / (b.mean() + 1e-4)),
                         "welch_t": t, "p_welch": p_t, "p_mwu": p_u})
    da = pd.DataFrame(rows); da.to_csv(f"{OUT}/niche_differential_abundance.csv", index=False)
    log("differential niche abundance:\n" + da[["experiment", "niche", "mean1", "mean2", "log2FC", "p_welch"]].round(4).to_string())

    ncmap = viz.cmap_for(adata.obs[NK].cat.categories)
    fig, axes = plt.subplots(1, len(contrasts), figsize=(5 * len(contrasts), 4.6), sharey=True, squeeze=False); axes = axes.ravel()
    for ax, (expt, g1, g2) in zip(axes, contrasts):
        ss = [s for s in order if meta.loc[s, "experiment"] == expt]
        ss = sorted(ss, key=lambda s: (meta.loc[s, "condition"] != g2, s))  # control/BSA first
        sub_ab = ab.loc[ss]
        bottom = np.zeros(len(ss))
        for n in sub_ab.columns:
            ax.bar(range(len(ss)), sub_ab[n], bottom=bottom, color=ncmap[n], width=0.8, edgecolor=viz.SURFACE, lw=1)
            bottom += sub_ab[n].values
        ax.set_xticks(range(len(ss))); ax.set_xticklabels([f"{s}\n{meta.loc[s,'condition']}" for s in ss], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{expt}: {g2} vs {g1}")
    axes[0].set_ylabel("fraction of cells in niche")
    viz.legend_outside(axes[-1], ncmap, "niche")
    fig.suptitle("Niche abundance per section, grouped by experiment", x=0.02, ha="left")
    fig.savefig(f"{OUT}/niche_abundance_per_sample.png"); plt.close(fig)

    # per-experiment log2FC dot chart with p-values
    fig, axes = plt.subplots(1, len(contrasts), figsize=(4.3 * len(contrasts), 0.4 * k_best + 1.5), sharey=True, squeeze=False); axes = axes.ravel()
    for ax, (expt, g1, g2) in zip(axes, contrasts):
        d = da[da.experiment == expt].set_index("niche").reindex(ab.columns)
        ax.axvline(0, color=viz.AXIS, lw=1)
        ax.scatter(d.log2FC, range(k_best), s=45, c=[ncmap[n] for n in d.index], zorder=3)
        for i, (n, r) in enumerate(d.iterrows()):
            if pd.notna(r.p_welch) and r.p_welch < 0.05:
                ax.text(r.log2FC, i + 0.3, f"p={r.p_welch:.3f}", fontsize=6, ha="center", color=viz.INK2)
        ax.set_title(f"{expt}: log2({g1} / {g2})"); ax.set_xlabel("log2 fold change of niche fraction")
        ax.grid(axis="x", color=viz.GRID)
    axes[0].set_yticks(range(k_best)); axes[0].set_yticklabels(ab.columns)
    fig.savefig(f"{OUT}/niche_log2fc_by_experiment.png"); plt.close(fig)

    # ------------------------------------------------------------------ 7 spatial niche maps
    n = len(order); ncol = min(5, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.2), squeeze=False)
    for ax, s in zip(axes.ravel(), order):
        m = (adata.obs.sample_id == s).values
        xy = adata.obsm["spatial"][m]
        ax.scatter(xy[:, 0], -xy[:, 1], s=0.15, c=adata.obs.loc[m, NK].map(ncmap), lw=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{s} ({meta.loc[s,'condition']})", fontsize=8)
        for sp_ in ax.spines.values(): sp_.set_visible(False)
    for ax in axes.ravel()[n:]: ax.axis("off")
    viz.legend_outside(axes[0, -1], ncmap, "niche")
    fig.suptitle(f"Spatial niches ({method})", x=0.02, ha="left")
    fig.savefig(f"{OUT}/spatial_niches_all_sections.png"); plt.close(fig)

    # per-niche cell-type composition within each condition (does niche content shift, not just abundance?)
    rows = []
    for (expt, g1, g2) in contrasts:
        for g in (g1, g2):
            m = (adata.obs.experiment == expt) & (adata.obs.condition == g)
            f = pd.crosstab(adata.obs.loc[m, NK], adata.obs.loc[m, CT_KEY], normalize="index")
            f["experiment"], f["condition"] = expt, g
            rows.append(f.reset_index())
    pd.concat(rows).to_csv(f"{OUT}/niche_composition_by_condition.csv", index=False)

