"""Is there a distinct immune / disease-associated oligodendroglial state? Subcluster the oligodendrocyte lineage
(Oligodendrocyte + OPC + lesion glia) within each Xenium batch using that batch's full panel (so C4b, Il6, S100b in the
2023 recovery batch; Socs3, Osmr, Tnfrsf1a, Tnf in the 2024 acute+infusion batch are included)."""
import os, sys, json
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common
ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/oligo_subcluster"; os.makedirs(OUT, exist_ok=True)
obs = common.read_light(f"{RES}/xenium_all.h5ad", keys=("obs",), obsm=())["obs"]; obs["sample_id"] = obs.sample_id.astype(str)
samples = pd.read_csv(f"{ROOT}/scripts/samples.tsv", sep="\t")
LIN = ["Oligodendrocyte", "OPC", "Lesion glia (Gfap+ Olig2+)"]
IMMUNE = ["C4b", "H2-Eb1", "Lgals3", "Socs3", "Osmr", "Tnfrsf1a", "Tnf", "Il6", "Il33", "Il6st", "Cd52", "S100b", "Trem2", "Cybb", "Igf1"]
batches = {"recovery_2023": samples[samples.experiment == "recovery"].sample_id.tolist(),
           "acute_infusion_2024": samples[samples.experiment != "recovery"].sample_id.tolist()}
for bname, sids in batches.items():
    ads = []
    for s in sids:
        a = sc.read_10x_mtx(f"{ROOT}/data/geo/{s}", var_names="gene_symbols", make_unique=True)
        a = a[:, a.var.feature_types == "Gene Expression"].copy(); a.obs_names = [f"{s}:{c}" for c in a.obs_names]
        keep = obs.index[(obs.sample_id == s) & obs.cell_type.isin(LIN)]; a = a[a.obs_names.isin(keep)].copy()
        a.obs = a.obs.join(obs.loc[a.obs_names, ["sample_id", "condition", "experiment", "cell_type", "niche_cc", "niche"]]); ads.append(a)
    A = ad.concat(ads, join="inner"); A.layers["counts"] = A.X.copy()
    A.obs["lesion"] = A.obs.niche_cc.astype(str).str.startswith("C7") | A.obs.niche.astype(str).str.startswith("N5")
    sc.pp.filter_cells(A, min_counts=20); sc.pp.normalize_total(A, target_sum=100); sc.pp.log1p(A)
    sc.pp.pca(A, n_comps=20, random_state=0); sc.pp.neighbors(A, n_neighbors=15, random_state=0)
    sc.tl.leiden(A, resolution=0.8, flavor="igraph", n_iterations=2, random_state=0, key_added="sub")
    print(f"\n######## {bname}: {A.n_obs:,} lineage cells, {A.n_vars} genes, {A.obs['sub'].nunique()} subclusters")
    # cluster table: size, condition share, lineage make-up, immune-gene detection
    genes = [g for g in IMMUNE if g in A.var_names]
    X = A[:, genes].layers["counts"].toarray() >= 1
    tab = pd.DataFrame({"n": A.obs.groupby("sub", observed=True).size()})
    cond = pd.crosstab(A.obs["sub"], A.obs.condition, normalize="index") * 100; tab = tab.join(cond.add_prefix("% "))
    tab["% in lesion niche"] = A.obs.groupby("sub", observed=True).lesion.mean() * 100
    lin = pd.crosstab(A.obs["sub"], A.obs.cell_type, normalize="index") * 100; tab = tab.join(lin.add_prefix("% "))
    det = pd.DataFrame(X, columns=genes, index=A.obs_names).groupby(A.obs["sub"].values).mean() * 100; tab = tab.join(det.add_prefix("det% "))
    sc.tl.rank_genes_groups(A, "sub", method="wilcoxon", n_genes=8)
    tab["top genes"] = [", ".join(A.uns["rank_genes_groups"]["names"][c][:8]) for c in tab.index]
    tab.round(1).to_csv(f"{OUT}/{bname}_subclusters.csv")
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    cols = ["n"] + [c for c in tab.columns if c.startswith("% ") and c.split("% ")[1] in ("CupRap", "Control", "OSM_infused", "BSA_infused")] + ["% in lesion niche", "% Oligodendrocyte", "% OPC"] + [f"det% {g}" for g in genes if g in ("C4b", "H2-Eb1", "Lgals3", "Socs3", "Osmr", "Il33", "Tnfrsf1a")] + ["top genes"]
    print(tab[[c for c in cols if c in tab.columns]].round(1).to_string())
    # immune score per cell = number of detected immune genes among the batch's available set (excluding Il33, Il6st, Tnfrsf1a which are constitutive)
    score_genes = [g for g in genes if g not in ("Il33", "Il6st", "Tnfrsf1a", "S100b", "Igf1", "Trem2", "Cybb")]
    A.obs["immune_score"] = (A[:, score_genes].layers["counts"].toarray() >= 1).sum(1)
    print(f"immune score genes: {score_genes}")
    summ = A.obs.groupby(["condition", "lesion", "cell_type"], observed=True).immune_score.agg(n="size", mean="mean", pct_ge2=lambda x: 100 * (x >= 2).mean())
    print("immune score (n detected immune genes) by condition / lesion niche / lineage type:\n" + summ.round(2).to_string())
    summ.round(3).to_csv(f"{OUT}/{bname}_immune_score.csv")
    sub = sc.pp.subsample(A, n_obs=min(60_000, A.n_obs), random_state=0, copy=True); sc.tl.umap(sub, random_state=0)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5)); xy = sub.obsm["X_umap"]
    cm = viz.cmap_for(sorted(sub.obs["sub"].unique(), key=int)); axes[0].scatter(xy[:, 0], xy[:, 1], s=0.5, c=sub.obs["sub"].map(cm), lw=0, rasterized=True); viz.legend_outside(axes[0], cm, "subcluster", ncol=2); axes[0].set_title("subcluster")
    cc_ = viz.cmap_for(sorted(sub.obs.condition.unique())); axes[1].scatter(xy[:, 0], xy[:, 1], s=0.5, c=sub.obs.condition.map(cc_), lw=0, rasterized=True); viz.legend_outside(axes[1], cc_, "condition"); axes[1].set_title("condition")
    for ax, g in zip(axes[2:], [g for g in ["C4b", "Socs3", "H2-Eb1"] if g in sub.var_names][:2]):
        v = sub[:, g].X.toarray().ravel(); o = np.argsort(v); ax.scatter(xy[o, 0], xy[o, 1], s=0.5, c=v[o], cmap=viz.seq_cmap(), lw=0, rasterized=True); ax.set_title(f"{g} (log-normalised)")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Oligodendrocyte lineage subclustering, {bname} batch ({A.n_obs:,} cells, {A.n_vars} genes)", x=0.02, ha="left")
    fig.savefig(f"{OUT}/{bname}_umap.png"); plt.close(fig)
    A.obs[["sample_id", "condition", "cell_type", "sub", "lesion", "immune_score"]].to_csv(f"{OUT}/{bname}_cells.csv.gz")
print("done")
