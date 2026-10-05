"""Does neighbourhood aggregation manufacture a 'rim' niche at the white/grey-matter interface?
Acute batch (3 control + 3 CupRap sections). (1) geometric interface bands; (2) CellCharter at fixed k with 0/1/2/3/5
aggregation layers; (3) shuffle control (permute embeddings within territory); (4) cell-intrinsic oligodendrocyte
expression across bands."""
import os, sys, json, logging, warnings
import numpy as np, pandas as pd, anndata as ad, squidpy as sq, cellcharter as cc
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common, focus_utils as fu
warnings.filterwarnings("ignore")
for n in ["lightning", "pytorch_lightning", "lightning.pytorch"]: logging.getLogger(n).setLevel(logging.ERROR)
ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/interface_artefact/k{os.environ.get('K', 12)}"; os.makedirs(OUT, exist_ok=True)
H5 = f"{RES}/xenium_all.h5ad"; K = int(os.environ.get("K", 12)); RIM = 75.0
TR = dict(accelerator="cpu", enable_progress_bar=False, enable_model_summary=False, logger=False)
SECTIONS = ["NoRecov_Cntl1", "NoRecov_Cntl2", "NoRecov_Cntl3", "NoRecov_CupRap1", "NoRecov_CupRap2", "NoRecov_CupRap3"]
d = common.read_light(H5, keys=("obs", "var"), obsm=("spatial", "X_pca"))
keep = d["obs"].sample_id.astype(str).isin(SECTIONS).values
obs = d["obs"][keep].copy(); xy = d["obsm"]["spatial"][keep]; pca = d["obsm"]["X_pca"][keep].astype(np.float32); var = d["var"]
obs["sample_id"] = obs.sample_id.astype(str); ct = obs.cell_type.astype(str).values
genes = [g for g in ["Il33", "Opalin", "Mog", "Mbp", "Mag", "Gjc3", "Enpp6", "Gpr17", "Pdgfra", "Klk6", "Aspa", "Ptgds", "Anln", "Egr1", "Mal", "Apod", "Trem2", "Cx3cr1"] if g in var.index]
full_idx = np.where(keep)[0]
cnt_all = fu.read_genes(H5, var, genes, len(d["obs"]))
cnt = cnt_all.iloc[full_idx].reset_index(drop=True); del cnt_all

# ---------------- 1. territories and interface bands (independent of niches)
WM_TYPES = ["Oligodendrocyte", "OPC", "Microglia", "Lesion glia (Gfap+ Olig2+)"]
GM_TYPES = ["Excitatory neuron", "Inhibitory neuron", "Striatal MSN", "Neuron (other)"]
ctx = fu.context_fractions(obs, xy, {"wm": WM_TYPES, "gm": GM_TYPES}, r=50.0)
terr = np.where(ctx.wm.values >= 0.40, "WM", np.where(ctx.gm.values >= 0.45, "GM", "other"))
sdist = np.full(len(obs), np.nan)
for s in SECTIONS:
    m = np.where((obs.sample_id == s).values)[0]; w = m[terr[m] == "WM"]; g = m[terr[m] == "GM"]
    dW, _ = cKDTree(xy[w]).query(xy[m]); dG, _ = cKDTree(xy[g]).query(xy[m])
    sdist[m] = np.where(terr[m] == "WM", dG, np.where(terr[m] == "GM", -dW, np.nan))   # + inside WM, - inside GM
band = np.where(np.isnan(sdist), "other", np.where(sdist > RIM, "WM deep", np.where(sdist > 0, "WM rim", np.where(sdist > -RIM, "GM rim", "GM deep"))))
obs["territory"], obs["signed_dist"], obs["band"] = terr, sdist, band
print(pd.crosstab(obs.sample_id, obs.band).to_string())

# ---------------- 2. CellCharter with different aggregation depths (fixed k)
A = ad.AnnData(obs=obs[["sample_id"]].copy(), obsm={"spatial": xy, "X_pca": pca}); A.obs["sample_id"] = A.obs.sample_id.astype("category")
sq.gr.spatial_neighbors(A, coord_type="generic", delaunay=True, library_key="sample_id"); cc.gr.remove_long_links(A)
rng = np.random.default_rng(0); fit_idx = np.sort(rng.choice(A.n_obs, min(150_000, A.n_obs), replace=False))
def run_gmm(X, tag):
    B = ad.AnnData(obs=A.obs.copy(), obsm={"X": X})
    g = cc.tl.Cluster(n_clusters=K, random_state=0, trainer_params=TR); g.fit(B[fit_idx].copy(), use_rep="X")
    lab = np.asarray(g.predict(B, use_rep="X")).astype(int); print(f"  {tag}: fitted k={K}", flush=True); return lab
def rim_metrics(lab, tag):
    tab = pd.crosstab(lab, band); frac = tab.div(tab.sum(1), axis=0); overall = pd.Series(band).value_counts(normalize=True)
    enr = frac / overall.reindex(frac.columns)
    rimcols = ["WM rim", "GM rim"]; rim_share_niche = frac[rimcols].sum(1)
    best = rim_share_niche.idxmax()
    return {"setting": tag, "rim niche": int(best), "n cells": int(tab.loc[best].sum()), "% of niche in rim bands": 100 * rim_share_niche[best],
            "% of all rim cells captured": 100 * tab.loc[best, rimcols].sum() / tab[rimcols].values.sum(),
            "enrichment WM rim": enr.loc[best, "WM rim"], "enrichment GM rim": enr.loc[best, "GM rim"], "enrichment WM deep": enr.loc[best, "WM deep"],
            "% of niche in CupRap sections": 100 * obs.sample_id.str.contains("CupRap").values[lab == best].mean()}
settings, labels = [], {}
labels["L0 (no aggregation)"] = run_gmm(pca, "L0"); settings.append(rim_metrics(labels["L0 (no aggregation)"], "L0 (no aggregation)"))
for L in [1, 2, 3, 5]:
    cc.gr.aggregate_neighbors(A, n_layers=L, use_rep="X_pca", out_key="X_cc", sample_key="sample_id")
    labels[f"L{L}"] = run_gmm(A.obsm["X_cc"], f"L{L}"); settings.append(rim_metrics(labels[f"L{L}"], f"L{L}"))
# ---------------- 3. shuffle control at L=3: permute embeddings within (section, territory)
pca_sh = pca.copy()
for s in SECTIONS:
    for t in ["WM", "GM", "other"]:
        m = np.where(((obs.sample_id == s).values) & (terr == t))[0]; pca_sh[m] = pca[rng.permutation(m)]
A.obsm["X_pca_sh"] = pca_sh; cc.gr.aggregate_neighbors(A, n_layers=3, use_rep="X_pca_sh", out_key="X_cc_sh", sample_key="sample_id")
labels["L3 shuffled within territory"] = run_gmm(A.obsm["X_cc_sh"], "L3 shuffled"); settings.append(rim_metrics(labels["L3 shuffled within territory"], "L3 shuffled within territory"))
res = pd.DataFrame(settings); res.to_csv(f"{OUT}/rim_niche_by_aggregation.csv", index=False)
pd.set_option("display.width", 250); print("\nMost rim-concentrated niche per setting (k fixed = %d):\n" % K + res.round(2).to_string(index=False))
lab_df = pd.DataFrame(labels, index=obs.index); lab_df.to_csv(f"{OUT}/niche_labels_by_setting.csv.gz")
# per-section: does the rim niche (L3) exist along lesioned vs non-lesioned border? share of WM-rim cells in the rim niche
best3 = res.set_index("setting").loc["L3", "rim niche"]; l3 = labels["L3"]
per = pd.DataFrame({"% WM-rim cells in rim niche": pd.Series((l3 == best3)[band == "WM rim"]).groupby(obs.sample_id.values[band == "WM rim"]).mean() * 100,
                    "% GM-rim cells in rim niche": pd.Series((l3 == best3)[band == "GM rim"]).groupby(obs.sample_id.values[band == "GM rim"]).mean() * 100,
                    "% WM-deep cells in rim niche": pd.Series((l3 == best3)[band == "WM deep"]).groupby(obs.sample_id.values[band == "WM deep"]).mean() * 100})
print("\nL3 rim niche coverage per section:\n" + per.round(1).to_string()); per.to_csv(f"{OUT}/rim_niche_L3_per_section.csv")

# ---------------- 4. cell-intrinsic oligodendrocyte expression across bands (controls), no aggregation
ol = (ct == "Oligodendrocyte") & obs.sample_id.str.contains("Cntl").values
tab = cnt[ol].groupby(band[ol]).mean().reindex(["WM deep", "WM rim", "GM rim", "GM deep"])
n = pd.Series(band[ol]).value_counts().reindex(tab.index); tab.insert(0, "n oligodendrocytes", n)
print("\nControl oligodendrocytes, mean raw counts per cell by band (cell-intrinsic):\n" + tab.round(2).to_string()); tab.round(3).to_csv(f"{OUT}/oligodendrocyte_expression_by_band_controls.csv")
# composition of the oligodendrocyte-lineage pool by band (controls)
pool = np.isin(ct, ["Oligodendrocyte", "OPC"]) & obs.sample_id.str.contains("Cntl").values
comp = pd.crosstab(band[pool], ct[pool], normalize="index").reindex(["WM deep", "WM rim", "GM rim", "GM deep"]) * 100
print("\nOL/OPC pool composition by band (controls, %):\n" + comp.round(1).to_string())

# ---------------- figure: bands, L0/L3/shuffled rim niche, one control + one CupRap section
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
bcol = {"WM deep": "#104281", "WM rim": "#3987e5", "GM rim": "#f4a98a", "GM deep": "#9a3a12", "other": "#e1e0d9"}
for i, s in enumerate(["NoRecov_Cntl1", "NoRecov_CupRap1"]):
    m = (obs.sample_id == s).values; p = xy[m]
    axes[i, 0].scatter(p[:, 0], -p[:, 1], s=0.3, c=[bcol[b] for b in band[m]], lw=0, rasterized=True); axes[i, 0].set_title(f"{s}: interface bands (±{RIM:.0f} µm)")
    for j, (tag, title) in enumerate([("L0 (no aggregation)", "L0: most rim-concentrated niche"), ("L3", "L3: most rim-concentrated niche"), ("L3 shuffled within territory", "L3 after within-territory shuffle")]):
        b = res.set_index("setting").loc[tag, "rim niche"]; lab = labels[tag]
        axes[i, j + 1].scatter(p[:, 0], -p[:, 1], s=0.3, c="#e1e0d9", lw=0, rasterized=True); mm = m & (lab == b)
        axes[i, j + 1].scatter(xy[mm, 0], -xy[mm, 1], s=0.6, c=viz.CAT8[7], lw=0, rasterized=True); axes[i, j + 1].set_title(f"{title} ({(lab[m] == b).mean()*100:.1f} % of cells)")
    for ax in axes[i]:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for q in ax.spines.values(): q.set_visible(False)
viz.legend_outside(axes[0, 0], bcol, "band")
fig.suptitle(f"Interface artefact test: CellCharter GMM (k={K}) with 0 vs 3 aggregation layers, and after shuffling embeddings within WM / GM", x=0.02, ha="left")
fig.savefig(f"{OUT}/interface_artefact_maps.png"); plt.close(fig)
print("done")
