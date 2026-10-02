"""Are the striatal microglia spots in the recovery CupRap sections the same lesion state as the acute callosal lesion?
Pseudobulk microglia (and all cells) per section x lesion compartment, compare genome(panel)-wide profiles."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import viz, common

ROOT = common.ROOT; RES = f"{ROOT}/results"; OUT = f"{RES}/striatal_lesion_check"
H5 = f"{RES}/xenium_all.h5ad"
d = common.read_light(H5, keys=("obs", "var"), obsm=("spatial",))
obs, var, xy = d["obs"], d["var"], d["obsm"]["spatial"]
obs["sample_id"] = obs.sample_id.astype(str)
cup = obs.sample_id.str.startswith(("Recov_", "NoRecov_")).values

# striatal territory (as in 08) and white-matter/lesion compartment = CellCharter C7 or k-means N5
msn_frac = np.zeros(len(obs))
for s in obs.sample_id[cup].unique():
    m = np.where((obs.sample_id == s).values)[0]
    is_msn = (obs.cell_type.values[m] == "Striatal MSN").astype(float)
    nb = cKDTree(xy[m]).query_ball_point(xy[m], r=75.0)
    msn_frac[m] = [is_msn[n].mean() if len(n) else 0 for n in nb]
stri = msn_frac >= 0.25
wm = obs.niche_cc.astype(str).str.startswith("C7").values | obs.niche.astype(str).str.startswith("N5").values
comp = np.where(~wm, "other", np.where(stri, "striatal bundles", "corpus callosum / WM"))
grp = (obs.experiment.astype(str) + ":" + obs.condition.astype(str)).values

def pseudobulk(mask_fn, label):
    rows, meta = [], []
    for s in sorted(obs.sample_id[cup].unique()):
        for c in ["corpus callosum / WM", "striatal bundles"]:
            idx = np.where(((obs.sample_id == s).values) & (comp == c) & mask_fn())[0]
            if len(idx) < 40: continue
            X, _ = common.read_rows(H5, idx)
            X = np.expm1(X.toarray()).mean(0)                     # mean normalised counts
            rows.append(np.log1p(X)); meta.append({"sample_id": s, "group": grp[idx[0]], "compartment": c, "n": len(idx)})
    pb = pd.DataFrame(rows, columns=var.index, index=[f"{m['group']} | {m['compartment']} | {m['sample_id']} (n={m['n']})" for m in meta])
    pb.to_csv(f"{OUT}/pseudobulk_{label}.csv")
    return pb, pd.DataFrame(meta)

pb_mg, meta_mg = pseudobulk(lambda: (obs.cell_type.values == "Microglia"), "microglia")
pb_all, meta_all = pseudobulk(lambda: np.ones(len(obs), bool), "allcells")

for pb, lab in [(pb_mg, "microglia"), (pb_all, "allcells")]:
    # use genes that vary across pseudobulks
    v = pb.var(0).sort_values(ascending=False); genes = v.index[:150]
    Z = pb[genes]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    C = np.corrcoef(Z.values)
    order = leaves_list(linkage(Z.values, "average", metric="correlation"))
    C = pd.DataFrame(C, index=pb.index, columns=pb.index).iloc[order, order]
    C.round(3).to_csv(f"{OUT}/pseudobulk_{lab}_correlation.csv")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(C.values, cmap=viz.div_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(C))); ax.set_xticklabels([i.split(" (")[0] for i in C.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(C))); ax.set_yticklabels([i for i in C.index], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r (z-scored log pseudobulk, top-150 variable genes)")
    ax.set_title(f"Pseudobulk similarity of {lab} by lesion compartment and section")
    fig.savefig(f"{OUT}/pseudobulk_{lab}_correlation.png"); plt.close(fig)
    print(f"\n=== {lab}: mean r between compartment groups")
    key = pd.Series([i.rsplit(" | ", 1)[0] for i in C.index], index=C.index)
    M = C.groupby(key).mean().T.groupby(key).mean()
    print(M.round(2).to_string())

# key genes, microglia pseudobulk averaged per group x compartment
genes = [g for g in ["Cx3cr1", "Tmem119", "Siglech", "Trem2", "Cd68", "Lgals3", "Spp1", "Cybb", "Igf1", "Lyz2", "Cd74", "H2-Eb1", "Mki67", "Top2a",
                     "Mbp", "Mog", "Gfap", "Vim", "Il33", "Apoe", "Cd63", "Ctsb", "Lpl", "Itgax", "Clec7a", "Axl", "Csf1", "Cd9"] if g in var.index]
key = pd.Series([i.rsplit(" | ", 1)[0] for i in pb_mg.index], index=pb_mg.index)
tab = np.expm1(pb_mg[genes]).groupby(key).mean()
tab.round(2).to_csv(f"{OUT}/microglia_key_genes_by_compartment.csv")
print("\nmicroglia mean normalised counts (per 100) by group | compartment:\n" + tab.round(2).T.to_string())
