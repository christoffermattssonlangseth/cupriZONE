"""Shared plotting helpers: palette (validated default categorical palette, fixed slot order),
chart chrome, and a categorical colour map builder for many-level biological labels."""
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CAT8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# For >8 biological categories (cell types), extend with darker/lighter steps of the same hues.
CAT_EXT = CAT8 + ["#104281", "#9a3a12", "#0d6b49", "#8a5c00", "#a3345e", "#0ca30c", "#9085e9", "#8f1d1c",
                  "#86b6ef", "#f4a98a", "#8fdcc0", "#f7d27a", "#f3bdd1", "#7dd07d", "#c7c1f4", "#f2a3a2"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#8f1d1c"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": False, "grid.color": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


def cmap_for(categories):
    cats = list(categories)
    cols = (CAT_EXT * (len(cats) // len(CAT_EXT) + 1))[: len(cats)]
    return dict(zip(cats, cols))


def seq_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ)


def div_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("div", DIV)


def legend_outside(ax, cmap, title=None, ms=6, ncol=1):
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=ms, label=k) for k, c in cmap.items()]
    ax.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=(1.01, 1), ncol=ncol,
              fontsize=7, title_fontsize=8, markerscale=1, handletextpad=0.4, labelspacing=0.3)
