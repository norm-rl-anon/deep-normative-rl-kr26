import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

sns.set_theme(
    style="whitegrid",
    context="paper",
    rc={
        "axes.spines.right": False,
        "axes.spines.top": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    },
)

BASES = ["vegan", "vegetarian", "earlybird", "contradiction", "solution", "penalty"]
ALGOS = ["dqn", "ppo"]
CONFIGS = {
    "1mil": "1 million",
    "2.5mil": "2.5 million",
    "5mil": "5 million",
    "10mil": "10 million",
    "20mil": "20 million",
}

rows = []
for base in BASES:
    for p in Path(base).glob("*.csv"):
        algo, cfg = p.stem.split("_", 1)
        if algo in ALGOS and cfg in CONFIGS:
            rows.append(pd.read_csv(p, usecols=["Value"]).assign(base=base, algo=algo, cfg=cfg))

df = pd.concat(rows, ignore_index=True)
df["config"] = df["cfg"].map(CONFIGS)

palette = dict(zip(ALGOS, sns.color_palette("deep", n_colors=len(ALGOS))))

box_kws = dict(
    width=0.72,
    linewidth=1.15,
    fliersize=2.5,
    flierprops=dict(marker="o", markersize=2.5, alpha=0.55),  # only affects outlier dots
    whis=(5, 95),
    boxprops=dict(alpha=0.95),
    medianprops=dict(linewidth=2.0),
    whiskerprops=dict(linewidth=1.15),
    capprops=dict(linewidth=1.15),
)

fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False, constrained_layout=True)
axes = axes.ravel()

for i, base in enumerate(BASES[:6]):
    ax = axes[i]
    d = df[df.base == base]
    if d.empty:
        ax.set_visible(False)
        continue

    order = [CONFIGS[k] for k in CONFIGS if k in set(d["cfg"])]

    sns.boxplot(
        data=d,
        x="config",
        y="Value",
        hue="algo",
        hue_order=ALGOS,
        order=order,
        palette=palette,
        ax=ax,
        **box_kws,
    )

    ax.set_title(base)
    ax.set_xlabel("")
    ax.set_ylabel("undiscounted return" if (i % 3 == 0) else "")

    ax.tick_params(axis="x", rotation=28)
    for t in ax.get_xticklabels():
        t.set_ha("right")

    ax.grid(True, axis="y")
    sns.despine(ax=ax)

    if ax.get_legend():
        ax.get_legend().remove()

# legend as filled patches (matches the boxes)
handles = [Patch(facecolor=palette[a], edgecolor="none", label=a.upper()) for a in ALGOS]
fig.legend(handles=handles, loc="upper right", ncol=len(ALGOS), frameon=False)

fig.savefig("all_bases.png", dpi=300, bbox_inches="tight")
fig.savefig("all_bases.pdf", bbox_inches="tight")
