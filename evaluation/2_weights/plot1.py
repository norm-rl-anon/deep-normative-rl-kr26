import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

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


def pareto_front(violations, score, atol=1e-12):
    v = np.asarray(violations)
    s = np.asarray(score)

    order = np.argsort(v, kind="mergesort")  # stable
    on_front = np.zeros(len(v), dtype=bool)

    best = -np.inf
    for i in order:
        if s[i] > best + atol:
            on_front[i] = True
            best = s[i]
    return on_front


def detect_columns(df):
    wcol = next(c for c in df.columns if c.startswith("Param dfa_"))
    vcol = next(c for c in df.columns if c.startswith("UserAttribute norm_"))
    scol = "UserAttribute pm_score"
    valcol = "UserAttribute Value" if "UserAttribute Value" in df.columns else "Value"
    if scol not in df.columns or valcol not in df.columns:
        raise KeyError("Missing expected score/value columns.")
    return wcol, vcol, scol, valcol


def load_run(csv_path):
    df = pd.read_csv(csv_path)
    wcol, vcol, scol, valcol = detect_columns(df)

    w = df[wcol].to_numpy()
    viol = df[vcol].to_numpy()
    score = df[scol].to_numpy()
    value = df[valcol].to_numpy()

    pareto = pareto_front(viol, score)
    best = int(np.nanargmax(value))

    return dict(w=w, viol=viol, score=score, value=value, pareto=pareto, best=best, wcol=wcol, valcol=valcol)


def clean_label(weight_col: str) -> str:
    return (
        weight_col.replace("Param dfa_", "")
        .replace("VegBlueDFA", "vegan punishment")
        .replace("EarlyBirdDFA1", "hungry punishment")
    )


def scatter_layer(ax, x, y, *, s, alpha, **kw):
    return ax.scatter(x, y, s=s, alpha=alpha, edgecolors="none", **kw)


def plot_normbase(fig, gs, col, title, run, norm, *, cmap="viridis", left_labels=False, xlabel=None):
    w, viol, score, value = run["w"], run["viol"], run["score"], run["value"]
    pareto, best = run["pareto"], run["best"]

    ax_s = fig.add_subplot(gs[0, col])
    ax_v = fig.add_subplot(gs[1, col], sharex=ax_s)

    # background
    scatter_layer(ax_s, w, score, s=14, alpha=0.35)
    scatter_layer(ax_v, w, viol, s=14, alpha=0.35)

    # pareto highlights (shared colormap scale)
    pareto_kw = dict(c=value[pareto], cmap=cmap, norm=norm, s=55, alpha=0.98, edgecolors="black", linewidths=0.6)
    sc = ax_s.scatter(w[pareto], score[pareto], **pareto_kw)
    ax_v.scatter(w[pareto], viol[pareto], **pareto_kw)

    # best marker
    star_kw = dict(s=190, marker="*", facecolors="none", edgecolors="black", linewidths=1.2, zorder=6)
    ax_s.scatter([w[best]], [score[best]], **star_kw)
    ax_v.scatter([w[best]], [viol[best]], **star_kw)

    ax_s.set_title(title)
    ax_s.set_ylabel("score" if left_labels else "")
    ax_v.set_ylabel("violation count" if left_labels else "")

    plt.setp(ax_s.get_xticklabels(), visible=False)

    ax_v.set_xscale("log")
    ax_v.invert_yaxis()
    ax_v.set_xlabel(xlabel if xlabel is not None else clean_label(run["wcol"]))

    for ax in (ax_s, ax_v):
        ax.grid(True, which="both", alpha=0.25)

    # callout (close + clear arrow)
    ax_s.annotate(
        f"w={w[best]:.3f}",
        xy=(w[best], score[best]),
        xytext=(-20, 18),
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", alpha=0.9),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.2,
            color="0.25",
            shrinkA=2,
            shrinkB=6,
            mutation_scale=12,
            alpha=0.95,
        ),
        zorder=20,
        annotation_clip=False,
    )

    return sc


def main(out_stem="bases_one_weight", *, cmap="viridis"):
    names = ["Vegan", "Vegetarian", "Hungry"]
    runs = [load_run(f"{n}.csv") for n in names]

    vals = np.concatenate([r["value"][r["pareto"]] for r in runs if np.any(r["pareto"])])
    if vals.size == 0:
        vals = np.concatenate([r["value"] for r in runs])
    norm = plt.Normalize(np.nanmin(vals), np.nanmax(vals))

    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    gs = GridSpec(2, 7, figure=fig, width_ratios=[1, 0.10, 1, 0.10, 1, 0.08, 0.045], wspace=0.03, hspace=0.02)

    sc = None
    for i, (name, run) in enumerate(zip(names, runs)):
        sc = plot_normbase(
            fig,
            gs,
            col=2 * i,
            title=name,
            run=run,
            norm=norm,
            cmap=cmap,
            left_labels=(i == 0),
            xlabel=("vegetarian punishment" if i == 1 else None),
        )

    cax = fig.add_subplot(gs[:, 6])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(runs[0]["valcol"])
    cb.ax.tick_params(pad=2)

    sns.despine(fig=fig)
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
