import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
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


def clean(s: str) -> str:
    return (
        s.replace("Param ", "")
        .replace("UserAttribute ", "")
        .replace("dfa_", "")
        .replace("norm_", "")
        .replace("VegBlueDFA", "vegan punishment")
        .replace("EarlyBirdDFA1", "hungry punishment")
        .replace("EarlyBird", "hungry violations")
        .replace("Vegan", "vegan violations")
        .replace("VegetarianBlue", "vegan violations")
        .replace("VegetarianOrange", "vegetarian violations")
    )


def fmt3(x: float) -> str:
    s = f"{x:.3f}"
    return s.rstrip("0").rstrip(".")


def pareto_max_min_min(score, v1, v2, eps=1e-12):
    s = np.asarray(score)
    a = np.asarray(v1)
    b = np.asarray(v2)

    keep = np.ones(len(s), dtype=bool)
    for i in range(len(s)):
        if not keep[i]:
            continue
        dom = (
            (s >= s[i] - eps)
            & (a <= a[i] + eps)
            & (b <= b[i] + eps)
            & ((s > s[i] + eps) | (a < a[i] - eps) | (b < b[i] - eps))
        )
        dom[i] = False
        if dom.any():
            keep[i] = False
    return keep


def voronoi_nn_grid(x, y, z, nx=600, ny=600):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    gx = np.logspace(np.log10(x.min()), np.log10(x.max()), nx)
    gy = np.logspace(np.log10(y.min()), np.log10(y.max()), ny)
    Xg, Yg = np.meshgrid(gx, gy)

    def edges(g):
        mids = np.sqrt(g[:-1] * g[1:])
        e = np.empty(len(g) + 1)
        e[1:-1] = mids
        e[0] = g[0] * (g[0] / mids[0])
        e[-1] = g[-1] * (g[-1] / mids[-1])
        return e

    gx_e, gy_e = edges(gx), edges(gy)

    xt, yt = np.log10(x), np.log10(y)
    qx, qy = np.log10(Xg.ravel()), np.log10(Yg.ravel())
    idx = cKDTree(np.c_[xt, yt]).query(np.c_[qx, qy], k=1)[1]
    return gx_e, gy_e, z[idx].reshape(ny, nx)


def plot_two_norm_weight_sweep(
    csv_path: str,
    *,
    norm_cols,
    score_col="UserAttribute pm_score",
    value_col="Value",
    out_stem=None,  # saves out_stem.{png,pdf} if provided
    bg_cmap="mako",  # Voronoi background
    pareto_cmap="viridis",  # bright (no dark colors): cyan -> magenta
):
    df = pd.read_csv(csv_path)
    if "State" in df.columns:
        df = df[df["State"].eq("COMPLETE")].copy()

    wcols = [c for c in df.columns if c.startswith("Param dfa_")]
    if len(wcols) != 2:
        raise ValueError(f"Expected exactly 2 'Param dfa_*' cols, got: {wcols}")
    w1, w2 = wcols
    n1, n2 = norm_cols

    x = df[w1].to_numpy(float)
    y = df[w2].to_numpy(float)
    score = df[score_col].to_numpy(float)
    v1 = df[n1].to_numpy(float)
    v2 = df[n2].to_numpy(float)
    val = df[value_col].to_numpy(float)

    pareto = pareto_max_min_min(score, v1, v2)
    best = int(np.nanargmax(val))

    panels = [(score_col, "score"), (n1, clean(n1)), (n2, clean(n2))]

    vals = val[pareto] if np.any(pareto) else val
    norm_val = Normalize(np.nanmin(vals), np.nanmax(vals))

    fig = plt.figure(figsize=(17.2, 5.2))
    gs = gridspec.GridSpec(
        1,
        10,
        figure=fig,
        width_ratios=[1, 0.055, 0.08, 1, 0.055, 0.08, 1, 0.055, 0.06, 0.07],
        wspace=0.07,
    )

    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 6])]
    caxes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4]), fig.add_subplot(gs[0, 7])]
    for spacer in (2, 5, 8):
        ax = fig.add_subplot(gs[0, spacer])
        ax.axis("off")
    vax = fig.add_subplot(gs[0, 9])

    value_mappable = None

    for i, (ax, cax, (col, title)) in enumerate(zip(axes, caxes, panels)):
        z = df[col].to_numpy(float)
        gx_e, gy_e, Zg = voronoi_nn_grid(x, y, z)

        if i > 0:  # For left and middle plots
            cmap_used = bg_cmap + "_r"  # Reverse the colormap
        else:  # For the right plot
            cmap_used = bg_cmap  # Use the original colormap

        mesh = ax.pcolormesh(
            gx_e,
            gy_e,
            Zg,
            shading="auto",
            cmap=cmap_used,
            norm=Normalize(np.nanmin(z), np.nanmax(z)),
            alpha=0.85,
        )

        ax.scatter(x, y, s=16, alpha=0.28, edgecolors="none", zorder=2)
        sc = ax.scatter(
            x[pareto],
            y[pareto],
            c=val[pareto],
            cmap=pareto_cmap,
            norm=norm_val,
            s=70,
            alpha=0.98,
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
        )
        value_mappable = value_mappable or sc

        ax.scatter(
            [x[best]], [y[best]], s=260, marker="*", facecolors="none", edgecolors="black", linewidths=1.4, zorder=6
        )

        ax.set_title(title, pad=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_xlabel("")

        if i == 0:
            ax.set_ylabel(clean(w2))
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        cb = fig.colorbar(mesh, cax=cax)
        cb.ax.tick_params(pad=2)

    cbv = fig.colorbar(value_mappable, cax=vax)
    cbv.set_label("Value")
    cbv.ax.tick_params(pad=2)

    fig.supxlabel(clean(w1), y=-0.0)
    # fig.suptitle("Two-norm weight sweep: score + relevant norms (Pareto colored by Value)", y=0.98)
    # fig.subplots_adjust(top=0.86, bottom=0.17)

    # callout: no Value line (consistent with your 1D plots)
    lx, ly = np.log10(x[best]), np.log10(y[best])
    xmid, ymid = np.nanmedian(np.log10(x)), np.nanmedian(np.log10(y))
    dx = -28 if lx > xmid else 28
    dy = 16 if ly < ymid else -24
    ha = "right" if dx < 0 else "left"
    va = "bottom" if dy > 0 else "top"

    axes[0].annotate(
        f"{clean(w1)}={fmt3(x[best])}\n{clean(w2)}={fmt3(y[best])}",
        xy=(x[best], y[best]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.22", alpha=0.9),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.1,
            color="0.25",
            shrinkA=2,
            shrinkB=6,
            mutation_scale=12,
            alpha=0.9,
        ),
        zorder=10,
        annotation_clip=False,
    )

    sns.despine(fig=fig)

    if out_stem:
        fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{out_stem}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    return fig


if __name__ == "__main__":
    plot_two_norm_weight_sweep(
        "solution.csv",
        norm_cols=("UserAttribute norm_EarlyBird", "UserAttribute norm_VegetarianOrange"),
        out_stem="solution",
    )
    plot_two_norm_weight_sweep(
        "contradiction.csv",
        norm_cols=("UserAttribute norm_EarlyBird", "UserAttribute norm_Vegan"),
        out_stem="contradiction",
    )
