from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

Poly = Sequence[float]  # t0..td (in increasing powers)
Interval = Tuple[float, float]  # closed [l, r], may include +/-inf


sns.set_theme(
    style="whitegrid",
    context="paper",
    rc={
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "axes.labelsize": 18,  # 11
        "axes.titlesize": 18,  # 12,
        "xtick.labelsize": 15,  # 10,
        "ytick.labelsize": 15,  # 10,
        "legend.fontsize": 15,  # 10,
        "legend.title_fontsize": 15,  # 10,
    },
)


# ----------------------------- Envelope utility -----------------------------


def envelope_regions_geq(
    polys: Iterable[Poly],
    *,
    x_clip: float = 1e6,
    real_tol: float = 1e-10,
    tie_tol: float = 1e-10,
) -> Dict[int, List[Interval]]:
    """
    Return intervals where each polynomial is (weakly) maximal.

    Output: {i: [(l,r), ...]} where poly i satisfies p_i(x) >= p_j(x) for all j.
    Intervals are closed; tie points are included in *all* maximizers at that x.
    Coefficients are in increasing powers: [t0, t1, ..., td].
    """
    P = [np.asarray(p, float) for p in polys]
    m = len(P)
    if m == 0:
        return {}
    if m == 1:
        return {0: [(-math.inf, math.inf)]}

    def pad(a: np.ndarray, n: int) -> np.ndarray:
        return np.pad(a, (0, max(0, n - len(a))))

    def real_roots(diff: np.ndarray) -> List[float]:
        r = np.polynomial.polynomial.polyroots(diff)  # increasing-order coeffs
        r = r[np.isfinite(r)]
        r = r[np.abs(r.imag) <= real_tol * np.maximum(1.0, np.abs(r.real))]
        return r.real.tolist()

    def eval_all(x: float) -> np.ndarray:
        # Horner evaluation, stable and fast
        vals = np.zeros(m)
        for i, c in enumerate(P):
            v = 0.0
            for a in reversed(c):
                v = v * x + a
            vals[i] = v
        return vals

    def pick_point(l: float, r: float) -> float:
        # representative point in open interval (l,r)
        if math.isinf(l) and math.isinf(r):
            return 0.0
        if math.isinf(l):
            return -x_clip
        if math.isinf(r):
            return x_clip
        return 0.5 * (l + r)

    # 1) real pairwise intersections -> breakpoints
    bps: List[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            n = max(len(P[i]), len(P[j]))
            diff = pad(P[i], n) - pad(P[j], n)
            if np.all(np.abs(diff) <= 1e-14):
                continue
            bps += real_roots(diff)

    bps.sort()

    # 2) deduplicate breakpoints
    cuts: List[float] = []
    for x in bps:
        if not cuts or abs(x - cuts[-1]) > real_tol * max(1.0, abs(x), abs(cuts[-1])):
            cuts.append(x)

    # 3) winners on open intervals
    out: Dict[int, List[Interval]] = {i: [] for i in range(m)}
    all_cuts = [-math.inf] + cuts + [math.inf]

    for l, r in zip(all_cuts[:-1], all_cuts[1:]):
        x0 = pick_point(l, r)
        vals = eval_all(x0)
        mx = vals.max()
        winners = np.flatnonzero(np.abs(vals - mx) <= tie_tol * max(1.0, abs(mx)))
        for w in map(int, winners):
            out[w].append((l, r))

    # 4) include each breakpoint itself for all maximizers at that x
    for x in cuts:
        vals = eval_all(x)
        mx = vals.max()
        winners = np.flatnonzero(np.abs(vals - mx) <= tie_tol * max(1.0, abs(mx)))
        for w in map(int, winners):
            out[w].append((x, x))

    # 5) merge overlapping / touching closed intervals
    def merge(intervals: List[Interval]) -> List[Interval]:
        if not intervals:
            return []
        intervals.sort(key=lambda t: (t[0], t[1]))
        merged = [intervals[0]]
        for a, b in intervals[1:]:
            la, lb = merged[-1]
            if a <= lb or math.isclose(a, lb, rel_tol=0.0, abs_tol=1e-12):
                merged[-1] = (la, max(lb, b))
            else:
                merged.append((a, b))
        return merged

    return {i: merge(segs) for i, segs in out.items() if segs}


# ----------------------------- Plotting utilities -----------------------------


_bprefix = re.compile(r"^\s*b[⁹⁸⁷⁶⁵⁴³²¹⁰0-9]+\s*:")
_sup_map = str.maketrans("⁹⁸⁷⁶⁵⁴³²¹⁰", "0123456789")


def _metric_rank(name: str, score_metric: str) -> tuple:
    """
    Sort order:
      0) score first
      1) then b¹:, b²:, ... (ascending)
      2) then everything else alphabetically
    """
    if name == score_metric:
        return (0, 0, name)

    s = name.strip()
    # try parse "b¹:" or "b2:" style
    if s.lower().startswith("b") and ":" in s:
        head = s.split(":", 1)[0]  # "b¹" or "b2"
        digits = head[1:].translate(_sup_map)
        if digits.isdigit():
            return (1, int(digits), s)

    return (2, 10**9, s)


def _draw_piecewise_constant(ax, df, *, x, y, color, lw=2.2) -> None:
    """
    Draw only horizontal segments between consecutive x points using y at left endpoint.
    Assumes df is sorted by x.
    """
    xs = df[x].to_numpy()
    ys = df[y].to_numpy()
    if len(xs) < 2:
        return
    for i in range(len(xs) - 1):
        ax.hlines(ys[i], xs[i], xs[i + 1], color=color, linewidth=lw)


def plot_score_and_violations(
    plot_df_long: pd.DataFrame,
    *,
    x: str = "b",
    metric_col: str = "metric",
    value_col: str = "value",
    score_metric: str = "b⁰: score",
    figsize: Tuple[float, float] = (10, 6.8),
    break_line_style: Mapping = None,
    label_breaks: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes], List[float]]:
    """
    Two stacked plots:
      - Top: score (linear y)
      - Bottom: violations (log y, inverted so 'lower is better' plots higher)
    Both share log x.

    Also draws faint vertical lines at breakpoints (where any metric changes),
    and labels those b's above the top panel.
    """
    break_line_style = break_line_style or dict(color="0.55", lw=1.0, alpha=0.45)

    df = plot_df_long.sort_values([metric_col, x]).copy()

    # stable, intended order (score, then b¹/b²..., then others)
    metrics = sorted(df[metric_col].unique(), key=lambda s: _metric_rank(s, score_metric))

    palette = dict(zip(metrics, sns.color_palette("deep", n_colors=len(metrics))))

    score_df = df[df[metric_col] == score_metric]
    viol_df = df[df[metric_col] != score_metric]
    viol_metrics = [m for m in metrics if m != score_metric and m in set(viol_df[metric_col].unique())]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,  # <-- add this
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.08},
    )

    # ---- top: score ----
    if not score_df.empty:
        _draw_piecewise_constant(
            ax1,
            score_df.sort_values(x),
            x=x,
            y=value_col,
            color=palette[score_metric],
            lw=2.6,
        )
    ax1.set_ylabel("score")
    ax1.margins(x=0.02)

    # ---- bottom: violations ----
    for m in viol_metrics:
        sub = viol_df[viol_df[metric_col] == m].sort_values(x)
        _draw_piecewise_constant(
            ax2,
            sub,
            x=x,
            y=value_col,
            color=palette[m],
            lw=2.2,
        )

    ax2.set_ylabel("violations")
    ax2.set_yscale("linear")  # or log
    ax2.invert_yaxis()
    ax2.set_xlabel(x)

    # shared log-x
    ax1.set_xscale("log")
    ax2.set_xscale("log")

    # ---- compute breakpoints: x where any metric changes ----
    change_bs: set[float] = set()
    for m in metrics:
        sub = df[df[metric_col] == m].sort_values(x)
        xs = sub[x].to_numpy()
        ys = sub[value_col].to_numpy()
        if len(xs) < 2:
            continue
        idx = np.where(ys[1:] != ys[:-1])[0] + 1
        change_bs.update(xs[idx])

    change_bs = sorted(change_bs)

    # ---- draw break lines + top labels (staggered, horizontal) ----
    for i, b0 in enumerate(change_bs):
        ax1.axvline(b0, **break_line_style)
        ax2.axvline(b0, **break_line_style)

        if label_breaks:
            # stagger slightly to reduce overlap
            # y_text = 1.03 if (i % 2 == 0) else 1.075
            # final version: don't stagger (unnecessary for solution norm base)
            y_text = 1.01
            ax1.text(
                b0,
                y_text,
                f"{b0:.3g}",
                transform=ax1.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=11,
                color="0.35",
                clip_on=False,
            )

    ax1.legend(
        handles=[Line2D([0], [0], color=palette[score_metric], lw=2.6, label=score_metric)],
        title=None,
        loc="right",
        frameon=True,
        borderaxespad=0.3,
    )

    ax2.legend(
        handles=[Line2D([0], [0], color=palette[m], lw=2.2, label=m) for m in viol_metrics],
        title=None,
        loc="right",
        frameon=True,
        borderaxespad=0.3,
    )

    sns.despine(fig=fig)

    return fig, (ax1, ax2), change_bs


# ----------------------------- Your loop (tiny changes) -----------------------------

experiments = [
    {
        "name": "vegan",
        "files": ["vegan.csv"],
        "poly": lambda x: (
            float(x["UserAttribute pm_score"] / 1795.0),
            float((4.0 - x["UserAttribute norm_Vegan"]) / 4.0),
        ),
        "plot": {
            "b⁰: score": "UserAttribute pm_score",
            "b¹: vegan violations": "UserAttribute norm_Vegan",
        },
    },
    {
        "name": "vegetarian",
        "files": ["vegetarian.csv"],
        "poly": lambda x: (
            float(x["UserAttribute pm_score"] / 1795.0),
            float((2.0 - x["UserAttribute norm_VegetarianOrange"]) / 2.0),
        ),
        "plot": {
            "b⁰: score": "UserAttribute pm_score",
            "b¹: vegetarian violations": "UserAttribute norm_VegetarianOrange",
        },
    },
    {
        "name": "earlybird",
        "files": ["earlybird.csv"],
        "poly": lambda x: (
            float(x["UserAttribute pm_score"] / 1795.0),
            float(1.0 - x["UserAttribute norm_EarlyBird"]),
        ),
        "plot": {
            "b⁰: score": "UserAttribute pm_score",
            "b¹: hungry violations": "UserAttribute norm_EarlyBird",
        },
    },
    {
        "name": "contradiction",
        "files": [
            "contradiction.csv",
        ],
        "poly": lambda x: (
            float(x["UserAttribute pm_score"] / 1795.0),
            float((4.0 - x["UserAttribute norm_Vegan"]) / 4.0),
            float(1.0 - x["UserAttribute norm_EarlyBird"]),
        ),
        "plot": {
            "b⁰: score": "UserAttribute pm_score",
            "b¹: vegan violations": "UserAttribute norm_Vegan",
            "b²: hungry violations": "UserAttribute norm_EarlyBird",
        },
    },
    {
        "name": "solution",
        "files": [
            "solution.csv",
        ],
        "poly": lambda x: (
            float(x["UserAttribute pm_score"] / 1795.0),
            float((2.0 - x["UserAttribute norm_VegetarianOrange"]) / 2.0),
            float(1.0 - x["UserAttribute norm_EarlyBird"]),
        ),
        "plot": {
            "b⁰: score": "UserAttribute pm_score",
            "b¹: vegetarian violations": "UserAttribute norm_VegetarianOrange",
            "b²: hungry violations": "UserAttribute norm_EarlyBird",
        },
    },
    {
        "name": "penalty",
        "files": ["penalty.csv"],
        "poly": lambda x: (
            float(x["UserAttribute pm_score"] / 1795.0),
            float((4.0 - x["UserAttribute norm_Vegan"]) / 4.0),
            float((4.0 - x["UserAttribute norm_CTD"]) / 4.0),
            float(1.0 - x["UserAttribute norm_EarlyBird"]),
        ),
        "plot": {
            "b⁰: score": "UserAttribute pm_score",
            "b¹: vegan violations": "UserAttribute norm_Vegan",
            "b²: atonement violations": "UserAttribute norm_CTD",
            "b³: hungry violations": "UserAttribute norm_EarlyBird",
        },
    },
]

for experiment in experiments:
    print(f"{experiment['name']}\n===")

    df = pd.concat([pd.read_csv(f).dropna() for f in experiment["files"]], ignore_index=True)
    polynomials = [experiment["poly"](row) for _, row in df.iterrows()]
    regions = envelope_regions_geq(polynomials)

    # Build piecewise-constant plot points at interval endpoints.
    plot_points: List[dict] = []
    for i, poly_regions in regions.items():
        good_regions = [(max(1.0, l), r) for (l, r) in poly_regions if r >= 1]
        for l, r in good_regions:
            print(f"[{l},{r}]", i)
            info = {k: df[v].iloc[i] for k, v in experiment["plot"].items()}
            plot_points.append({"b": l, **info})
            plot_points.append({"b": r - 1e-12, **info})  # open-right visual convention

    plot_points.sort(key=lambda d: d["b"])
    # extend last point to a "nice" decade for the log axis
    if len(plot_points) >= 2:
        plot_points[-1]["b"] = 2 * 10 ** math.ceil(math.log10(plot_points[-2]["b"]))

    plot_df = pd.DataFrame(plot_points).melt(
        id_vars="b",
        value_vars=list(experiment["plot"].keys()),
        var_name="metric",
        value_name="value",
    )

    fig, _, _ = plot_score_and_violations(plot_df, score_metric="b⁰: score")
    fig.savefig(f"{experiment['name']}.pdf", bbox_inches="tight")
    fig.savefig(f"{experiment['name']}.png", bbox_inches="tight")
