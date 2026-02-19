import numpy as np
import pandas as pd


def pareto_max_min(df: pd.DataFrame, *, maximize: list[str], minimize: list[str], eps: float = 1e-12) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows (maximize some cols, minimize others)."""
    M = df[maximize].to_numpy(float)
    m = df[minimize].to_numpy(float)

    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue

        ge_all = (M >= M[i] - eps).all(axis=1)
        le_all = (m <= m[i] + eps).all(axis=1)
        strictly = (M > M[i] + eps).any(axis=1) | (m < m[i] - eps).any(axis=1)

        dominated = ge_all & le_all & strictly
        dominated[i] = False
        if dominated.any():
            keep[i] = False

    return keep


def fmt(x, ndp: int) -> str:
    return f"{float(x):.{ndp}f}"


def make_latex_table(df: pd.DataFrame) -> str:
    # data columns
    col_rank = "Rank"
    col_score = "UserAttribute pm_score"

    viol_cols = [
        ("UserAttribute norm_EarlyBird", "Hungry"),
        ("UserAttribute norm_CTD", "Penalty"),
        ("UserAttribute norm_Vegan", "Vegan"),
    ]
    w_cols = [
        ("Param dfa_EarlyBirdDFA1", "Hungry"),
        ("Param dfa_CTDBlueDFA", "Penalty"),
        ("Param dfa_VegBlueDFA", "Vegan"),
    ]

    # header (booktabs + cmidrule)
    group_header = r" &  & \multicolumn{3}{c}{Violation count} & \multicolumn{3}{c}{Punishment weight} \\"
    cmidrules = r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}"
    header = ["\\#", "Score"] + [h for _, h in viol_cols] + [h for _, h in w_cols]

    lines = [
        r"\begin{tabular}{r r r r r r r r}",
        r"\toprule",
        group_header,
        cmidrules,
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        out = [str(int(row[col_rank]))]
        out.append(fmt(row[col_score], 2))  # score: 2 decimals
        out += [fmt(row[c], 3) for c, _ in viol_cols]  # violations: 3 decimals
        out += [fmt(row[c], 2) for c, _ in w_cols]  # weights: 2 decimals
        lines.append(" & ".join(out) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(csv_path: str, k: int = 5) -> None:
    df = pd.read_csv(csv_path)
    if "State" in df.columns:
        df = df[df["State"].eq("COMPLETE")].copy()

    pareto = pareto_max_min(
        df,
        maximize=["UserAttribute pm_score"],
        minimize=[
            "UserAttribute norm_EarlyBird",  # hungry
            "UserAttribute norm_CTD",  # penalty
            "UserAttribute norm_Vegan",  # vegan
        ],
    )

    top = (
        df.loc[pareto]
        .sort_values("Value", ascending=False)  # Value only used to rank Pareto points
        .head(k)
        .copy()
        .reset_index(drop=True)
    )
    top.insert(0, "Rank", np.arange(1, len(top) + 1))

    print(make_latex_table(top))


if __name__ == "__main__":
    main("penalty.csv", k=3)
