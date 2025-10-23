import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_ROLES = ("filler", "guilty_suspect", "innocent_suspect")

def _extract_role_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get role-distance pairs from either long or wide DataFrame.
    Support three formats:
      a. long table with 'role' and 'distance' columns
      b. wide table with 'min_by_role[role]' columns (default output of run())
      c. wide table with member columns named like 'Perp[guilty_suspect]' from which to extract all distances
    Priority: long table > min_by_role > member columns
    """
    # a. long table
    if {"role", "distance"}.issubset(df.columns):
        out = df[["role", "distance"]].dropna()
        return out

    # b. wide table with min_by_role[role] columns
    min_cols = [c for c in df.columns if c.startswith("min_by_role[") and c.endswith("]")]
    if min_cols:
        m = df[min_cols].copy()
        # extract role names
        m.columns = [re.findall(r"min_by_role\[(.+?)\]", c)[0] for c in m.columns]
        long = m.melt(var_name="role", value_name="distance").dropna()
        return long[["role", "distance"]]

    # c. wide table with member columns named like 'Perp[guilty_suspect]'
    member_cols = []
    for c in df.columns:
        if "[" in c and "]" in c and not c.startswith("min_by_role["):
            member_cols.append(c)
    if member_cols:
        sub = df[member_cols].copy()
        rows = []
        for col in member_cols:
            m = re.findall(r"\[(.+?)\]", col)
            role = m[-1] if m else "unknown"
            series = sub[col].dropna()
            if not series.empty:
                rows.append(
                    pd.DataFrame({"role": role, "distance": series.values})
                )
        if rows:
            return pd.concat(rows, ignore_index=True)

    raise ValueError("DataFrame does not contain recognizable role-distance columns.")

def plot_role_histograms(df: pd.DataFrame,
                         roles=_ROLES,
                         bins=40,
                         xlim=None,
                         alpha=0.75,
                         edgecolor="none",
                         figsize=(7,4),
                         title="Role-wise similarity histograms",
                         show_legend=True,
                         colors=None,
                         ax=None,
                         show_grid=True,
                         grid_kw=None,
                         frameon=True) -> plt.Axes:
    """
    Plot histograms of distances by fillers / guilty_suspect / innocent_suspect (just counts with no standardization).
    - Use min_by_role[role] columns if present (output of run()).
    - Else if member columns named like 'Perp[guilty_suspect]' exist, extract role from column names and use all distances.
    - Else if long table with 'role' and 'distance' columns exist, use them directly.
    - Raise error if none of the above formats are found.
    """
    data = _extract_role_distance(df)

    # Only keep specified roles
    data = data[data["role"].isin(roles)]
    if xlim is None and not data["distance"].empty:
        lo, hi = float(data["distance"].min()), float(data["distance"].max())
    else:
        lo, hi = (xlim if xlim is not None else (0.0, 1.0))

    if isinstance(bins, int):
        bins = np.linspace(lo, hi, bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    def _color_for_role(role: str, idx: int):
        if colors is None:
            return None
        if isinstance(colors, dict):
            return colors.get(role, None)
        if isinstance(colors, (list, tuple)):
            if 0 <= idx < len(colors):
                return colors[idx]
            return None
        if isinstance(colors, str):
            return colors
        return None

    plotted_any = False
    for i, r in enumerate(roles):
        vals = data.loc[data["role"] == r, "distance"].to_numpy()
        if len(vals) == 0:
            continue
        c = _color_for_role(r, i)
        hist_kwargs = dict(
            bins=bins,
            alpha=alpha,
            edgecolor=edgecolor,
            label=r.replace("_", " ")
        )
        if c is not None:
            hist_kwargs["color"] = c
        ax.hist(vals, **hist_kwargs)
        plotted_any = True

    ax.set_xlim(lo, hi)
    ax.set_xlabel("Euclidean distance (lower = more similar)")
    ax.set_ylabel("count")
    if title:
        ax.set_title(title)
    if show_legend and plotted_any:
        ax.legend(frameon=frameon)
    ax.grid(True, ls=":", alpha=0.4)
    if show_grid:
        kw = {"ls": ":", "alpha": 0.4}
        if isinstance(grid_kw, dict):
            kw.update(grid_kw)
        ax.grid(True, **kw)
    else:
        ax.grid(False)

    return ax