import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_bins(val, bin_points, outlier_val=50, dropna=True):
    """
    Compute bin edges by cumulative probability, after removing outliers.

    Parameters
    ----------
    val: values (confidence value in general)
    bin_points: int
        Number of bins to create. The data will be divided into (bin_points-1) bins.
    outlier_val: default 50
        Values greater than outlier_val are considered outliers and removed before binning.
    dropna: bool, default True
        If True, drop NaNs before computation.
    """
    s = pd.Series(val).astype("float")
    if dropna:
        s = s.dropna()

    # Remove outliers
    s = s[s < outlier_val]
    s= s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        raise ValueError("No valid data points available after removing outliers and NaNs.")

    # Ensure bin_points is at least 2
    bin_points = int(bin_points)
    if bin_points < 2:
        raise ValueError("bin_points must be at least 2 to create bins.")

    pobs = np.linspace(0, 1, bin_points)
    edges = np.quantile(s.values, pobs, method="linear")
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = np.nextafter(edges[i-1], np.inf)

    counts, _ = np.histogram(s.values, bins=edges)
    # return edges, counts, s
    return edges

def plot_hist_with_edges(data: np.ndarray, edges: np.ndarray, ax=None, **hist_kwargs):
    ax = ax or plt.gca()
    ax.hist(data, bins=edges, edgecolor="black", **hist_kwargs)
    ax.set_xlabel("confidence (binned)")
    ax.set_ylabel("count")
    return ax












