"""Plotting utilities for scdia.

All plotting functions in this module follow a consistent contract that
prioritises post-publication customisation:

- Accept ``ax`` (or ``axes`` for multi-panel plots). If ``None``, the function
  creates its own ``Figure``; if provided, the function draws onto the supplied
  axes so callers can compose panels into larger figures.
- Never call ``plt.show()`` or ``fig.savefig()``; display and export are
  decided by the caller.
- Never call ``tight_layout`` or use ``constrained_layout`` by default; layout
  decisions belong to the caller.
- Return the ``Axes`` (or ``dict[str, Axes]`` for multi-panel plots). Pass
  ``return_data=True`` to additionally receive the processed data used to
  draw the plot, which is useful for source-data exports or replotting.
"""

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from anndata import AnnData
from scipy.sparse import issparse


def _get_matrix(adata: AnnData, layer: Optional[str] = None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' was not found in adata.layers.")
    return adata.layers[layer]


def _evenly_spaced_indices(indices, max_items):
    indices = np.asarray(indices)
    if max_items is None or indices.size <= max_items:
        return indices
    if max_items <= 0:
        raise ValueError("max_items must be positive or None.")
    positions = np.linspace(0, indices.size - 1, int(max_items))
    positions = np.unique(np.round(positions).astype(int))
    return indices[positions]


def plot_data_completeness(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    color: str = "tab:blue",
    figsize: Tuple[float, float] = (4, 3),
    ax: Optional[plt.Axes] = None,
    return_data: bool = False,
):
    """Plot data completeness across cells as a binned bar chart.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Optional layer name. If ``None``, ``adata.X`` is used.
    color
        Bar color.
    figsize
        Figure size used only when ``ax`` is ``None``.
    ax
        Optional pre-created axes to draw onto. If ``None``, a new figure is
        created with ``figsize``.
    return_data
        If ``True``, also return a ``DataFrame`` with the binned counts.

    Returns
    -------
    matplotlib.axes.Axes
        The axes onto which the bars were drawn.
    """
    X = _get_matrix(adata, layer)
    n_cells, n_proteins = X.shape

    if issparse(X):
        detections_per_protein = np.asarray(X.getnnz(axis=0)).ravel()
    else:
        detections_per_protein = np.count_nonzero(np.asarray(X), axis=0)

    frac_missing = (1 - (detections_per_protein / n_cells)) * 100
    thresholds = [0, 20, 40, 60, 80, 99]
    tick_labels = [100, 80, 60, 40, 20, 1]
    res_values = [int(np.sum(frac_missing <= t)) for t in thresholds]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.bar(thresholds, res_values, width=5, color=color)
    ax.axhline(n_proteins, color="black", linestyle="--")
    ax.set_xticks(thresholds)
    ax.set_xticklabels(tick_labels)
    ax.text(
        12,
        n_proteins * 0.97,
        str(n_proteins),
        horizontalalignment="center",
        verticalalignment="top",
    )
    ax.set_xlabel("Percent data completeness")
    ax.set_ylabel("Number of proteins")
    ax.set_title(f"Data completeness across {n_cells} cells")

    if return_data:
        data = pd.DataFrame(
            {
                "completeness_threshold": tick_labels,
                "missing_max_pct": thresholds,
                "n_proteins": res_values,
            }
        )
        return ax, data
    return ax


def plot_saturation_curve(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    color: str = "tab:blue",
    n_shuffles: int = 20,
    random_state: Optional[int] = None,
    figsize: Tuple[float, float] = (4, 3),
    ax: Optional[plt.Axes] = None,
    return_data: bool = False,
):
    """Plot a cumulative protein saturation curve over an increasing number of cells.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Optional layer name. If ``None``, ``adata.X`` is used.
    color
        Line and band color.
    n_shuffles
        Number of random permutations used to compute the mean curve.
    random_state
        Optional seed for the shuffles. ``None`` keeps the global random state.
    figsize
        Figure size used only when ``ax`` is ``None``.
    ax
        Optional pre-created axes to draw onto.
    return_data
        If ``True``, also return a ``DataFrame`` with the per-step mean and
        standard deviation.

    Returns
    -------
    matplotlib.axes.Axes
    """
    X = _get_matrix(adata, layer)

    if issparse(X):
        X_bin = (X > 0).toarray().astype(bool)
    else:
        X_bin = (np.asarray(X) > 0).astype(bool)

    n_cells, n_proteins = X_bin.shape
    rng = np.random.default_rng(random_state)

    saturation_curves = []
    for _ in range(n_shuffles):
        indices = rng.permutation(n_cells)
        X_shuffled = X_bin[indices, :]
        cum_detected_mask = np.maximum.accumulate(X_shuffled, axis=0)
        saturation_curves.append(cum_detected_mask.sum(axis=1))

    saturation_curves = np.array(saturation_curves)
    mean_curve = saturation_curves.mean(axis=0)
    std_curve = saturation_curves.std(axis=0)
    x_axis = np.arange(1, n_cells + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(x_axis, mean_curve, color=color, label="Mean")
    ax.fill_between(
        x_axis,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color=color,
        alpha=0.2,
        label="Std. Dev.",
    )
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Cumulative unique proteins")
    ax.set_title("Protein saturation curve")

    final_count = int(mean_curve[-1])
    ax.text(
        n_cells * 0.9,
        final_count * 0.7,
        f"Number of cells: {n_cells}",
        ha="right",
        va="bottom",
    )
    ax.text(
        n_cells * 0.9,
        final_count * 0.6,
        f"Number of proteins: {n_proteins}",
        ha="right",
        va="bottom",
    )

    if return_data:
        data = pd.DataFrame({"n_cells": x_axis, "mean": mean_curve, "std": std_curve})
        return ax, data
    return ax


def plot_detection_heatmap_by_batch(
    adata: AnnData,
    batch_col: str,
    *,
    layer: Optional[str] = None,
    zero_as_missing: bool = True,
    max_cells: int = 2000,
    max_features: int = 4000,
    sort_features_by_detection: bool = True,
    cluster_features: bool = False,
    cluster_metric: str = "jaccard",
    cluster_method: str = "average",
    batch_label: Optional[str] = "auto",
    batch_palette=None,
    figsize: Tuple[float, float] = (10, 6),
    axes: Optional[dict] = None,
    return_data: bool = False,
):
    """Plot a feature-by-cell detection heatmap grouped by batch.

    Detected values are shown as black pixels and missing values as white pixels.
    To keep plotting responsive, cells and features are evenly downsampled when
    the matrix exceeds ``max_cells`` or ``max_features``.

    Parameters
    ----------
    axes
        Optional ``dict`` with keys ``"batch"`` and ``"heatmap"`` providing
        pre-created axes. If ``None``, a new ``Figure`` with the appropriate
        2-row gridspec is created using ``figsize``.
    batch_label
        Label shown next to the top batch color strip. Use ``None`` to hide it,
        ``"auto"`` (default) to use ``batch_col``, or a custom string.
    return_data
        If ``True``, also return a ``dict`` containing the binary feature-by-cell
        matrix being plotted (``"detected"``) along with the row/column index
        arrays used for selection.

    Returns
    -------
    dict[str, matplotlib.axes.Axes]
        Mapping with keys ``"batch"`` and ``"heatmap"``.
    """
    from matplotlib.colors import ListedColormap

    if batch_col not in adata.obs:
        raise ValueError(f"Column '{batch_col}' was not found in adata.obs.")

    X = _get_matrix(adata, layer)
    batch_values = (
        adata.obs[batch_col]
        .astype("object")
        .where(adata.obs[batch_col].notna(), "NA")
        .astype(str)
    )
    batch_order = pd.Index(batch_values.drop_duplicates())

    selected_cells = []
    selected_batch_labels = []
    for batch in batch_order:
        batch_indices = np.flatnonzero(batch_values.to_numpy() == batch)
        if max_cells is None:
            batch_max = None
        else:
            batch_max = max(1, int(round(max_cells * batch_indices.size / adata.n_obs)))
        picked = _evenly_spaced_indices(batch_indices, batch_max)
        selected_cells.extend(picked.tolist())
        selected_batch_labels.extend([batch] * len(picked))

    selected_cells = np.asarray(selected_cells, dtype=int)
    if max_cells is not None and selected_cells.size > max_cells:
        selected_cells = _evenly_spaced_indices(selected_cells, max_cells).astype(int)
        selected_batch_labels = batch_values.iloc[selected_cells].tolist()

    if issparse(X):
        feature_detected_counts = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        X_arr = np.asarray(X)
        finite_mask = np.isfinite(X_arr)
        if zero_as_missing:
            finite_mask &= X_arr != 0
        feature_detected_counts = finite_mask.sum(axis=0)

    feature_order = np.arange(adata.n_vars)
    if sort_features_by_detection:
        feature_order = feature_order[np.argsort(feature_detected_counts)[::-1]]
    selected_features = _evenly_spaced_indices(feature_order, max_features).astype(int)

    X_view = X[selected_cells, :][:, selected_features]
    if issparse(X_view):
        if zero_as_missing:
            detected = (X_view != 0).toarray()
        else:
            detected = np.isfinite(X_view.toarray())
    else:
        X_view = np.asarray(X_view)
        detected = np.isfinite(X_view)
        if zero_as_missing:
            detected &= X_view != 0
    detected = detected.T.astype(np.uint8, copy=False)

    if cluster_features and detected.shape[0] > 1:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        if detected.shape[0] > 2000:
            warnings.warn(
                "Feature clustering is being applied to more than 2000 rows. "
                "Consider lowering max_features if plotting becomes slow.",
                RuntimeWarning,
                stacklevel=2,
            )

        distances = pdist(detected.astype(bool), metric=cluster_metric)
        finite_distances = distances[np.isfinite(distances)]
        if finite_distances.size == 0 or np.all(finite_distances == 0):
            warnings.warn(
                "Feature clustering was skipped because pairwise distances are all zero or invalid.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            row_order = leaves_list(linkage(distances, method=cluster_method))
            detected = detected[row_order, :]
            selected_features = selected_features[row_order]

    if (selected_cells.size < adata.n_obs) or (selected_features.size < adata.n_vars):
        warnings.warn(
            "Detection heatmap was downsampled for plotting: "
            f"{selected_features.size}/{adata.n_vars} features and "
            f"{selected_cells.size}/{adata.n_obs} cells shown.",
            RuntimeWarning,
            stacklevel=2,
        )

    batch_labels_series = pd.Series(selected_batch_labels, dtype="object")
    batch_categories = pd.Index(batch_labels_series.drop_duplicates())
    batch_codes = batch_labels_series.map(
        {batch: i for i, batch in enumerate(batch_categories)}
    ).to_numpy()
    if batch_palette is None:
        batch_colors = plt.cm.tab20(np.linspace(0, 1, max(len(batch_categories), 1)))
    elif isinstance(batch_palette, dict):
        missing_batches = [batch for batch in batch_categories if batch not in batch_palette]
        if missing_batches:
            raise ValueError(f"batch_palette is missing colors for batches: {missing_batches}")
        batch_colors = [batch_palette[batch] for batch in batch_categories]
    else:
        batch_colors = list(batch_palette)
        if len(batch_colors) < len(batch_categories):
            raise ValueError(
                "batch_palette must contain at least one color per batch "
                f"({len(batch_categories)} required)."
            )
        batch_colors = batch_colors[: len(batch_categories)]

    batch_cmap = ListedColormap(batch_colors)
    detection_cmap = ListedColormap(["white", "black"])

    if axes is None:
        _, (ax_batch, ax_heatmap) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [0.25, 5], "hspace": 0.03},
            sharex=True,
        )
    else:
        try:
            ax_batch = axes["batch"]
            ax_heatmap = axes["heatmap"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "axes must be a dict with keys 'batch' and 'heatmap'."
            ) from exc

    ax_batch.imshow(batch_codes[None, :], aspect="auto", interpolation="nearest", cmap=batch_cmap)
    ax_batch.set_yticks([])
    if batch_label == "auto":
        batch_label_text = batch_col
    elif batch_label is None:
        batch_label_text = ""
    else:
        batch_label_text = str(batch_label)
    ax_batch.set_ylabel(batch_label_text, rotation=0, ha="right", va="center")

    ax_heatmap.imshow(
        detected,
        aspect="auto",
        interpolation="nearest",
        cmap=detection_cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    ax_heatmap.set_ylabel("Features")
    ax_heatmap.set_yticks([])

    boundaries = []
    start = 0
    for batch in batch_categories:
        count = int((batch_labels_series == batch).sum())
        if count == 0:
            continue
        end = start + count
        boundaries.append(end - 0.5)
        start = end

    for boundary in boundaries[:-1]:
        ax_batch.axvline(boundary, color="white", linewidth=1)
        ax_heatmap.axvline(boundary, color="tab:red", linewidth=0.6, alpha=0.5)

    ax_batch.tick_params(bottom=False, top=False, left=False, right=False)
    ax_heatmap.tick_params(bottom=False, top=False, left=False, right=False)

    result_axes = {"batch": ax_batch, "heatmap": ax_heatmap}
    if return_data:
        data = {
            "detected": detected,
            "batch_labels": np.asarray(selected_batch_labels),
            "selected_features": selected_features,
            "selected_cells": selected_cells,
        }
        return result_axes, data
    return result_axes


def plot_qc_cutoff_search(
    cutoffs: dict,
    *,
    figsize: Tuple[float, float] = (8, 8),
    ax: Optional[plt.Axes] = None,
):
    """Visualise the slope-based gene-count cutoff search.

    Parameters
    ----------
    cutoffs
        Dictionary returned by :func:`scdia.qc.calculate_qc_cutoffs`. Must
        contain the ``gene_cutoff_search`` and ``columns`` sub-keys.
    figsize
        Figure size used only when ``ax`` is ``None``.
    ax
        Optional pre-created axes to draw onto.

    Returns
    -------
    matplotlib.axes.Axes
    """
    search = cutoffs.get("gene_cutoff_search")
    columns = cutoffs.get("columns", {})
    if search is None:
        raise KeyError(
            "cutoffs is missing the 'gene_cutoff_search' key. "
            "Did you call qc.calculate_qc_cutoffs?"
        )

    values = np.asarray(search["values"])
    x_positions = np.asarray(search["x_positions"])
    slope_initial = float(search["slope_initial"])
    slope_final = float(search["slope_final"])
    x_pt_initial = int(search["x_pt_initial"])
    x_pt_final = int(search["x_pt_final"])
    y_cutoff_initial = float(search["y_cutoff_initial"])
    gene_cutoff = float(cutoffs["gene_cutoff"])
    n = values.size
    n_detected_col = columns.get("n_detected_col", "n_genes_detected")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(x_positions, values, color="black", linewidth=2, label=n_detected_col)

    line_initial = slope_initial * x_positions + values[0]
    line_refined = slope_final * x_positions + values[0]
    line_final = slope_final * x_positions + (gene_cutoff - slope_final * x_pt_final)

    ax.plot(x_positions, line_initial, color="skyblue", linewidth=2.5, alpha=0.7, label="Initial slope")
    ax.plot(x_positions, line_refined, color="skyblue", linewidth=2.5, alpha=0.7, linestyle="-.", label="Refined slope")
    ax.plot(x_positions, line_final, color="skyblue", linewidth=2.5, alpha=0.55, linestyle="--", label="Cutoff tangent")

    point_a = (1, values[0])
    point_b = (n, values[-1])
    point_c = (x_pt_initial, y_cutoff_initial)
    point_d = (x_pt_final, gene_cutoff)
    ax.scatter(*point_a, color="orange", s=35, zorder=5)
    ax.scatter(*point_b, color="seagreen", s=35, zorder=5)
    ax.scatter(*point_c, color="purple", s=35, zorder=5)
    ax.scatter(*point_d, color="crimson", s=40, zorder=6)

    y_span = values[-1] - values[0] if values[-1] != values[0] else 1
    ax.annotate(
        "A (start)", xy=point_a, xytext=(10, 10), textcoords="offset points",
        color="darkorange", fontsize=10,
    )
    ax.annotate(
        "B (end)", xy=point_b, xytext=(-40, -20), textcoords="offset points",
        color="seagreen", fontsize=10,
    )
    ax.annotate(
        "C", xy=point_c, xytext=(-25, 10), textcoords="offset points",
        color="purple", fontsize=10, fontweight="bold",
    )
    ax.annotate(
        f"D cutoff\nrank={x_pt_final}, {n_detected_col}={gene_cutoff:.0f}",
        xy=point_d, xytext=(15, -5), textcoords="offset points",
        color="crimson", fontsize=10, va="center",
    )

    ax.set_ylim(values[0] - y_span * 0.05, values[-1] + y_span * 0.1)
    ax.set_xlabel("Rank")
    ax.set_ylabel(n_detected_col)
    ax.set_title("Slope-based cutoff calculation")
    ax.grid(alpha=0.2)

    return ax


def plot_qc_filter_comparison(
    adata: AnnData,
    cutoffs: dict,
    *,
    n_detected_col: Optional[str] = None,
    total_intensity_col: Optional[str] = None,
    qc_pass_col: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    axes: Optional[dict] = None,
):
    """Compare per-cell QC metrics before and after the filter pass.

    Parameters
    ----------
    cutoffs
        Dictionary returned by :func:`scdia.qc.calculate_qc_cutoffs`. Used for
        the cutoff annotation lines and (when not overridden) for resolving the
        relevant ``obs`` column names.
    n_detected_col, total_intensity_col, qc_pass_col
        Optional overrides for the ``adata.obs`` columns. If ``None``, the
        names recorded in ``cutoffs['columns']`` are used.
    axes
        Optional ``dict`` with keys ``"before_intensity"``, ``"before_genes"``,
        ``"after_intensity"``, and ``"after_genes"``. If ``None``, a new figure
        is created.

    Returns
    -------
    dict[str, matplotlib.axes.Axes]
    """
    columns = cutoffs.get("columns", {})
    n_detected_col = n_detected_col or columns.get("n_detected_col", "n_genes_detected")
    total_intensity_col = total_intensity_col or columns.get("total_intensity_col", "log2_total_intensity")
    qc_pass_col = qc_pass_col or columns.get("qc_pass_col", "pass_qc_filter")
    gene_cutoff = float(cutoffs["gene_cutoff"])
    intensity_upper = float(cutoffs["intensity_upper"])
    intensity_lower = float(cutoffs["intensity_lower"])

    for col in (n_detected_col, total_intensity_col, qc_pass_col):
        if col not in adata.obs:
            raise KeyError(f"Column '{col}' was not found in adata.obs.")

    pass_mask = adata.obs[qc_pass_col].astype(bool)

    if axes is None:
        gridspec = dict(hspace=0.0, height_ratios=[1, 1, 0.5, 1, 1])
        _, axs = plt.subplots(
            nrows=5, ncols=1, figsize=figsize, gridspec_kw=gridspec, sharex=False
        )
        axs[2].set_visible(False)
        result_axes = {
            "before_intensity": axs[0],
            "before_genes": axs[1],
            "after_intensity": axs[3],
            "after_genes": axs[4],
        }
    else:
        required = {"before_intensity", "before_genes", "after_intensity", "after_genes"}
        missing = required - set(axes)
        if missing:
            raise ValueError(f"axes is missing required keys: {sorted(missing)}.")
        result_axes = axes

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_specs = [
        ("before_intensity", total_intensity_col, colors[0], None, "Cells before filtering"),
        ("before_genes", n_detected_col, colors[1], None, None),
        ("after_intensity", total_intensity_col, colors[0], pass_mask, "Cells after filtering"),
        ("after_genes", n_detected_col, colors[1], pass_mask, None),
    ]

    for key, col, color, mask, title in plot_specs:
        ax = result_axes[key]
        if mask is None:
            series = adata.obs[col].reset_index(drop=True)
        else:
            series = adata.obs.loc[mask, col].reset_index(drop=True)
        series.plot(ax=ax, grid=True, color=color)
        ax.set_ylabel(col)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if title is not None:
            ax.set_title(title)

    result_axes["before_intensity"].axhline(intensity_upper, color="black", linestyle="--")
    result_axes["before_intensity"].axhline(intensity_lower, color="black", linestyle="--")
    result_axes["before_genes"].axhline(gene_cutoff, color="black", linestyle="--")

    for key in ("before_intensity", "after_intensity"):
        result_axes[key].xaxis.set_ticklabels([])
    for key in ("before_genes", "after_genes"):
        result_axes[key].set_xlabel("Cell index")

    return result_axes


def plot_de_volcano(
    adata: AnnData,
    de_key: str,
    *,
    adj_pval_cutoff: float = 0.05,
    log2foldchange_cutoff: float = 0.5,
    title: Optional[str] = None,
    legend: bool = True,
    show_labels: bool = True,
    n_top: Optional[int] = None,
    rank_by: str = "pval_adj",
    genes_of_interest: Optional[Sequence[str]] = None,
    fontsize: float = 12,
    figsize: Tuple[float, float] = (4, 4),
    ax: Optional[plt.Axes] = None,
    return_data: bool = False,
):
    """Plot differential expression results as a volcano plot.

    Significant up-regulated features are shown in red, significant
    down-regulated features are shown in blue, and non-significant features
    are shown in grey.

    Parameters
    ----------
    adata
        Annotated data matrix containing differential expression results in
        ``adata.uns``.
    de_key
        Key in ``adata.uns`` where ``de_test()`` stored its output.
    adj_pval_cutoff, log2foldchange_cutoff
        Significance and effect-size thresholds.
    n_top
        Maximum number of ranked significant features to label automatically.
        ``None`` disables automatic labelling; explicit ``genes_of_interest``
        are still labelled when ``show_labels=True``.
    rank_by
        Ranking method used when selecting the top ``n_top`` significant
        features for text annotation.
    genes_of_interest
        Optional list-like collection of genes to always label when
        ``show_labels=True``.
    fontsize
        Base font size for x- and y-axis labels. The title is set one point
        larger.
    figsize
        Figure size used only when ``ax`` is ``None``.
    ax
        Optional pre-created axes to draw onto.
    return_data
        If ``True``, also return the augmented results DataFrame used to
        produce the plot.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from adjustText import adjust_text
    from matplotlib import patches

    if de_key not in adata.uns:
        raise KeyError(f"Key '{de_key}' was not found in adata.uns.")

    df = adata.uns[de_key]["results"].dropna(subset=["log2foldchange"], axis=0).copy()
    if df.empty:
        raise ValueError(f"No valid results found in adata.uns['{de_key}']['results'].")
    if n_top is not None and n_top < 1:
        raise ValueError("n_top must be a positive integer or None.")
    supported_rank_by = {"pval_adj", "log2foldchange"}
    if rank_by not in supported_rank_by:
        supported_rank_by_str = ", ".join(sorted(supported_rank_by))
        raise ValueError(f"Unsupported rank_by '{rank_by}'. Choose from: {supported_rank_by_str}.")
    if genes_of_interest is not None and isinstance(genes_of_interest, str):
        genes_of_interest = [genes_of_interest]

    df["nlog_pval_adj"] = -np.log10(np.clip(df["pval_adj"], a_min=np.finfo(float).tiny, a_max=None))
    df["up"] = (df["pval_adj"] <= adj_pval_cutoff) & (df["log2foldchange"] >= log2foldchange_cutoff)
    df["down"] = (df["pval_adj"] <= adj_pval_cutoff) & (df["log2foldchange"] <= -log2foldchange_cutoff)
    df["unchanged"] = (~df["up"]) & (~df["down"])
    df["is_gene_of_interest"] = False

    gene_list = []
    if genes_of_interest is not None:
        gene_list = list(dict.fromkeys(genes_of_interest))
        missing_genes = sorted(set(gene_list) - set(df["gene"]))
        if missing_genes:
            warnings.warn(
                "The following genes_of_interest were not found in the differential "
                f"expression results and will be skipped: {', '.join(missing_genes)}",
                stacklevel=2,
            )
        if gene_list:
            df["is_gene_of_interest"] = df["gene"].isin(gene_list)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    colors = {"up": "#D9485F", "down": "#2B6CB0", "unchanged": "#B8BEC7"}

    ax.scatter(
        x=df[df["unchanged"]]["log2foldchange"],
        y=df[df["unchanged"]]["nlog_pval_adj"],
        color=colors["unchanged"], label="Not significant",
        s=18, alpha=0.45, linewidths=0,
    )
    ax.scatter(
        x=df[df["down"]]["log2foldchange"],
        y=df[df["down"]]["nlog_pval_adj"],
        color=colors["down"], label="Down-regulated",
        s=22, alpha=0.85, linewidths=0,
    )
    ax.scatter(
        x=df[df["up"]]["log2foldchange"],
        y=df[df["up"]]["nlog_pval_adj"],
        color=colors["up"], label="Up-regulated",
        s=22, alpha=0.85, linewidths=0,
    )

    sig_line_y = -np.log10(adj_pval_cutoff)
    ax.axhline(sig_line_y, color="black", linestyle="--", alpha=0.45, linewidth=1)
    ax.axvline(-log2foldchange_cutoff, color="black", linestyle="--", alpha=0.45, linewidth=1)
    ax.axvline(log2foldchange_cutoff, color="black", linestyle="--", alpha=0.45, linewidth=1)
    ax.set_xlabel("log2 Fold Change", fontsize=fontsize)
    ax.set_ylabel("-log10 Adjusted P-value", fontsize=fontsize)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_min, x_max = df["log2foldchange"].min(), df["log2foldchange"].max()
    x_len = x_max - x_min
    x_pad = x_len * 0.1 if x_len > 0 else 0.5
    y_max = df["nlog_pval_adj"].max()
    y_upper = y_max * 1.15 if y_max > 0 else 1
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0, y_upper)

    label_df = pd.DataFrame(columns=df.columns)
    if show_labels:
        label_parts = []
        if n_top is not None:
            significant_df = df.loc[~df["unchanged"]].copy()
            if rank_by == "pval_adj":
                ranked_index = significant_df["pval_adj"].sort_values(ascending=True).index[:n_top]
            else:
                ranked_index = significant_df["log2foldchange"].abs().sort_values(ascending=False).index[:n_top]
            label_parts.append(significant_df.loc[ranked_index])
        if df["is_gene_of_interest"].any():
            label_parts.append(df.loc[df["is_gene_of_interest"]].copy())
        if label_parts:
            label_df = pd.concat(label_parts, axis=0).drop_duplicates(subset="gene", keep="first")
            if rank_by == "pval_adj":
                label_df = label_df.sort_values(
                    ["is_gene_of_interest", "pval_adj"], ascending=[False, True]
                )
            else:
                label_df = label_df.assign(
                    _abs_log2foldchange=label_df["log2foldchange"].abs()
                ).sort_values(
                    ["is_gene_of_interest", "_abs_log2foldchange"],
                    ascending=[False, False],
                )

    if not label_df.empty:
        texts = [
            ax.text(
                label_df.loc[i, "log2foldchange"],
                label_df.loc[i, "nlog_pval_adj"],
                label_df.loc[i, "gene"].upper(),
                ha="center", va="center", fontsize=9,
            )
            for i in label_df.index
        ]

        patch_ymax = patches.Rectangle(
            (x_min - x_pad, y_upper * 0.96),
            (x_max - x_min) + 2 * x_pad,
            max(y_upper * 0.08, 0.5),
            fill=False, alpha=0.0,
        )
        ax.add_patch(patch_ymax)
        patch_hl = patches.Rectangle(
            (x_min - x_pad, 0),
            (x_max - x_min) + 2 * x_pad,
            max(sig_line_y, 0.5),
            fill=False, alpha=0.0,
        )
        ax.add_patch(patch_hl)

        adjust_text(
            texts,
            x=df["log2foldchange"].values,
            y=df["nlog_pval_adj"].values,
            arrowprops=dict(arrowstyle="-", color="grey", alpha=0.5, lw=0.5),
            add_objects=[patch_ymax, patch_hl],
            expand_objects=(1, 1),
            lim=100,
        )

    if title:
        ax.set_title(title, fontsize=fontsize + 1)
    else:
        ax.set_title(
            "{} vs. {}".format(
                adata.uns[de_key]["group1"], adata.uns[de_key]["group2"]
            )
        )
    if legend:
        ax.legend(frameon=False)

    if return_data:
        return ax, df
    return ax
