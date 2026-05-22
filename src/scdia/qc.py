import warnings

from anndata import AnnData

import numpy as np
import pandas as pd

from scipy.sparse import issparse
from scipy.optimize import minimize_scalar


def _get_matrix(adata: AnnData, layer: str = None):
    """Return adata.X or an AnnData layer."""
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' was not found in adata.layers.")
    return adata.layers[layer]


def _to_dense_float(X):
    """Convert sparse/dense input to a 2D float array."""
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")
    return X


def _mask_missing_values(X, zero_as_missing: bool = True):
    """Return a float copy where non-finite values and optional zeros are NaN."""
    X_masked = _to_dense_float(X).copy()
    missing_mask = ~np.isfinite(X_masked)
    if zero_as_missing:
        missing_mask |= X_masked == 0
    X_masked[missing_mask] = np.nan
    return X_masked


def calculate_cell_qc_metrics(adata: AnnData,
                              layer: str = None,
                              zero_as_missing: bool = True,
                              inplace: bool = True) -> pd.DataFrame:
    """
    Calculate cell-level QC metrics from a linear intensity matrix.

    Zeros and non-finite values are treated as missing by default.
    """
    X = _mask_missing_values(_get_matrix(adata, layer), zero_as_missing=zero_as_missing)
    n_cells, n_features = X.shape

    detected = np.sum(np.isfinite(X), axis=1)
    total_intensity = np.nansum(X, axis=1)
    total_intensity[detected == 0] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        log2_total_intensity = np.where(total_intensity > 0, np.log2(total_intensity), np.nan)

    metrics = pd.DataFrame(
        {
            "n_genes_detected": detected.astype(int),
            "total_intensity": total_intensity,
            "log2_total_intensity": log2_total_intensity,
            "pct_missing": (1 - detected / n_features) * 100 if n_features else np.nan,
        },
        index=adata.obs_names,
    )

    if inplace:
        for col in metrics.columns:
            adata.obs[col] = metrics[col].values
        return None

    return metrics


def calculate_feature_qc_metrics(adata: AnnData,
                                 layer: str = None,
                                 zero_as_missing: bool = True,
                                 inplace: bool = True) -> pd.DataFrame:
    """
    Calculate feature-level detection and intensity summaries.

    Rows are cells/samples and columns are features.
    """
    X = _mask_missing_values(_get_matrix(adata, layer), zero_as_missing=zero_as_missing)
    n_cells = X.shape[0]

    detected = np.sum(np.isfinite(X), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_intensity = np.nanmean(X, axis=0)
        median_intensity = np.nanmedian(X, axis=0)

    metrics = pd.DataFrame(
        {
            "n_cells_detected": detected.astype(int),
            "pct_detected_cells": (detected / n_cells) * 100 if n_cells else np.nan,
            "mean_intensity": mean_intensity,
            "median_intensity": median_intensity,
        },
        index=adata.var_names,
    )

    if inplace:
        for col in metrics.columns:
            adata.var[col] = metrics[col].values
        return None

    return metrics


def report_missing_values(adata: AnnData,
                          layer: str = None,
                          groupby: str = None,
                          zero_as_missing: bool = True) -> pd.DataFrame:
    """
    Report missing-value summaries globally or by obs group.
    """
    X = _mask_missing_values(_get_matrix(adata, layer), zero_as_missing=zero_as_missing)

    def _summarize(mask, group_name):
        X_group = X[mask, :]
        n_cells, n_features = X_group.shape
        detected = np.isfinite(X_group)
        detected_per_cell = detected.sum(axis=1)
        total_entries = n_cells * n_features
        detected_entries = int(detected.sum())

        return {
            "group": group_name,
            "n_cells": int(n_cells),
            "n_detected_features_total": int(detected.any(axis=0).sum()) if n_cells else 0,
            "detected_features_percent": float((detected.any(axis=0).sum() / n_features) * 100) if n_features else np.nan,
            "mean_detected_features_per_cell": float(np.mean(detected_per_cell)) if n_cells else np.nan,
            "sd_detected_features_per_cell": float(np.std(detected_per_cell, ddof=1)) if n_cells > 1 else np.nan,
            "detection_rate": detected_entries / total_entries if total_entries else np.nan,
            "missing_rate": 1 - detected_entries / total_entries if total_entries else np.nan,
        }

    if groupby is None:
        rows = [_summarize(np.ones(adata.n_obs, dtype=bool), "all")]
    else:
        if groupby not in adata.obs:
            raise ValueError(f"Column '{groupby}' was not found in adata.obs.")
        rows = []
        groups = adata.obs[groupby]
        for group_name in groups.dropna().unique():
            rows.append(_summarize((groups == group_name).to_numpy(), group_name))

    return pd.DataFrame(rows)


def calculate_qc_cutoffs(adata: AnnData,
                         n_detected_col: str = 'n_genes_detected',
                         total_intensity_col: str = 'log2_total_intensity',
                         intensity_thresh: float = 3.5,
                         qc_pass_col: str = 'pass_qc_filter',
                         verbose: bool = False) -> dict:
    '''
    Calculate QC cutoffs for cell filtering.

    This function performs two tasks:
    1. Calculates a minimum gene count threshold using a slope-based "elbow" optimization.
    2. Calculates upper and lower intensity limits using a MAD-based outlier detection method.

    Use :func:`scdia.pl.plot_qc_cutoff_search` and
    :func:`scdia.pl.plot_qc_filter_comparison` to visualise the result.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the metrics in ``.obs``.
    n_detected_col : str
        The column name in ``adata.obs`` for the number of genes detected.
    total_intensity_col : str
        The column name in ``adata.obs`` for the log2 transformed total intensity.
    intensity_thresh : float
        The threshold for modified Z-score in MAD-based outlier detection. Defaults to 3.5.
    qc_pass_col : str
        Column name in ``adata.obs`` used to store the final pass/fail QC flag.
    verbose : bool
        If True, prints optimization progress and results.

    Returns
    -------
    dict
        Dictionary with the following keys:
        - ``gene_cutoff``: Calculated minimum gene count.
        - ``intensity_upper`` / ``intensity_lower``: Intensity limits from MAD.
        - ``n_cells_total`` / ``n_cells_pass`` / ``pass_ratio``: filter summary.
        - ``columns``: input column names, useful for downstream plotting.
        - ``gene_cutoff_search``: intermediate slope-search state consumed by
          :func:`scdia.pl.plot_qc_cutoff_search`.
    '''
    
    # 1. Calculate gene number minimum cutoff (Slope Method)
    gene_counts = np.array(adata.obs[n_detected_col])
    values = np.sort(np.clip(np.asarray(gene_counts, dtype=float), a_min=0, a_max=None))
    n = values.size
    if n == 0:
        raise ValueError(f'Column {n_detected_col} is empty or not found.')
    
    x_positions = np.arange(1, n + 1, dtype=float)
    
    def count_points_above_line(x_value, slope_value):
        x_int = int(x_value)
        if x_int < 1 or x_int > n:
            return n
        y_point = values[x_int - 1]
        intercept = y_point - slope_value * x_int
        return np.sum(values >= (x_positions * slope_value + intercept))
    
    # Step 1: Initial coarse optimization
    slope_initial = (values[-1] - values[0]) / n if n > 0 else 0.0
    result_initial = minimize_scalar(lambda x: count_points_above_line(x, slope_initial), 
                                     bounds=(1, n), method='bounded')
    x_pt_initial = int(np.clip(np.round(result_initial.x), 1, n))
    y_cutoff_initial = values[x_pt_initial - 1]
    
    # Step 2: Refined optimization
    slope_final = (y_cutoff_initial - values[0]) / x_pt_initial if x_pt_initial > 0 else 0.0
    result_final = minimize_scalar(lambda x: count_points_above_line(x, slope_final), 
                                   bounds=(1, x_pt_initial), method='bounded')
    x_pt_final = int(np.clip(np.round(result_final.x), 1, n))
    gene_cutoff = values[x_pt_final - 1]

    if verbose:
        print(f"Calculated gene count cutoff (slope method): {gene_cutoff}")

    # 2. Calculate intensity limits (MAD-based Outlier Method)
    intensities = adata.obs[total_intensity_col].to_numpy()
    if len(intensities.shape) == 1:
        points = intensities[:, None]
    else:
        points = intensities
        
    median = np.median(points, axis=0)
    diff = np.sqrt(np.sum((points - median) ** 2, axis=-1))
    med_abs_deviation = np.median(diff)

    # Avoid division by zero
    if med_abs_deviation == 0:
        modified_z_score = np.zeros_like(diff)
    else:
        modified_z_score = 0.6745 * diff / med_abs_deviation
        
    df_intensity = pd.DataFrame({
        "points": points.flatten(), 
        "modified_z_score": modified_z_score.flatten()
    })
    
    intensity_upper = df_intensity[
        (df_intensity["modified_z_score"] <= intensity_thresh) & (df_intensity["points"] >= median[0])
    ]["points"].max()
    
    intensity_lower = df_intensity[
        (df_intensity["modified_z_score"] <= intensity_thresh) & (df_intensity["points"] <= median[0])
    ]["points"].min()

    # 3. Apply the QC filter and annotate adata.obs
    pass_mask = (
        (adata.obs[n_detected_col] >= gene_cutoff)
        & (adata.obs[total_intensity_col] >= intensity_lower)
        & (adata.obs[total_intensity_col] <= intensity_upper)
    )
    adata.obs[qc_pass_col] = pass_mask.astype(bool)

    n_cells_total = int(adata.n_obs)
    n_cells_pass = int(pass_mask.sum())
    pass_ratio = n_cells_pass / n_cells_total if n_cells_total > 0 else 0.0

    if verbose:
        print(f"Calculated intensity limits (MAD method): lower={intensity_lower:.4f}, upper={intensity_upper:.4f}")
        print(f"QC pass cells: {n_cells_pass}/{n_cells_total} ({pass_ratio:.2%})")

    return {
        'gene_cutoff': gene_cutoff,
        'intensity_upper': intensity_upper,
        'intensity_lower': intensity_lower,
        'n_cells_total': n_cells_total,
        'n_cells_pass': n_cells_pass,
        'pass_ratio': pass_ratio,
        'columns': {
            'n_detected_col': n_detected_col,
            'total_intensity_col': total_intensity_col,
            'qc_pass_col': qc_pass_col,
        },
        'gene_cutoff_search': {
            'values': values,
            'x_positions': x_positions,
            'slope_initial': slope_initial,
            'slope_final': slope_final,
            'x_pt_initial': x_pt_initial,
            'x_pt_final': x_pt_final,
            'y_cutoff_initial': y_cutoff_initial,
        },
    }
