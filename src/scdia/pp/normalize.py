import warnings
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import gmean

NormalizeMethod = Literal["total", "median", "median_ratio", "quantile"]


def _get_matrix_from_adata(adata, layer: str | None = None):
    """Return an AnnData matrix and its source name."""
    if layer is None:
        return adata.X, "X"
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")
    return adata.layers[layer], layer


def _prepare_dense_float_matrix(
    matrix,
    *,
    zero_as_missing: bool = True,
    check_negative: bool = True,
):
    """Convert matrix to dense float and mask missing values as NaN."""
    was_sparse = issparse(matrix)
    if was_sparse:
        arr = matrix.toarray().astype(np.float64, copy=False)
    else:
        arr = np.asarray(matrix, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    if check_negative and np.any(np.isfinite(arr) & (arr < 0)):
        raise ValueError(
            "Negative values detected. Preprocessing normalization expects "
            "non-log, non-negative intensity data."
        )

    arr = arr.copy()
    missing_mask = ~np.isfinite(arr)
    if zero_as_missing:
        missing_mask |= arr == 0
    arr[missing_mask] = np.nan
    return arr, was_sparse


def _validate_observed_rows(arr: np.ndarray, method: str):
    observed_per_row = np.sum(np.isfinite(arr), axis=1)
    bad_rows = np.where(observed_per_row == 0)[0]
    if bad_rows.size:
        preview = bad_rows[:10].tolist()
        raise ValueError(
            f"{method} cannot normalize observations with no observed values. "
            f"Found {bad_rows.size} empty rows; first indices: {preview}."
        )


def _finalize_and_store(
    adata,
    values: np.ndarray,
    *,
    layer: str | None,
    input_name: str,
    output_layer: str | None,
    was_sparse: bool,
    preserve_missing: bool,
):
    values = values.copy()
    if not preserve_missing:
        values[~np.isfinite(values)] = 0

    stored = csr_matrix(values) if was_sparse else values
    if output_layer is not None:
        adata.layers[output_layer] = stored
        return output_layer
    if layer is None:
        adata.X = stored
        return "X"
    adata.layers[layer] = stored
    return input_name


def _quantile_normalize_nan(feature_by_sample: np.ndarray):
    """Classical quantile normalization with NaN support."""
    x = np.asarray(feature_by_sample, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    result = np.full_like(x, np.nan)
    sorted_cols = []
    for j in range(x.shape[1]):
        vals = x[:, j][np.isfinite(x[:, j])]
        vals.sort()
        sorted_cols.append(vals)

    max_len = max((len(vals) for vals in sorted_cols), default=0)
    if max_len == 0:
        return result

    sorted_mat = np.full((max_len, x.shape[1]), np.nan, dtype=np.float64)
    for j, vals in enumerate(sorted_cols):
        sorted_mat[: len(vals), j] = vals

    ref = np.nanmean(sorted_mat, axis=1)

    for j in range(x.shape[1]):
        col = x[:, j]
        finite_mask = np.isfinite(col)
        n_valid = int(finite_mask.sum())
        if n_valid == 0:
            continue

        order = np.argsort(col[finite_mask], kind="mergesort")
        ranks = np.empty(n_valid, dtype=int)
        ranks[order] = np.arange(n_valid)
        result[np.where(finite_mask)[0], j] = ref[ranks]

    return result


def normalize(
    adata,
    method: NormalizeMethod = "total",
    *,
    layer: str | None = None,
    output_layer: str | None = None,
    inplace: bool = True,
    **kwargs,
):
    """Normalize an AnnData matrix using a Scanpy-style dispatcher."""
    methods = {
        "total": normalize_total,
        "median": normalize_median,
        "median_ratio": normalize_median_ratio,
        "quantile": normalize_quantile,
    }
    if method not in methods:
        supported = ", ".join(sorted(methods))
        raise ValueError(f"Unsupported normalization method '{method}'. Choose from: {supported}.")

    return methods[method](
        adata,
        layer=layer,
        output_layer=output_layer,
        inplace=inplace,
        **kwargs,
    )


def normalize_total(
    adata,
    *,
    target_sum: float | None = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: str | None = None,
    layer: str | None = None,
    output_layer: str | None = None,
    inplace: bool = True,
    zero_as_missing: bool = True,
    preserve_missing: bool = False,
    check_negative: bool = True,
):
    """Normalize each observation by total observed intensity.

    This follows the main Scanpy ``normalize_total`` behavior: if
    ``target_sum`` is ``None``, each observation is normalized to the median
    positive total intensity.
    """
    if max_fraction < 0 or max_fraction > 1:
        raise ValueError("max_fraction must be between 0 and 1.")
    if target_sum is not None and target_sum <= 0:
        raise ValueError("target_sum must be positive.")

    ad = adata if inplace else adata.copy()
    matrix, input_name = _get_matrix_from_adata(ad, layer)
    x, was_sparse = _prepare_dense_float_matrix(
        matrix,
        zero_as_missing=zero_as_missing,
        check_negative=check_negative,
    )
    _validate_observed_rows(x, "normalize_total")

    counts = np.nansum(x, axis=1)

    gene_subset = None
    if exclude_highly_expressed:
        high = x > counts[:, None] * max_fraction
        gene_subset = np.nansum(high, axis=0) == 0
        if not np.any(gene_subset):
            raise ValueError("No features remain after excluding highly expressed features.")
        counts = np.nansum(x[:, gene_subset], axis=1)

    if target_sum is None:
        positive_counts = counts[counts > 0]
        if positive_counts.size == 0:
            raise ValueError("Cannot infer target_sum because no positive totals were found.")
        resolved_target_sum = float(np.median(positive_counts))
    else:
        resolved_target_sum = float(target_sum)

    with np.errstate(divide="ignore", invalid="ignore"):
        size_factors = counts / resolved_target_sum

    bad = ~np.isfinite(size_factors) | (size_factors <= 0)
    if np.any(bad):
        bad_rows = np.where(bad)[0][:10].tolist()
        raise ValueError(f"Invalid total-normalization size factors; first rows: {bad_rows}.")

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = x / size_factors[:, None]

    output_name = _finalize_and_store(
        ad,
        normalized,
        layer=layer,
        input_name=input_name,
        output_layer=output_layer,
        was_sparse=was_sparse,
        preserve_missing=preserve_missing,
    )

    if key_added is not None:
        ad.obs[key_added] = size_factors

    if not inplace:
        return ad
    return None


def normalize_median(
    adata,
    *,
    target: str | float = "geometric_mean",
    layer: str | None = None,
    output_layer: str | None = None,
    inplace: bool = True,
    zero_as_missing: bool = True,
    preserve_missing: bool = False,
    check_negative: bool = True,
    store_factors: bool = True,
):
    """Per-observation median normalization for non-log intensity data."""
    ad = adata if inplace else adata.copy()
    matrix, input_name = _get_matrix_from_adata(ad, layer)
    x, was_sparse = _prepare_dense_float_matrix(
        matrix,
        zero_as_missing=zero_as_missing,
        check_negative=check_negative,
    )
    _validate_observed_rows(x, "normalize_median")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_medians = np.nanmedian(x, axis=1)

    valid = np.isfinite(row_medians) & (row_medians > 0)
    if not np.all(valid):
        bad_rows = np.where(~valid)[0][:10].tolist()
        raise ValueError(f"Invalid row medians for median normalization; first rows: {bad_rows}.")

    if target == "geometric_mean":
        target_median = float(np.exp(np.mean(np.log(row_medians))))
    elif target == "median_of_medians":
        target_median = float(np.median(row_medians))
    elif isinstance(target, (int, float, np.integer, np.floating)):
        target_median = float(target)
        if target_median <= 0:
            raise ValueError("Numeric target must be positive.")
    else:
        raise ValueError("target must be 'geometric_mean', 'median_of_medians', or a positive number.")

    scale_factors = target_median / row_medians
    bad = ~np.isfinite(scale_factors) | (scale_factors <= 0)
    if np.any(bad):
        bad_rows = np.where(bad)[0][:10].tolist()
        raise ValueError(f"Invalid median-normalization scale factors; first rows: {bad_rows}.")

    normalized = x * scale_factors[:, None]
    output_name = _finalize_and_store(
        ad,
        normalized,
        layer=layer,
        input_name=input_name,
        output_layer=output_layer,
        was_sparse=was_sparse,
        preserve_missing=preserve_missing,
    )

    if store_factors:
        ad.obs[f"{output_name}_median_before"] = row_medians
        ad.obs[f"{output_name}_scale_factor"] = scale_factors

    if not inplace:
        return ad
    return None


def normalize_median_ratio(
    adata,
    *,
    layer: str | None = None,
    output_layer: str | None = None,
    inplace: bool = True,
    zero_as_missing: bool = True,
    preserve_missing: bool = False,
    check_negative: bool = True,
    store_factors: bool = True,
):
    """Apply SCeptre-style median-ratio normalization."""
    ad = adata if inplace else adata.copy()
    matrix, input_name = _get_matrix_from_adata(ad, layer)
    x, was_sparse = _prepare_dense_float_matrix(
        matrix,
        zero_as_missing=zero_as_missing,
        check_negative=check_negative,
    )
    _validate_observed_rows(x, "normalize_median_ratio")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore"):
            geometric_means = gmean(x, axis=0, nan_policy="omit")
    geometric_means[~np.isfinite(geometric_means) | (geometric_means <= 0)] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = x / geometric_means[None, :]
            size_factors = np.nanmedian(ratios, axis=1)

    bad = ~np.isfinite(size_factors) | (size_factors <= 0)
    if np.any(bad):
        bad_rows = np.where(bad)[0][:10].tolist()
        raise ValueError(f"Invalid median-ratio size factors; first rows: {bad_rows}.")

    normalized = x / size_factors[:, None]
    output_name = _finalize_and_store(
        ad,
        normalized,
        layer=layer,
        input_name=input_name,
        output_layer=output_layer,
        was_sparse=was_sparse,
        preserve_missing=preserve_missing,
    )

    if store_factors:
        ad.obs[f"{output_name}_size_factor"] = size_factors

    if not inplace:
        return ad
    return None


def normalize_quantile(
    adata,
    *,
    layer: str | None = None,
    output_layer: str | None = None,
    inplace: bool = True,
    zero_as_missing: bool = True,
    preserve_missing: bool = False,
    check_negative: bool = True,
):
    """Classical quantile normalization across observations."""
    ad = adata if inplace else adata.copy()
    matrix, input_name = _get_matrix_from_adata(ad, layer)
    x, was_sparse = _prepare_dense_float_matrix(
        matrix,
        zero_as_missing=zero_as_missing,
        check_negative=check_negative,
    )
    _validate_observed_rows(x, "normalize_quantile")

    normalized = _quantile_normalize_nan(x.T).T
    output_name = _finalize_and_store(
        ad,
        normalized,
        layer=layer,
        input_name=input_name,
        output_layer=output_layer,
        was_sparse=was_sparse,
        preserve_missing=preserve_missing,
    )

    if not inplace:
        return ad
    return None
