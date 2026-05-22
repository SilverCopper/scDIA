import numpy as np
import pandas as pd
from typing import Union, Literal, Optional, Sequence, Mapping
from anndata import AnnData


def _to_1d_array(values):
    """Convert dense or sparse column slices to a flat NumPy array."""
    if hasattr(values, "toarray"):
        return values.toarray().ravel()
    return np.asarray(values).ravel()


def _deduplicate_preserve_order(values):
    seen = set()
    deduplicated = []
    for value in values:
        if pd.isna(value):
            continue
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def _is_gmt_path(value):
    return isinstance(value, str) and value.lower().endswith(".gmt")


def _is_enrichr_library_input(gene_sets):
    if isinstance(gene_sets, Mapping):
        return False
    if isinstance(gene_sets, str):
        return not _is_gmt_path(gene_sets)
    if isinstance(gene_sets, Sequence) and not isinstance(gene_sets, str):
        return all(isinstance(item, str) and not _is_gmt_path(item) for item in gene_sets)
    return False


def _download_enrichr_library_as_dict(
    gp,
    gene_sets,
    organism: str,
    min_size: int,
    max_size: int,
):
    if isinstance(gene_sets, str):
        return gp.get_library(
            name=gene_sets,
            organism=organism,
            min_size=min_size,
            max_size=max_size,
        )

    resolved = {}
    for library in gene_sets:
        library_dict = gp.get_library(
            name=library,
            organism=organism,
            min_size=min_size,
            max_size=max_size,
        )
        for term, genes in library_dict.items():
            resolved[f"{library}: {term}"] = genes
    return resolved


def _normalize_gene_case(genes, gene_case):
    if gene_case == "upper":
        return _deduplicate_preserve_order([gene.upper() for gene in genes])
    if gene_case == "as_is":
        return _deduplicate_preserve_order(genes)
    raise ValueError("gene_case must resolve to 'upper' or 'as_is'.")


def _detected_genes_from_adata(
    adata: AnnData,
    layer: Optional[str] = None,
    missing_value: Union[float, None] = 0,
    min_detected: int = 1,
):
    if min_detected < 1:
        raise ValueError("min_detected must be at least 1.")
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' was not found in adata.layers.")
        matrix = adata.layers[layer]
    else:
        matrix = adata.X

    if missing_value is None:
        return list(adata.var_names)

    if hasattr(matrix, "getnnz") and missing_value == 0:
        detected_counts = np.asarray(matrix.getnnz(axis=0)).ravel()
    else:
        X = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        if isinstance(missing_value, float) and np.isnan(missing_value):
            detected_mask = np.isfinite(X)
        else:
            detected_mask = np.isfinite(X) & (X != missing_value)
        detected_counts = detected_mask.sum(axis=0)
    return list(adata.var_names[np.asarray(detected_counts) >= min_detected])


def get_gseapy_libraries(organism: str = "Mouse"):
    """Return active Enrichr library names available through GSEApy.

    Parameters
    ----------
    organism
        Enrichr organism database. GSEApy supports ``"Human"``, ``"Mouse"``,
        ``"Yeast"``, ``"Fly"``, ``"Fish"``, and ``"Worm"``.

    Returns
    -------
    list[str]
        Active Enrichr library names for the selected organism.
    """
    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            "gseapy is required to list enrichment libraries. Install it with "
            "`pip install gseapy`."
        ) from exc

    return gp.get_library_name(organism=organism)


def enrich_de_genes(
    adata: AnnData,
    de_key: str,
    gene_sets: Union[str, Sequence[str], Mapping[str, Sequence[str]]],
    direction: Literal["up", "down"] = "up",
    log2foldchange_cutoff: float = 0.5,
    pvalue_cutoff: float = 0.05,
    pvalue_col: str = "pval_adj",
    organism: str = "Mouse",
    background: Union[Literal["detected"], Sequence[str], int, str, None] = "detected",
    layer: Optional[str] = None,
    background_missing_value: Union[float, None] = 0,
    min_detected: int = 1,
    gene_case: Literal["auto", "upper", "as_is"] = "auto",
    library_mode: Literal["auto", "local", "online"] = "auto",
    gene_set_min_size: int = 0,
    gene_set_max_size: int = 2000,
    key_added: Optional[str] = None,
    outdir: Optional[str] = None,
    cutoff: float = 0.05,
    plot: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Run GSEApy/Enrichr over-representation analysis from DE results.

    The function reads differential expression results from ``adata.uns[de_key]``,
    selects up- or down-regulated genes with the requested log2 fold-change
    and p-value thresholds, and uses detected genes from ``adata`` as the
    default enrichment background.

    Parameters
    ----------
    adata
        Annotated data matrix.
    de_key
        Key in ``adata.uns`` where ``de_test()`` stored differential
        expression results.
    gene_sets
        Enrichr library name, list of library names, GMT path, or a dictionary
        mapping terms to gene lists.
    direction
        Select ``"up"`` genes with positive log2 fold-change or ``"down"``
        genes with negative log2 fold-change.
    log2foldchange_cutoff
        Absolute log2 fold-change cutoff used for selecting DE genes.
    pvalue_cutoff
        P-value cutoff applied to ``pvalue_col``.
    pvalue_col
        P-value column in the DE result table. Defaults to ``"pval_adj"``.
    organism
        Enrichr organism argument passed to GSEApy.
    background
        ``"detected"`` uses all detected genes in ``adata``. A custom gene
        list, integer, string background, or ``None`` is passed through to
        GSEApy.
    layer
        Layer used to determine detected genes when ``background="detected"``.
        If ``None``, the layer stored by ``de_test()`` is reused when present,
        otherwise ``adata.X`` is used.
    background_missing_value
        Value treated as not detected when deriving the background from
        ``adata``. Defaults to zero.
    min_detected
        Minimum number of cells/samples in which a gene must be detected to be
        included in the background.
    gene_case
        ``"auto"`` uppercases genes for Enrichr library names and preserves
        case for custom GMT/dict gene sets. Use ``"upper"`` or ``"as_is"`` to
        force a behavior.
    library_mode
        How to handle Enrichr library names. ``"auto"`` downloads Enrichr
        libraries as local dictionaries before enrichment, which avoids the
        online Speedrichr background endpoint. ``"online"`` preserves the
        original online Enrichr call.
    gene_set_min_size
        Minimum size for downloaded Enrichr gene sets when
        ``library_mode="auto"`` or ``"local"``.
    gene_set_max_size
        Maximum size for downloaded Enrichr gene sets when
        ``library_mode="auto"`` or ``"local"``.
    key_added
        Key used to store enrichment output in ``adata.uns``. Defaults to
        ``"{de_key}_{direction}_enrichr"``.
    outdir
        Output directory passed to GSEApy. Defaults to ``None`` to avoid
        writing files.
    cutoff, plot, verbose, kwargs
        Additional GSEApy ``enrichr`` arguments.

    Returns
    -------
    pandas.DataFrame
        Enrichment result table returned by GSEApy.
    """
    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            "gseapy is required for enrichment analysis. Install it with "
            "`pip install gseapy`."
        ) from exc

    if de_key not in adata.uns:
        raise KeyError(f"Key '{de_key}' was not found in adata.uns.")
    if "results" not in adata.uns[de_key]:
        raise KeyError(f"adata.uns['{de_key}'] does not contain a 'results' table.")
    supported_directions = {"up", "down"}
    if direction not in supported_directions:
        raise ValueError("direction must be either 'up' or 'down'.")
    supported_library_modes = {"auto", "local", "online"}
    if library_mode not in supported_library_modes:
        raise ValueError("library_mode must be one of: 'auto', 'local', 'online'.")

    de_results = adata.uns[de_key]["results"].copy()
    required_cols = {"gene", "log2foldchange", pvalue_col}
    missing_cols = sorted(required_cols - set(de_results.columns))
    if missing_cols:
        raise KeyError(
            "DE result table is missing required column(s): "
            + ", ".join(missing_cols)
        )

    valid = de_results.dropna(subset=["gene", "log2foldchange", pvalue_col]).copy()
    if direction == "up":
        selected_mask = (
            (valid["log2foldchange"] >= log2foldchange_cutoff)
            & (valid[pvalue_col] <= pvalue_cutoff)
        )
    else:
        selected_mask = (
            (valid["log2foldchange"] <= -log2foldchange_cutoff)
            & (valid[pvalue_col] <= pvalue_cutoff)
        )
    selected_genes_raw = _deduplicate_preserve_order(valid.loc[selected_mask, "gene"])
    if not selected_genes_raw:
        raise ValueError(
            "No genes passed the requested enrichment thresholds "
            f"for direction='{direction}'."
        )

    if gene_case == "auto":
        resolved_gene_case = "upper" if _is_enrichr_library_input(gene_sets) else "as_is"
    elif gene_case in {"upper", "as_is"}:
        resolved_gene_case = gene_case
    else:
        raise ValueError("gene_case must be one of: 'auto', 'upper', 'as_is'.")
    selected_genes = _normalize_gene_case(selected_genes_raw, resolved_gene_case)

    if isinstance(background, str) and background == "detected":
        layer_for_background = (
            layer if layer is not None else adata.uns[de_key].get("layer")
        )
        background_raw = _detected_genes_from_adata(
            adata,
            layer=layer_for_background,
            missing_value=background_missing_value,
            min_detected=min_detected,
        )
        enrichment_background = _normalize_gene_case(background_raw, resolved_gene_case)
    elif isinstance(background, Sequence) and not isinstance(background, str):
        background_raw = _deduplicate_preserve_order(background)
        enrichment_background = _normalize_gene_case(background_raw, resolved_gene_case)
    else:
        background_raw = background
        enrichment_background = background

    resolved_gene_sets = gene_sets
    used_library_mode = library_mode
    if _is_enrichr_library_input(gene_sets):
        if library_mode in {"auto", "local"}:
            resolved_gene_sets = _download_enrichr_library_as_dict(
                gp,
                gene_sets=gene_sets,
                organism=organism,
                min_size=gene_set_min_size,
                max_size=gene_set_max_size,
            )
            used_library_mode = "local"
        else:
            used_library_mode = "online"

    enr = gp.enrichr(
        gene_list=selected_genes,
        gene_sets=resolved_gene_sets,
        organism=organism,
        background=enrichment_background,
        outdir=outdir,
        cutoff=cutoff,
        no_plot=not plot,
        verbose=verbose,
        **kwargs,
    )
    results = enr.results.copy() if getattr(enr, "results", None) is not None else pd.DataFrame()
    key_added = key_added or f"{de_key}_{direction}_enrichr"
    resolved_gene_sets_n_terms = (
        len(resolved_gene_sets) if isinstance(resolved_gene_sets, Mapping) else None
    )
    adata.uns[key_added] = {
        "de_key": de_key,
        "direction": direction,
        "gene_sets": gene_sets,
        "organism": organism,
        "log2foldchange_cutoff": log2foldchange_cutoff,
        "pvalue_cutoff": pvalue_cutoff,
        "pvalue_col": pvalue_col,
        "gene_case": resolved_gene_case,
        "library_mode": used_library_mode,
        "gene_set_min_size": gene_set_min_size,
        "gene_set_max_size": gene_set_max_size,
        "selected_genes": selected_genes,
        "selected_genes_original": selected_genes_raw,
        "background": enrichment_background,
        "background_original": background_raw,
        "resolved_gene_sets_n_terms": resolved_gene_sets_n_terms,
        "results": results,
    }
    return results


def de_test(
    adata: AnnData,
    groupby: str,
    group1: str,
    group2: Union[str, Sequence[str], None] = None,
    test: Literal["mannwhitneyu", "wilcoxon", "welch", "student"] = "mannwhitneyu",
    key_added: str = "de_test",
    layer: Optional[str] = None,
    missing_value: Union[float, None] = 0,
    log_input: bool = True,
):
    """Perform differential expression testing between two cell groups.

    This function compares feature-level abundances between ``group1`` and
    ``group2`` using one of four supported test labels: Mann-Whitney U test,
    Wilcoxon rank-sum test, Welch's t-test, or Student's t-test. If
    ``group2`` is not provided,
    ``group1`` is compared against all remaining cells. Raw p-values are
    adjusted using the Benjamini-Hochberg false discovery rate procedure.

    Missing values can be excluded by setting ``missing_value`` to ``0`` or ``np.nan``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        Column in ``adata.obs`` containing the group labels.
    group1
        Name of the first comparison group.
    group2
        Name of the second comparison group, or a list-like collection of
        group names. If ``None``, ``group1`` is tested against all remaining
        cells.
    test
        Statistical test to apply. Supported values are ``"mannwhitneyu"``,
        ``"wilcoxon"``, ``"welch"``, and ``"student"``.
    key_added
        Key used to store the output in ``adata.uns``.
    layer
        Layer name in ``adata.layers`` to use as the input matrix. If ``None``,
        ``adata.X`` is used.
    missing_value
        Value to exclude before testing. Common choices are ``0`` and
        ``np.nan`` for missing-value handling. If ``None``, no filtering is
        applied.
    log_input
        Whether the input data are already log-transformed. If ``True``, the
        reported effect size is the difference in group means. If ``False``,
        the reported effect size is the ratio of group means.

    Returns
    -------
    None
        Results are stored in ``adata.uns[key_added]`` as a dictionary containing the
        comparison metadata and a results table with p-values, adjusted
        p-values, effect sizes, per-group sample counts, and per-group
        detection fractions (``pct1`` and ``pct2``).

    Raises
    ------
    ValueError
        If ``test`` is not one of the supported methods.
    KeyError
        If ``layer`` is provided but does not exist in ``adata.layers``.
    """
    from scipy.stats import mannwhitneyu, ranksums
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    supported_tests = {"mannwhitneyu", "wilcoxon", "welch", "student"}
    if test not in supported_tests:
        supported_tests_str = ", ".join(sorted(supported_tests))
        raise ValueError(f"Unsupported test '{test}'. Choose from: {supported_tests_str}.")

    if groupby not in adata.obs:
        raise KeyError(f"Column '{groupby}' was not found in adata.obs.")

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' was not found in adata.layers.")
        data_matrix = adata.layers[layer]
    else:
        data_matrix = adata.X

    results = []
    genes = adata.var_names
    group1_vals = data_matrix[adata.obs[groupby] == group1]
    if group2 is not None:
        # Normalize to a list so both a single label and a list of labels work.
        group2_labels = [group2] if isinstance(group2, str) else list(group2)
        group2_vals = data_matrix[adata.obs[groupby].isin(group2_labels)]
    else:
        group2_vals = data_matrix[adata.obs[groupby] != group1]
    for i, gene in enumerate(genes):
        vals1_all = _to_1d_array(group1_vals[:, i])
        vals2_all = _to_1d_array(group2_vals[:, i])
        vals1 = vals1_all.copy()
        vals2 = vals2_all.copy()
        if missing_value is not None:
            if np.isnan(missing_value):
                vals1 = vals1[~np.isnan(vals1)]
                vals2 = vals2[~np.isnan(vals2)]
            else:
                vals1 = vals1[vals1 != missing_value]
                vals2 = vals2[vals2 != missing_value]
        pct1 = len(vals1) / len(vals1_all) if len(vals1_all) > 0 else np.nan
        pct2 = len(vals2) / len(vals2_all) if len(vals2_all) > 0 else np.nan
        if len(vals1) > 2 and len(vals2) > 2:  # at least 3 values per group
            if test == "welch":
                test_res = ttest_ind(vals1, vals2, equal_var=False)
            elif test == "student":
                test_res = ttest_ind(vals1, vals2, equal_var=True)
            elif test == "wilcoxon":
                test_res = ranksums(vals1, vals2)
            else:
                test_res = mannwhitneyu(vals1, vals2, alternative="two-sided")
            if log_input:
                fold_change = np.mean(vals1) - np.mean(vals2)
            else:
                fold_change = np.mean(vals1) / np.mean(vals2)
            results.append(
                {
                    "gene": gene,
                    "pval": test_res[1],
                    "log2foldchange": fold_change,
                    "size1": len(vals1),
                    "size2": len(vals2),
                    "pct1": pct1,
                    "pct2": pct2,
                }
            )
        else:
            results.append(
                {
                    "gene": gene,
                    "pval": 1.0,
                    "log2foldchange": np.nan,
                    "size1": len(vals1),
                    "size2": len(vals2),
                    "pct1": pct1,
                    "pct2": pct2,
                }
            )

    results = pd.DataFrame.from_records(results)
    results["pval"] = results["pval"].fillna(1.0)
    _, pvals_adj, _, _ = multipletests(results["pval"], alpha=0.05, method="fdr_bh")
    results["pval_adj"] = pvals_adj
    results = results.sort_values("log2foldchange", ascending=False)
    adata.uns[key_added] = {
        "groupby": groupby,
        "group1": group1,
        "group2": group2 if group2 is not None else "Rest",
        "layer": layer,
        "missing_value": missing_value,
        "log_input": log_input,
        "results": results,
    }

    
