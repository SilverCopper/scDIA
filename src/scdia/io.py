"""Reading utilities for DIA single-cell proteomics reports.

This module standardizes search-engine reports into three matrix styles that
are useful in downstream analysis:

1. precursor-level wide matrix
2. protein-group-level wide matrix
3. gene-level wide matrix

Supported engines:

- ``spectronaut``: long-format report.tsv
- ``diann``: long-format DIA-NN report.tsv
- ``fragpipe``: wide-format ``combined_protein.tsv`` (pg/gene levels) and
  ``combined_ion.tsv`` (precursor level). Sample IDs are taken from the
  column-name prefix; ``quantity_mode`` controls which intensity flavour
  (``"maxlfq"`` / ``"intensity"`` / ``"unique"`` / ``"razor"``) is selected.

The engine is auto-detected from input columns by default; pass an explicit
``engine=...`` to override.
"""

import os

import numpy as np
import pandas as pd


ENGINE_SCHEMAS = {
    "spectronaut": {
        "format": "long",
        "detect_columns": {"PG.ProteinGroups", "PG.Genes", "R.FileName"},
        "sample_metrics": {
            "sample_col": "R.FileName",
            "proteingroups_col": "PG.ProteinGroups",
            "peptide_col": "EG.ModifiedPeptide",
        },
        "levels": {
            "precursor": {
                "sample_col": "R.FileName",
                "protein_col": "PG.ProteinGroups",
                "gene_col": "PG.Genes",
                "quantity_modes": {
                    "default": "FG.Quantity",
                },
                "feature_id_candidates": ['FG.Id'],
                "feature_fallback_cols": ["EG.ModifiedPeptide", "FG.Charge"],
                "quality_filters": [
                    ("EG.IsDecoy", "==", False),
                    ("EG.UsedForPeptideQuantity", "==", True),
                ],
            },
            "pg": {
                "sample_col": "R.FileName",
                "protein_col": "PG.ProteinGroups",
                "gene_col": "PG.Genes",
                "quantity_modes": {
                    "default": "PG.Quantity",
                    "ms1": "PG.MS1Quantity",
                    "ms2": "PG.MS2Quantity",
                },
                "quality_filters": [
                    ("EG.IsDecoy", "==", False),
                ],
            },
            "gene": {
                "sample_col": "R.FileName",
                "protein_col": "PG.ProteinGroups",
                "gene_col": "PG.Genes",
                "quantity_modes": {
                    "default": "PG.Quantity",
                    "ms1": "PG.MS1Quantity",
                    "ms2": "PG.MS2Quantity",
                },
                "quality_filters": [
                    ("EG.IsDecoy", "==", False),
                ],
            },
        },
    },
    "diann": {
        "format": "long",
        "detect_columns": {"Protein.Group", "Genes", "Run"},
        "sample_metrics": {
            "sample_col": "Run",
            "proteingroups_col": "Protein.Group",
            "peptide_col": "Modified.Sequence",
        },
        "levels": {
            "precursor": {
                "sample_col": "Run",
                "protein_col": "Protein.Group",
                "gene_col": "Genes",
                "quantity_modes": {
                    "default": "Precursor.Quantity",
                    "ms1": "Ms1.Area",
                },
                "feature_id_candidates": ["Precursor.Id"],
                "feature_fallback_cols": ["Modified.Sequence", "Precursor.Charge"],
                "quality_filters": [
                    ("Lib.PG.Q.Value", "<=", 0.01),
                ],
            },
            "pg": {
                "sample_col": "Run",
                "protein_col": "Protein.Group",
                "gene_col": "Genes",
                "quantity_modes": {
                    "default": "PG.MaxLFQ",
                },
                "quality_filters": [],
            },
            "gene": {
                "sample_col": "File.Name",
                "protein_col": "Protein.Group",
                "gene_col": "Genes",
                "quantity_modes": {
                    "default": "Genes.MaxLFQ",
                },
                "quality_filters": [],
            },
        },
    },
    # FragPipe outputs combined_*.tsv as wide tables: one column per sample
    # (suffixed with " Intensity", " MaxLFQ Intensity", etc.). We treat the
    # column suffix as the "quantity_mode" so the public API is uniform with
    # the long-format engines.
    "fragpipe": {
        "format": "wide",
        "detect_columns": {"Protein", "Protein ID", "Entry Name"},
        "levels": {
            "precursor": {
                # combined_ion.tsv
                "protein_col": "Protein",
                "gene_col": "Gene",
                "extra_id_cols": ["Modified Sequence", "Charge"],
                "feature_fallback_cols": ["Modified Sequence", "Charge"],
                "quantity_modes": {
                    "default": " Intensity",
                    "intensity": " Intensity",
                },
                "quality_filters": [],
            },
            "pg": {
                # combined_protein.tsv
                "protein_col": "Protein",
                "gene_col": "Gene",
                "quantity_modes": {
                    "default": " MaxLFQ Intensity",
                    "maxlfq": " MaxLFQ Intensity",
                    "intensity": " Intensity",
                    "unique": " Unique Intensity",
                    "razor": " Razor Intensity",
                },
                "quality_filters": [],
            },
            "gene": {
                # combined_protein.tsv (aggregated by Gene downstream)
                "protein_col": "Protein",
                "gene_col": "Gene",
                "quantity_modes": {
                    "default": " MaxLFQ Intensity",
                    "maxlfq": " MaxLFQ Intensity",
                    "intensity": " Intensity",
                },
                "quality_filters": [],
            },
        },
    },
}


def _normalize_reports_path(reports_path):
    if isinstance(reports_path, (str, os.PathLike)):
        reports_path = [reports_path]
    reports_path = [str(path) for path in reports_path]
    if not reports_path:
        raise ValueError("reports_path is empty")
    return reports_path


def _read_columns(path):
    return set(pd.read_csv(path, sep="\t", nrows=0).columns)


def _ensure_columns_exist(columns, required_columns, context):
    missing_columns = [col for col in required_columns if col not in columns]
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns for {context}: {missing_str}")


def _detect_engine(reports_path, engine):
    if engine != "auto":
        if engine not in ENGINE_SCHEMAS:
            raise ValueError(f"Unsupported engine '{engine}'.")
        return engine

    columns = _read_columns(reports_path[0])
    detected = []
    for engine_name, engine_schema in ENGINE_SCHEMAS.items():
        if engine_schema["detect_columns"].issubset(columns):
            detected.append(engine_name)

    if len(detected) == 1:
        return detected[0]
    if not detected:
        raise ValueError("Could not auto-detect engine from input columns.")
    raise ValueError(f"Ambiguous engine detection: {detected}")


def _get_level_schema(engine, output_level):
    if output_level not in {"precursor", "pg", "gene"}:
        raise ValueError("output_level must be one of: 'precursor', 'pg', 'gene'")
    return ENGINE_SCHEMAS[engine]["levels"][output_level]


def _resolve_quantity_col(level_schema, quantity_mode):
    quantity_modes = level_schema["quantity_modes"]
    if quantity_mode not in quantity_modes:
        supported = ", ".join(sorted(quantity_modes))
        raise ValueError(
            f"Unsupported quantity_mode '{quantity_mode}'. Supported values: {supported}"
        )
    return quantity_modes[quantity_mode]


def _read_reports_with_usecols(reports_path, usecols, required_cols=None, context="report reading"):
    required_cols = required_cols or []
    report_list = []
    for report in reports_path:
        available_columns = _read_columns(report)
        _ensure_columns_exist(available_columns, required_cols, f"{context} ({report})")

        selected_usecols = [col for col in usecols if col in available_columns]
        report_list.append(pd.read_csv(report, sep="\t", usecols=selected_usecols))
    return pd.concat(report_list, ignore_index=True)


_COMPARATORS = {
    "==": lambda col, value: col == value,
    "!=": lambda col, value: col != value,
    "<=": lambda col, value: col.astype(type(value)) <= value,
    "<":  lambda col, value: col.astype(type(value)) < value,
    ">=": lambda col, value: col.astype(type(value)) >= value,
    ">":  lambda col, value: col.astype(type(value)) > value,
}


def _apply_filter_specs(df, filter_specs, context):
    if not filter_specs:
        return df

    _ensure_columns_exist(df.columns, [spec[0] for spec in filter_specs], context)
    mask = np.ones(len(df), dtype=bool)
    for column, comparator, value in filter_specs:
        if comparator not in _COMPARATORS:
            raise ValueError(f"Unsupported comparator '{comparator}' in {context}.")
        mask &= _COMPARATORS[comparator](df[column], value).to_numpy()
    return df.loc[mask].copy()


def _apply_contamination_filters(
    df,
    protein_col,
    gene_col,
    contaminant_pattern=r"CON_",
    gene_exclude_pattern=None,
):
    mask_valid = (
        df[protein_col].notna() & (df[protein_col] != "") &
        df[gene_col].notna() & (df[gene_col] != "")
    )
    filtered = df.loc[mask_valid].copy()

    if contaminant_pattern:
        mask_contaminant = filtered[protein_col].astype(str).str.contains(
            contaminant_pattern,
            na=False,
            regex=True,
        )
        filtered = filtered.loc[~mask_contaminant].copy()

    if gene_exclude_pattern:
        mask_gene_pattern = filtered[gene_col].astype(str).str.contains(
            gene_exclude_pattern,
            case=False,
            na=False,
            regex=True,
        )
        filtered = filtered.loc[~mask_gene_pattern].copy()

    return filtered

def _ordered_unique_nonempty(series):
    values = []
    seen = set()
    for value in series:
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        if not value_str or value_str in seen:
            continue
        seen.add(value_str)
        values.append(value_str)
    return values


def _collapse_unique_values(series, separator="|"):
    return separator.join(_ordered_unique_nonempty(series))


def _build_fallback_feature_id(df, fallback_cols):
    _ensure_columns_exist(df.columns, fallback_cols, "feature ID fallback construction")
    parts = []
    for col in fallback_cols:
        parts.append(df[col].fillna("nan").astype(str).str.strip())
    return pd.Series(
        ["".join(values) for values in zip(*parts)],
        index=df.index,
        dtype="object",
    )


def _resolve_feature_id(df, level_schema):
    feature_id_candidates = level_schema.get("feature_id_candidates", [])
    for candidate in feature_id_candidates:
        if candidate in df.columns:
            values = df[candidate].fillna("").astype(str).str.strip()
            if (values != "").any():
                return values.replace("", np.nan)

    fallback_cols = level_schema.get("feature_fallback_cols")
    if not fallback_cols:
        raise ValueError("No feature ID candidate or fallback columns configured.")
    return _build_fallback_feature_id(df, fallback_cols)


def _prepare_canonical_long_table(
    reports_path,
    engine,
    output_level,
    quantity_mode="default",
    sample_col=None,
    quality_filter=False,
    drop_contaminate=False,
    gene_exclude_pattern=None,
    contaminant_pattern=r"CON_",
):
    """Convert one or more reports into a canonical long table.

    Dispatches based on the engine ``format``:
    - ``"long"`` (Spectronaut, DIA-NN): the input file is a long-format report;
      sample IDs live in a column.
    - ``"wide"`` (FragPipe): the input file is a wide-format combined table;
      sample IDs are encoded as column-name suffixes.

    Parameters
    ----------
    reports_path : str or path-like or list
        One report path or multiple report paths to be concatenated.
    engine : {"auto", "spectronaut", "diann", "fragpipe"}
        Search engine type. ``"auto"`` detects the engine from input columns.
    output_level : {"precursor", "pg", "gene"}
        Target matrix level. This controls which columns and default quantity
        fields are used.
    quantity_mode : str, default="default"
        Quantification mode defined in the schema for the selected engine and
        output level. For long-format engines this resolves to a quantity
        column (e.g. ``"PG.Quantity"``); for wide-format engines it resolves
        to a sample-column suffix (e.g. ``" MaxLFQ Intensity"``).
    sample_col : str or None, default=None
        Optional override for the sample/run column for long-format inputs.
        Ignored for wide-format inputs.
    quality_filter : bool, default=False
        Whether to apply search-engine specific quality filters.
    drop_contaminate : bool, default=False
        Whether to remove rows with missing annotations or contaminant protein
        groups.
    gene_exclude_pattern : str or None, default=None
        Optional regex used during contamination filtering to exclude genes.
    contaminant_pattern : str or None, default="CON_"
        Regex used during contamination filtering to remove contaminant protein groups.

    Returns
    -------
    pandas.DataFrame
        Canonical long table with columns ``SampleID``, ``ProteinGroups``,
        ``Genes``, ``Quantity``, and ``FeatureID`` for precursor-level output.
    """
    reports_path = _normalize_reports_path(reports_path)
    engine = _detect_engine(reports_path, engine)
    engine_schema = ENGINE_SCHEMAS[engine]
    fmt = engine_schema.get("format", "long")
    level_schema = _get_level_schema(engine, output_level)

    if fmt == "long":
        canonical = _prepare_from_long_reports(
            reports_path=reports_path,
            engine=engine,
            output_level=output_level,
            level_schema=level_schema,
            quantity_mode=quantity_mode,
            sample_col_override=sample_col,
            quality_filter=quality_filter,
            drop_contaminate=drop_contaminate,
            gene_exclude_pattern=gene_exclude_pattern,
            contaminant_pattern=contaminant_pattern,
        )
    elif fmt == "wide":
        canonical = _prepare_from_wide_reports(
            reports_path=reports_path,
            engine=engine,
            output_level=output_level,
            level_schema=level_schema,
            quantity_mode=quantity_mode,
            quality_filter=quality_filter,
            drop_contaminate=drop_contaminate,
            gene_exclude_pattern=gene_exclude_pattern,
            contaminant_pattern=contaminant_pattern,
        )
    else:
        raise ValueError(f"Unsupported engine format '{fmt}' for engine '{engine}'.")

    if canonical.empty:
        raise ValueError("No valid rows found after filtering and preprocessing.")
    return canonical


def _prepare_from_long_reports(
    *,
    reports_path,
    engine,
    output_level,
    level_schema,
    quantity_mode,
    sample_col_override,
    quality_filter,
    drop_contaminate,
    gene_exclude_pattern,
    contaminant_pattern,
):
    sample_col = sample_col_override or level_schema["sample_col"]
    quantity_col = _resolve_quantity_col(level_schema, quantity_mode)
    protein_col = level_schema["protein_col"]
    gene_col = level_schema["gene_col"]

    usecols = [sample_col, protein_col, gene_col, quantity_col]
    usecols.extend(level_schema.get("feature_id_candidates", []))
    usecols.extend(level_schema.get("feature_fallback_cols", []))
    if quality_filter:
        usecols.extend([spec[0] for spec in level_schema.get("quality_filters", [])])
    usecols = list(dict.fromkeys(usecols))

    required_cols = [sample_col, protein_col, gene_col, quantity_col]
    if quality_filter:
        required_cols.extend([spec[0] for spec in level_schema.get("quality_filters", [])])
    required_cols = list(dict.fromkeys(required_cols))

    df = _read_reports_with_usecols(
        reports_path,
        usecols,
        required_cols=required_cols,
        context=f"{engine} {output_level}",
    )

    if quality_filter:
        df = _apply_filter_specs(
            df,
            level_schema.get("quality_filters", []),
            f"{engine} {output_level} quality filtering",
        )

    if drop_contaminate:
        df = _apply_contamination_filters(
            df,
            protein_col=protein_col,
            gene_col=gene_col,
            contaminant_pattern=contaminant_pattern,
            gene_exclude_pattern=gene_exclude_pattern,
        )

    quantity = pd.to_numeric(df[quantity_col], errors="coerce")
    valid_mask = (
        df[sample_col].notna()
        & (df[sample_col] != "")
        & quantity.notna()
    )

    feature_id = None
    if output_level == "precursor":
        feature_id = _resolve_feature_id(df, level_schema).astype(str)
        valid_mask &= feature_id.notna() & (feature_id != "")

    valid_mask = valid_mask.to_numpy()
    df = df.loc[valid_mask]
    quantity = quantity.loc[valid_mask]

    canonical = pd.DataFrame(
        {
            "SampleID": df[sample_col].astype(str).to_numpy(),
            "ProteinGroups": df[protein_col].fillna("").astype(str).to_numpy(),
            "Genes": df[gene_col].fillna("").astype(str).to_numpy(),
            "Quantity": quantity.to_numpy(),
        }
    )

    if feature_id is not None:
        canonical["FeatureID"] = feature_id.to_numpy()[valid_mask]

    return canonical


def _select_wide_sample_columns(df_columns, sample_suffix, level_schema):
    """Return columns ending in ``sample_suffix`` while excluding columns whose
    name happens to end with a more specific suffix declared in the same level.

    Without this, picking ``" Intensity"`` for FragPipe would also catch
    ``" MaxLFQ Intensity"`` / ``" Unique Intensity"`` etc.
    """
    other_suffixes = [
        suffix for suffix in level_schema["quantity_modes"].values()
        if suffix != sample_suffix and suffix.endswith(sample_suffix)
    ]
    sample_cols = []
    for col in df_columns:
        if not isinstance(col, str) or not col.endswith(sample_suffix):
            continue
        if len(col) <= len(sample_suffix):
            continue
        if any(col.endswith(longer) for longer in other_suffixes):
            continue
        sample_cols.append(col)
    return sample_cols


def _prepare_from_wide_reports(
    *,
    reports_path,
    engine,
    output_level,
    level_schema,
    quantity_mode,
    quality_filter,
    drop_contaminate,
    gene_exclude_pattern,
    contaminant_pattern,
):
    sample_suffix = _resolve_quantity_col(level_schema, quantity_mode)
    protein_col = level_schema["protein_col"]
    gene_col = level_schema["gene_col"]
    extra_id_cols = list(level_schema.get("extra_id_cols", []))

    required_meta = list(dict.fromkeys([protein_col, gene_col, *extra_id_cols]))

    canonical_parts = []
    for report in reports_path:
        df = pd.read_csv(report, sep="\t")
        context = f"{engine} {output_level} ({report})"
        _ensure_columns_exist(df.columns, required_meta, context)

        sample_cols = _select_wide_sample_columns(df.columns, sample_suffix, level_schema)
        if not sample_cols:
            raise ValueError(
                f"No sample columns ending with '{sample_suffix}' were found in {report}."
            )

        if quality_filter:
            df = _apply_filter_specs(
                df,
                level_schema.get("quality_filters", []),
                f"{engine} {output_level} quality filtering",
            )

        if drop_contaminate:
            df = _apply_contamination_filters(
                df,
                protein_col=protein_col,
                gene_col=gene_col,
                contaminant_pattern=contaminant_pattern,
                gene_exclude_pattern=gene_exclude_pattern,
            )

        id_vars = required_meta
        long = df[id_vars + sample_cols].melt(
            id_vars=id_vars,
            value_vars=sample_cols,
            var_name="_sample_raw",
            value_name="Quantity",
        )
        long["SampleID"] = long["_sample_raw"].str.slice(0, -len(sample_suffix)).str.rstrip()
        long["Quantity"] = pd.to_numeric(long["Quantity"], errors="coerce")

        valid = (
            long["Quantity"].notna()
            & (long["Quantity"] > 0)
            & (long["SampleID"] != "")
        )

        feature_id = None
        if output_level == "precursor":
            feature_id = _resolve_feature_id(long, level_schema).astype(str)
            valid &= feature_id.notna() & (feature_id != "")

        valid = valid.to_numpy()
        long = long.loc[valid]

        canonical = pd.DataFrame(
            {
                "SampleID": long["SampleID"].astype(str).to_numpy(),
                "ProteinGroups": long[protein_col].fillna("").astype(str).to_numpy(),
                "Genes": long[gene_col].fillna("").astype(str).to_numpy(),
                "Quantity": long["Quantity"].to_numpy(),
            }
        )
        if feature_id is not None:
            canonical["FeatureID"] = feature_id.to_numpy()[valid]

        canonical_parts.append(canonical)

    return pd.concat(canonical_parts, ignore_index=True) if canonical_parts else pd.DataFrame(
        columns=["SampleID", "ProteinGroups", "Genes", "Quantity"]
    )


def _build_precursor_matrix_from_canonical(canonical, aggfunc="first"):
    sample_order = _ordered_unique_nonempty(canonical["SampleID"])
    feature_pairs = canonical[["ProteinGroups", "FeatureID"]].drop_duplicates()
    feature_order = pd.MultiIndex.from_frame(feature_pairs)

    mapping = (
        canonical.groupby(["ProteinGroups", "FeatureID"], sort=False)
        .agg(
            Genes=("Genes", _collapse_unique_values),
        )
        .reset_index()
    )

    wide = canonical.pivot_table(
        index=["ProteinGroups", "FeatureID"],
        columns="SampleID",
        values="Quantity",
        aggfunc=aggfunc,
        sort=False,
    )
    wide = wide.reindex(index=feature_order)
    wide = wide.reindex(columns=sample_order)
    wide = wide.reset_index()

    result = mapping.merge(wide, on=["ProteinGroups", "FeatureID"], how="right")
    sample_cols = [col for col in result.columns if col not in {"ProteinGroups", "Genes", "FeatureID"}]
    return result[["ProteinGroups", "Genes", "FeatureID", *sample_cols]]


def _build_pg_matrix_from_canonical(canonical, aggfunc="first"):
    sample_order = _ordered_unique_nonempty(canonical["SampleID"])
    feature_order = _ordered_unique_nonempty(canonical["ProteinGroups"])

    mapping = (
        canonical.groupby("ProteinGroups", sort=False)
        .agg(Genes=("Genes", _collapse_unique_values))
        .reset_index()
    )

    wide = canonical.pivot_table(
        index="ProteinGroups",
        columns="SampleID",
        values="Quantity",
        aggfunc=aggfunc,
        sort=False,
    )
    wide = wide.reindex(index=feature_order)
    wide = wide.reindex(columns=sample_order)
    wide = wide.reset_index()

    result = wide.merge(mapping, on="ProteinGroups", how="left")
    sample_cols = [col for col in result.columns if col not in {"ProteinGroups", "Genes"}]
    return result[["ProteinGroups", "Genes", *sample_cols]]


def _build_gene_matrix_from_canonical(canonical, aggfunc="sum"):
    sample_order = _ordered_unique_nonempty(canonical["SampleID"])
    feature_order = _ordered_unique_nonempty(canonical["Genes"])

    mapping = (
        canonical.groupby("Genes", sort=False)
        .agg(ProteinGroups=("ProteinGroups", _collapse_unique_values))
        .reset_index()
    )

    wide = canonical.pivot_table(
        index="Genes",
        columns="SampleID",
        values="Quantity",
        aggfunc=aggfunc,
        sort=False,
    )
    wide = wide.reindex(index=feature_order)
    wide = wide.reindex(columns=sample_order)
    wide = wide.reset_index()

    result = wide.merge(mapping, on="Genes", how="left")
    sample_cols = [col for col in result.columns if col not in {"Genes", "ProteinGroups"}]
    return result[["Genes", "ProteinGroups", *sample_cols]]


def read_precursor_matrix(
    reports_path,
    engine="auto",
    sample_col=None,
    quantity_mode="default",
    quality_filter=False,
    drop_contaminate=False,
    aggfunc="first",
    gene_exclude_pattern=None,
    contaminant_pattern=r"^CON_",
):
    """Read report(s) and construct a precursor-level wide matrix.

    Parameters
    ----------
    reports_path : str or path-like or list
        One report path or multiple report paths to be concatenated.
    engine : {"auto", "spectronaut", "diann", "fragpipe"}, default="auto"
        Search engine type. ``"auto"`` detects the engine from input columns.
    sample_col : str or None, default=None
        Optional override for the sample/run column. If ``None``, the engine
        default is used.
    quantity_mode : str, default="default"
        Quantification field to use for precursor intensity. Supported values
        depend on the engine schema.
    quality_filter : bool, default=False
        Whether to apply search-engine specific quality filters.
    drop_contaminate : bool, default=False
        Whether to remove rows with missing annotations or contaminant protein
        groups.
    aggfunc : str or callable, default="first"
        Aggregation used when the same ``FeatureID`` and sample appear more
        than once in the input.
    gene_exclude_pattern : str or None, default=None
        Optional regular expression used during contamination filtering to exclude
        gene annotations matching the pattern.
    contaminant_pattern : str or None, default="^CON_"
        Regular expression used during contamination filtering to remove contaminant
        protein groups. Pass ``None`` to disable this filter.

    Returns
    -------
    pandas.DataFrame
        Wide matrix with columns ``ProteinGroups``, ``Genes``, ``FeatureID``,
        followed by one column per sample.
    """
    canonical = _prepare_canonical_long_table(
        reports_path,
        engine=engine,
        output_level="precursor",
        quantity_mode=quantity_mode,
        sample_col=sample_col,
        quality_filter=quality_filter,
        drop_contaminate=drop_contaminate,
        gene_exclude_pattern=gene_exclude_pattern,
        contaminant_pattern=contaminant_pattern,
    )
    return _build_precursor_matrix_from_canonical(canonical, aggfunc=aggfunc)


def read_pg_matrix(
    reports_path,
    engine="auto",
    sample_col=None,
    quantity_mode="default",
    quality_filter=False,
    drop_contaminate=False,
    aggfunc="first",
    gene_exclude_pattern=None,
    contaminant_pattern=r"^CON_",
):
    """Read report(s) and construct a protein-group-level wide matrix.

    Parameters
    ----------
    reports_path : str or path-like or list
        One report path or multiple report paths to be concatenated.
    engine : {"auto", "spectronaut", "diann", "fragpipe"}, default="auto"
        Search engine type. ``"auto"`` detects the engine from input columns.
    sample_col : str or None, default=None
        Optional override for the sample/run column. If ``None``, the engine
        default is used.
    quantity_mode : str, default="default"
        Quantification field to use for protein-group intensity. Supported
        values depend on the engine schema.
    quality_filter : bool, default=False
        Whether to apply search-engine specific quality filters.
    drop_contaminate : bool, default=False
        Whether to remove rows with missing annotations or contaminant protein
        groups.
    aggfunc : str or callable, default="first"
        Aggregation used when the same protein group and sample appear more
        than once in the input.
    gene_exclude_pattern : str or None, default=None
        Optional regular expression used during contamination filtering to exclude
        gene annotations matching the pattern.
    contaminant_pattern : str or None, default="^CON_"
        Regular expression used during contamination filtering to remove contaminant
        protein groups. Pass ``None`` to disable this filter.

    Returns
    -------
    pandas.DataFrame
        Wide matrix with columns ``ProteinGroups``, ``Genes``, followed by one
        column per sample.
    """
    canonical = _prepare_canonical_long_table(
        reports_path,
        engine=engine,
        output_level="pg",
        quantity_mode=quantity_mode,
        sample_col=sample_col,
        quality_filter=quality_filter,
        drop_contaminate=drop_contaminate,
        gene_exclude_pattern=gene_exclude_pattern,
        contaminant_pattern=contaminant_pattern,
    )
    return _build_pg_matrix_from_canonical(canonical, aggfunc=aggfunc)


def read_gene_matrix(
    reports_path,
    engine="auto",
    sample_col=None,
    quantity_mode="default",
    quality_filter=False,
    drop_contaminate=False,
    aggfunc="sum",
    gene_exclude_pattern=None,
    contaminant_pattern=r"^CON_",
):
    """Read report(s) and construct a gene-level wide matrix.

    This function does not run ``split_multigene_rows()`` automatically. If the
    input gene annotation contains multi-gene entries such as ``"GeneA;GeneB"``,
    those values are preserved by default.

    Parameters
    ----------
    reports_path : str or path-like or list
        One report path or multiple report paths to be concatenated.
    engine : {"auto", "spectronaut", "diann", "fragpipe"}, default="auto"
        Search engine type. ``"auto"`` detects the engine from input columns.
    sample_col : str or None, default=None
        Optional override for the sample/run column. If ``None``, the engine
        default is used.
    quantity_mode : str, default="default"
        Quantification field to use for gene-level intensity. Supported values
        depend on the engine schema.
    quality_filter : bool, default=False
        Whether to apply search-engine specific quality filters.
    drop_contaminate : bool, default=False
        Whether to remove rows with missing annotations or contaminant protein
        groups.
    aggfunc : str or callable, default="sum"
        Aggregation used when the same gene annotation and sample appear more
        than once in the input.
    gene_exclude_pattern : str or None, default=None
        Optional regular expression used during contamination filtering to exclude
        gene annotations matching the pattern.
    contaminant_pattern : str or None, default="^CON_"
        Regular expression used during contamination filtering to remove contaminant
        protein groups. Pass ``None`` to disable this filter.

    Returns
    -------
    pandas.DataFrame
        Wide matrix with columns ``Genes``, ``ProteinGroups``, followed by one
        column per sample.
    """
    canonical = _prepare_canonical_long_table(
        reports_path,
        engine=engine,
        output_level="gene",
        quantity_mode=quantity_mode,
        sample_col=sample_col,
        quality_filter=quality_filter,
        drop_contaminate=drop_contaminate,
        gene_exclude_pattern=gene_exclude_pattern,
        contaminant_pattern=contaminant_pattern,
    )
    return _build_gene_matrix_from_canonical(canonical, aggfunc=aggfunc)


def calculate_sample_metrics(
    reports_path,
    engine="auto",
    *,
    filename_col=None,
    proteingroups_col=None,
    peptide_col=None,
    quantity_mode="default",
):
    """Calculate simple per-sample identification metrics.

    For long-format engines (Spectronaut, DIA-NN) the metrics are computed
    from per-row identifications: precursor rows per sample, unique peptide
    sequences per sample, and unique protein groups per sample.

    For wide-format engines (FragPipe) the input table already has one column
    per sample. The metric reported depends on which combined_*.tsv file is
    passed: ``combined_protein.tsv`` yields ``ProteinGroups_identified``,
    ``combined_modified_peptide.tsv`` yields ``Peptides_identified``, and
    ``combined_ion.tsv`` yields ``Precursors_identified``. The remaining
    columns are left as ``NaN`` for that file type.

    Parameters
    ----------
    reports_path : str or path-like or list
        One report path or multiple report paths to be concatenated.
    engine : {"auto", "spectronaut", "diann", "fragpipe"}, default="auto"
        Search engine type. ``"auto"`` detects the engine from input columns.
    filename_col, proteingroups_col, peptide_col : str, optional
        Optional overrides for the long-format column names. If ``None``, the
        defaults from the engine schema are used.
    quantity_mode : str, default="default"
        Quantification mode used for wide-format engines. Determines which
        sample-column suffix identifies the per-sample data columns
        (e.g. ``" MaxLFQ Intensity"``).

    Returns
    -------
    pandas.DataFrame
        Table with one row per sample and the columns ``sample``,
        ``Precursors_identified``, ``Peptides_identified``, and
        ``ProteinGroups_identified``.
    """
    reports_path = _normalize_reports_path(reports_path)
    engine = _detect_engine(reports_path, engine)
    engine_schema = ENGINE_SCHEMAS[engine]
    fmt = engine_schema.get("format", "long")

    if fmt == "long":
        return _calculate_sample_metrics_long(
            reports_path,
            engine_schema,
            filename_col=filename_col,
            proteingroups_col=proteingroups_col,
            peptide_col=peptide_col,
        )
    if fmt == "wide":
        return _calculate_sample_metrics_wide(
            reports_path,
            engine_schema,
            quantity_mode=quantity_mode,
        )
    raise ValueError(f"Unsupported engine format '{fmt}' for engine '{engine}'.")


def _calculate_sample_metrics_long(
    reports_path,
    engine_schema,
    *,
    filename_col,
    proteingroups_col,
    peptide_col,
):
    schema = engine_schema.get("sample_metrics")
    if schema is None:
        raise ValueError(
            "The selected engine does not declare sample_metrics columns; "
            "pass filename_col / proteingroups_col / peptide_col explicitly."
        )
    filename_col = filename_col or schema["sample_col"]
    proteingroups_col = proteingroups_col or schema["proteingroups_col"]
    peptide_col = peptide_col or schema["peptide_col"]

    usecols = [filename_col, proteingroups_col, peptide_col]
    df = _read_reports_with_usecols(
        reports_path,
        usecols,
        required_cols=usecols,
        context="sample metrics",
    )
    mask_valid = (
        df[filename_col].notna() & (df[filename_col] != "")
        & df[proteingroups_col].notna() & (df[proteingroups_col] != "")
        & df[peptide_col].notna() & (df[peptide_col] != "")
    )
    df = df.loc[mask_valid]

    if df.empty:
        raise ValueError("No valid rows found after filtering empty metric columns.")

    sample_order = df[filename_col].dropna().drop_duplicates().tolist()

    metrics = (
        df.groupby(filename_col, sort=False)
        .agg(
            Precursors_identified=(filename_col, "size"),
            Peptides_identified=(peptide_col, "nunique"),
            ProteinGroups_identified=(proteingroups_col, "nunique"),
        )
        .reindex(sample_order)
        .reset_index()
        .rename(columns={filename_col: "sample"})
    )

    return metrics


def _calculate_sample_metrics_wide(reports_path, engine_schema, *, quantity_mode):
    # For a wide table we cannot infer all three metrics from a single file —
    # FragPipe needs combined_protein.tsv, combined_modified_peptide.tsv, and
    # combined_ion.tsv respectively. We auto-detect which file was given and
    # populate the corresponding metric column; the others stay NaN.
    levels = engine_schema["levels"]

    def _resolve_metric_col(level_name, df_columns):
        level_schema = levels.get(level_name)
        if level_schema is None:
            return None
        suffix = level_schema["quantity_modes"].get(quantity_mode)
        if suffix is None:
            return None
        extras = level_schema.get("extra_id_cols", [])
        if not all(col in df_columns for col in extras):
            return None
        sample_cols = _select_wide_sample_columns(df_columns, suffix, level_schema)
        if not sample_cols:
            return None
        return suffix, sample_cols

    metric_column_for_level = {
        "precursor": "Precursors_identified",
        "pg": "ProteinGroups_identified",
        "gene": "ProteinGroups_identified",
    }

    per_sample = {}
    for report in reports_path:
        df = pd.read_csv(report, sep="\t")
        chosen = None
        # combined_ion.tsv has the precursor markers (Charge etc.); pick it
        # first so that combined_protein.tsv doesn't accidentally win.
        for level in ("precursor", "pg", "gene"):
            resolved = _resolve_metric_col(level, df.columns)
            if resolved is None:
                continue
            chosen = (level, *resolved)
            break
        if chosen is None:
            raise ValueError(
                f"Could not identify a FragPipe-style sample column in {report}. "
                "Make sure the file is a combined_*.tsv output."
            )
        level, suffix, sample_cols = chosen
        metric_name = metric_column_for_level[level]

        for col in sample_cols:
            sample_id = col[:-len(suffix)].rstrip()
            values = pd.to_numeric(df[col], errors="coerce")
            count = int((values.notna() & (values > 0)).sum())
            entry = per_sample.setdefault(sample_id, {})
            entry[metric_name] = entry.get(metric_name, 0) + count

    if not per_sample:
        raise ValueError("No sample columns were detected in the wide-format input.")

    rows = []
    for sample_id, counts in per_sample.items():
        rows.append({
            "sample": sample_id,
            "Precursors_identified": counts.get("Precursors_identified", np.nan),
            "Peptides_identified": counts.get("Peptides_identified", np.nan),
            "ProteinGroups_identified": counts.get("ProteinGroups_identified", np.nan),
        })
    return pd.DataFrame(rows)


def split_multigene_rows(
    gene_matrix,
    genes_col="Genes",
    split_sep=";",
    split_aggfunc="max",
):
    """Split multi-gene rows into a single-gene matrix.

    This function is intentionally kept separate from the default reading path
    because the split logic is a user-chosen post-processing strategy rather
    than a mandatory package behavior.

    Parameters
    ----------
    gene_matrix : pandas.DataFrame
        Gene-level wide matrix produced by ``read_gene_matrix()`` or another
        compatible table. It must contain one gene annotation column and
        numeric sample columns.
    genes_col : str, default="Genes"
        Name of the gene annotation column.
    split_sep : str, default=";"
        Separator used to split multi-gene entries such as ``"GeneA;GeneB"``.
    split_aggfunc : str or callable, default="max"
        Aggregation used after expansion when one gene appears in multiple
        rows.

    Returns
    -------
    pandas.DataFrame
        Single-gene wide matrix with ``Genes`` as the feature column and
        numeric sample columns.
    """
    if genes_col in gene_matrix.columns:
        mat = gene_matrix.copy()
    else:
        if gene_matrix.index.name != genes_col:
            gene_matrix = gene_matrix.copy()
            gene_matrix.index.name = genes_col
        mat = gene_matrix.reset_index()

    if genes_col not in mat.columns:
        raise ValueError(f"'{genes_col}' column was not found in input table")

    numeric_cols = mat.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Input table must contain numeric sample columns")

    mat = mat.copy()
    mat[genes_col] = mat[genes_col].fillna("").astype(str).str.strip()
    mat["gene_list"] = mat[genes_col].apply(
        lambda gene_value: [
            gene.strip()
            for gene in dict.fromkeys(str(gene_value).split(split_sep))
            if gene.strip()
        ]
    )
    mat = mat.loc[mat["gene_list"].map(bool)].reset_index(drop=True)

    gene_to_row_count = {}
    for gene_set in mat["gene_list"]:
        for gene in gene_set:
            gene_to_row_count[gene] = gene_to_row_count.get(gene, 0) + 1

    mat["label"] = mat["gene_list"].apply(
        lambda genes: "duplicate"
        if any(gene_to_row_count[gene] > 1 for gene in genes)
        else "unique"
    )

    expanded = (
        mat
        .explode("gene_list")
        .drop(columns=[genes_col])
        .rename(columns={"gene_list": "Genes"})
    )
    expanded["Genes"] = expanded["Genes"].fillna("").astype(str).str.strip()
    expanded = expanded.loc[expanded["Genes"] != ""].reset_index(drop=True)

    expanded["type"] = expanded.groupby("Genes")["Genes"].transform("size").gt(1).astype(int)
    filtered = expanded.loc[
        ~((expanded["label"] == "duplicate") & (expanded["type"] == 0))
    ].copy()

    split_matrix = (
        filtered.groupby("Genes", sort=False)[numeric_cols]
        .agg(split_aggfunc)
        .reset_index()
    )

    return split_matrix
