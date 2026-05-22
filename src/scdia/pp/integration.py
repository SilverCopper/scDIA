import json
import subprocess
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import scanpy as sc
from scanpy import AnnData

from .normalize import normalize

PYTHON_METHODS = {
    "Combat",
    "Harmony",
    "Scanorama",
    "BBKNN",
    "scVI",
    "scVI_normal",
    "scVI_nb",
}
R_METHODS = {"CCA", "RPCA", "FastMNN", "Liger"}
ALL_METHODS = [
    "Combat",
    "Harmony",
    "Scanorama",
    "BBKNN",
    "scVI_normal",
    "scVI_nb",
    "CCA",
    "RPCA",
    "FastMNN",
    "Liger",
]


def _as_dict(value):
    return {} if value is None else dict(value)


def _copy_matrix(matrix):
    return matrix.copy() if hasattr(matrix, "copy") else matrix


def _resolve_methods(methods):
    if methods == "all":
        return list(ALL_METHODS)
    if isinstance(methods, str):
        return [methods]
    return list(methods)


def _validate_layer(adata: AnnData, layer: str | None, *, name: str):
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"{name} '{layer}' was not found in adata.layers.")


def _ensure_default_raw_layer(adata: AnnData, raw_layer: str | None):
    if raw_layer == "raw_matrix" and raw_layer not in adata.layers:
        adata.layers[raw_layer] = _copy_matrix(adata.X)


def _get_hvg_source_layer(raw_layer: str | None, hvg_flavor: str, normalized_layer: str):
    if hvg_flavor == "seurat":
        return normalized_layer
    if hvg_flavor == "seurat_v3":
        return raw_layer
    raise ValueError("hvg_flavor must be either 'seurat' or 'seurat_v3'.")


def _use_highly_variable(adata: AnnData):
    return "highly_variable" in adata.var and bool(adata.var["highly_variable"].any())


def preprocess_for_integration(
    adata: AnnData,
    raw_layer: str | None = "raw_matrix",
    normalized_layer: str = "log2_total_normalized",
    batch_key: str | None = None,
    normalization_method: str = "total",
    zero_as_missing: bool = True,
    log1p: bool = True,
    log1p_base: float | None = None,
    calculate_hvg: bool = True,
    hvg_flavor: Literal["seurat", "seurat_v3"] = "seurat",
    n_top_genes: int | None = 2000,
    scale: bool = False,
    n_pcs: int = 20,
    n_neighbors: int = 20,
    inplace: bool = True,
    normalization_kwargs: dict | None = None,
    hvg_kwargs: dict | None = None,
    scale_kwargs: dict | None = None,
    pca_kwargs: dict | None = None,
    neighbors_kwargs: dict | None = None,
) -> AnnData | None:
    """Prepare an AnnData object for batch correction."""
    if normalized_layer is None:
        raise ValueError("normalized_layer must be a layer name, not None.")
    if n_pcs <= 0:
        raise ValueError("n_pcs must be positive.")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    if n_top_genes is not None and n_top_genes <= 0:
        raise ValueError("n_top_genes must be positive when provided.")
    if batch_key is not None and batch_key not in adata.obs:
        raise ValueError(f"batch_key '{batch_key}' was not found in adata.obs.")

    ad = adata if inplace else adata.copy()
    _ensure_default_raw_layer(ad, raw_layer)
    _validate_layer(ad, raw_layer, name="raw_layer")

    normalization_kwargs = _as_dict(normalization_kwargs)
    hvg_kwargs = _as_dict(hvg_kwargs)
    scale_kwargs = _as_dict(scale_kwargs)
    pca_kwargs = _as_dict(pca_kwargs)
    neighbors_kwargs = _as_dict(neighbors_kwargs)
    if "log1p_base" in pca_kwargs:
        old_log1p_base = pca_kwargs.pop("log1p_base")
        if log1p_base is None:
            log1p_base = old_log1p_base
        warnings.warn(
            "Passing log1p_base through pca_kwargs is deprecated; use the "
            "top-level log1p_base parameter instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    normalize(
        ad,
        method=normalization_method,
        layer=raw_layer,
        output_layer=normalized_layer,
        inplace=True,
        zero_as_missing=zero_as_missing,
        **normalization_kwargs,
    )

    if log1p:
        sc.pp.log1p(ad, layer=normalized_layer, base=log1p_base)

    if calculate_hvg:
        hvg_source_layer = _get_hvg_source_layer(raw_layer, hvg_flavor, normalized_layer)
        hvg_call_kwargs = {
            "flavor": hvg_flavor,
            **hvg_kwargs,
        }
        if n_top_genes is not None:
            hvg_call_kwargs["n_top_genes"] = n_top_genes
        if hvg_source_layer is not None:
            hvg_call_kwargs["layer"] = hvg_source_layer
        sc.pp.highly_variable_genes(ad, **hvg_call_kwargs)

    use_highly_variable = _use_highly_variable(ad)

    ad.X = _copy_matrix(ad.layers[normalized_layer])
    if scale:
        scale_call_kwargs = {"max_value": 10, **scale_kwargs}
        sc.pp.scale(ad, **scale_call_kwargs)

    pca_call_kwargs = {
        "n_comps": n_pcs,
        "svd_solver": "arpack",
        "use_highly_variable": use_highly_variable,
        **pca_kwargs,
    }
    sc.tl.pca(ad, **pca_call_kwargs)

    neighbors_call_kwargs = {
        "n_neighbors": n_neighbors,
        "n_pcs": n_pcs,
        "use_rep": "X_pca",
        **neighbors_kwargs,
    }
    sc.pp.neighbors(ad, **neighbors_call_kwargs)

    if not inplace:
        return ad
    return None


def _merge_method_params(method_params: dict | None):
    defaults = {
        "Harmony": {"max_iter_harmony": 20},
        "BBKNN": {"neighbors_within_batch": 3},
        "Scanorama": {},
        "Combat": {},
        "CCA": {},
        "RPCA": {},
        "FastMNN": {},
        "Liger": {"k": 20},
        "scVI_normal": {
            "layer": "normalized",
            "model": {
                "n_latent": 20,
                "n_layers": 1,
                "gene_likelihood": "normal",
                "dispersion": "gene",
            },
            "train": {"max_epochs": 600, "early_stopping": True, "batch_size": 1024},
        },
        "scVI_nb": {
            "layer": "raw",
            "model": {
                "n_latent": 20,
                "n_layers": 1,
                "gene_likelihood": "nb",
                "dispersion": "gene",
            },
            "train": {"max_epochs": 600, "early_stopping": True, "batch_size": 1024},
        },
    }
    if method_params is None:
        return defaults

    merged = deepcopy(defaults)
    for method, params in method_params.items():
        params = dict(params)
        if method not in merged:
            merged[method] = params
            continue
        for key, value in params.items():
            if isinstance(value, dict) and isinstance(merged[method].get(key), dict):
                merged[method][key].update(value)
            else:
                merged[method][key] = value
    return merged


def _get_r_script_path():
    return Path(__file__).resolve().parent / "batch_correction.R"


def _write_r_input_h5ad(
    adata: AnnData,
    path: Path,
    *,
    layer: str | None,
):
    ad_r = adata.copy()
    if layer is not None:
        if layer not in ad_r.layers:
            raise ValueError(f"R input layer '{layer}' was not found in adata.layers.")
        ad_r.X = _copy_matrix(ad_r.layers[layer])
    ad_r.write_h5ad(path)


def _read_embedding_csv(path: Path, obs_names):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    obs_names = [str(cell) for cell in obs_names]
    missing = [cell for cell in obs_names if cell not in df.index]
    if missing:
        preview = missing[:10]
        raise ValueError(
            f"R embedding file '{path}' is missing {len(missing)} cells; "
            f"first missing cells: {preview}."
        )
    df = df.loc[obs_names]
    return df.to_numpy()


def _run_r_integrations(
    adata: AnnData,
    *,
    methods: Sequence[str],
    batch_key: str,
    layer: str | None,
    n_top_genes: int | None,
    n_pcs: int,
    params: dict,
    rscript_path: str | None = None,
    r_executable: str = "Rscript",
    temp_dir: str | None = None,
    keep_temp: bool = False,
):
    if not methods:
        return {}

    script_path = Path(rscript_path) if rscript_path is not None else _get_r_script_path()
    if not script_path.exists():
        raise FileNotFoundError(f"R batch correction script was not found: {script_path}")

    statuses = {}
    tmp_context = tempfile.TemporaryDirectory(prefix="scdia_r_batch_", dir=temp_dir)
    work_dir = Path(tmp_context.name)
    try:
        input_h5ad = work_dir / "input.h5ad"
        output_dir = work_dir / "embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_r_input_h5ad(adata, input_h5ad, layer=layer)

        r_params = {
            method: params.get(method, {})
            for method in methods
        }
        params_json = work_dir / "r_method_params.json"
        params_json.write_text(json.dumps(r_params, ensure_ascii=False, default=str))

        command = [
            r_executable,
            str(script_path),
            str(input_h5ad),
            str(output_dir),
            batch_key,
            ",".join(methods),
            str(n_top_genes if n_top_genes is not None else 0),
            str(n_pcs),
            str(params.get("Liger", {}).get("k", 20)),
            str(params_json),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "R batch correction failed with exit code "
                f"{completed.returncode}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        for method in methods:
            embedding_path = output_dir / f"{method}.csv"
            if embedding_path.exists():
                adata.obsm[method] = _read_embedding_csv(embedding_path, adata.obs_names)
                statuses[method] = {"status": "success", "backend": "R"}
            else:
                statuses[method] = {
                    "status": "failed",
                    "error": f"Expected R output file was not created: {embedding_path}",
                    "backend": "R",
                }
        return statuses
    finally:
        if keep_temp:
            warnings.warn(f"R batch correction temporary directory kept at: {work_dir}")
        else:
            tmp_context.cleanup()


def _resolve_scvi_layer(layer_spec, raw_layer: str | None, normalized_layer: str):
    if layer_spec == "raw":
        return raw_layer
    if layer_spec == "normalized":
        return normalized_layer
    return layer_spec


def _run_scvi(
    adata: AnnData,
    *,
    method: str,
    batch_key: str,
    layer: str | None,
    model_params: dict,
    train_params: dict,
    save_path: str | None = None,
):
    import scvi

    ad_scvi = adata.copy()
    if _use_highly_variable(ad_scvi):
        ad_scvi = ad_scvi[:, ad_scvi.var["highly_variable"].to_numpy()].copy()

    if layer is not None and layer not in ad_scvi.layers:
        raise ValueError(f"{method} layer '{layer}' was not found in adata.layers.")

    scvi.model.SCVI.setup_anndata(ad_scvi, layer=layer, batch_key=batch_key)
    model = scvi.model.SCVI(ad_scvi, **model_params)
    model.train(**train_params)
    if save_path is not None:
        model.save(save_path, overwrite=True)
    adata.obsm[method] = model.get_latent_representation()


def integrate(
    adata: AnnData,
    batch_key: str,
    raw_layer: str | None = "raw_matrix",
    normalized_layer: str = "log2_total_normalized",
    methods: str | Sequence[str] = "all",
    preprocess: bool = True,
    preprocess_kwargs: dict | None = None,
    method_params: dict | None = None,
    run_r: bool = True,
    rscript_path: str | None = None,
    r_executable: str = "Rscript",
    temp_dir: str | None = None,
    keep_temp: bool = False,
    inplace: bool = False,
    fail_fast: bool = True,
) -> AnnData | None:
    """Run one-step batch correction using preprocessed PCA embeddings."""
    if batch_key not in adata.obs:
        raise ValueError(f"batch_key '{batch_key}' was not found in adata.obs.")

    selected_methods = _resolve_methods(methods)
    supported_methods = PYTHON_METHODS | R_METHODS
    unsupported = sorted(set(selected_methods) - supported_methods)
    if unsupported:
        raise ValueError(
            f"Unsupported integration methods: {unsupported}. "
            f"Supported methods are: {sorted(supported_methods)}."
        )

    ad = adata if inplace else adata.copy()
    preprocess_kwargs = _as_dict(preprocess_kwargs)
    n_pcs = preprocess_kwargs.get("n_pcs", 20)
    n_neighbors = preprocess_kwargs.get("n_neighbors", 20)
    n_top_genes = preprocess_kwargs.get("n_top_genes", 2000)

    if preprocess:
        preprocess_for_integration(
            ad,
            raw_layer=raw_layer,
            normalized_layer=normalized_layer,
            batch_key=batch_key,
            inplace=True,
            **preprocess_kwargs,
        )
    elif "X_pca" not in ad.obsm:
        raise ValueError(
            "preprocess=False requires an existing adata.obsm['X_pca']. "
            "Run preprocess_for_integration first or set preprocess=True."
        )

    params = _merge_method_params(method_params)

    def _run_or_record(method_name, func):
        try:
            func()
        except Exception as exc:
            if fail_fast:
                raise
            warnings.warn(f"{method_name} failed: {exc}", RuntimeWarning)

    if "Combat" in selected_methods:
        def run_combat():
            combat_params = params.get("Combat", {})
            ad_combat = ad.copy()
            ad_combat.X = _copy_matrix(ad.layers[normalized_layer])
            if _use_highly_variable(ad_combat):
                ad_combat = ad_combat[:, ad_combat.var["highly_variable"].to_numpy()].copy()
            sc.pp.combat(ad_combat, key=batch_key, inplace=True, **combat_params)
            sc.tl.pca(ad_combat, n_comps=n_pcs, svd_solver="arpack")
            ad.obsm["Combat"] = ad_combat.obsm["X_pca"].copy()

        _run_or_record("Combat", run_combat)

    if "Harmony" in selected_methods:
        def run_harmony():
            from harmony import harmonize

            harmony_params = params.get("Harmony", {})
            ad.obsm["Harmony"] = harmonize(
                ad.obsm["X_pca"][:, :n_pcs],
                batch_mat=ad.obs,
                batch_key=batch_key,
                **harmony_params,
            )

        _run_or_record("Harmony", run_harmony)

    if "Scanorama" in selected_methods:
        def run_scanorama():
            scanorama_params = {"knn": n_neighbors, **params.get("Scanorama", {})}
            ad_scanorama = ad.copy()
            sc.external.pp.scanorama_integrate(
                ad_scanorama,
                key=batch_key,
                basis="X_pca",
                adjusted_basis="X_scanorama",
                **scanorama_params,
            )
            ad.obsm["Scanorama"] = ad_scanorama.obsm["X_scanorama"].copy()

        _run_or_record("Scanorama", run_scanorama)

    if "BBKNN" in selected_methods:
        def run_bbknn():
            bbknn_params = {"neighbors_within_batch": 3, **params.get("BBKNN", {})}
            ad_bbknn = ad.copy()
            sc.external.pp.bbknn(
                ad_bbknn,
                batch_key=batch_key,
                use_rep="X_pca",
                n_pcs=n_pcs,
                copy=False,
                **bbknn_params,
            )
            ad.uns["bbknn"] = ad_bbknn.uns["neighbors"].copy()
            ad.obsp["bbknn_connectivities"] = ad_bbknn.obsp["connectivities"].copy()
            ad.obsp["bbknn_distances"] = ad_bbknn.obsp["distances"].copy()

        _run_or_record("BBKNN", run_bbknn)

    scvi_methods = []
    if "scVI" in selected_methods:
        scvi_methods.append("scVI_normal")
    for method in ("scVI_normal", "scVI_nb"):
        if method in selected_methods:
            scvi_methods.append(method)

    for method in dict.fromkeys(scvi_methods):
        def run_scvi_method(method_name=method):
            scvi_params = deepcopy(params.get(method_name, {}))
            layer_spec = scvi_params.pop("layer", "normalized")
            layer = _resolve_scvi_layer(layer_spec, raw_layer, normalized_layer)
            save_path = scvi_params.pop("save_path", None)
            model_params = scvi_params.pop("model", {})
            train_params = scvi_params.pop("train", {})
            model_params.update(scvi_params)
            _run_scvi(
                ad,
                method=method_name,
                batch_key=batch_key,
                layer=layer,
                model_params=model_params,
                train_params=train_params,
                save_path=save_path,
            )

        _run_or_record(method, run_scvi_method)

    requested_r_methods = [method for method in selected_methods if method in R_METHODS]
    if requested_r_methods:
        if run_r:
            def run_r_methods():
                statuses = _run_r_integrations(
                    ad,
                    methods=requested_r_methods,
                    batch_key=batch_key,
                    layer=raw_layer,
                    n_top_genes=n_top_genes,
                    n_pcs=n_pcs,
                    params=params,
                    rscript_path=rscript_path,
                    r_executable=r_executable,
                    temp_dir=temp_dir,
                    keep_temp=keep_temp,
                )
                failed = {
                    method: status
                    for method, status in statuses.items()
                    if status.get("status") != "success"
                }
                if failed:
                    raise RuntimeError(f"R integration methods failed: {failed}")

            try:
                run_r_methods()
            except Exception as exc:
                if fail_fast:
                    raise
                warnings.warn(f"R integration failed: {exc}", RuntimeWarning)

    if not inplace:
        return ad
    return None
