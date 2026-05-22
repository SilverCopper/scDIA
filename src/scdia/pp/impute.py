import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .normalize import _get_matrix_from_adata


PIMMS_METHODS = {
    "PIMMS_DAE": "DAE",
    "PIMMS_VAE": "VAE",
    "PIMMS_CF": "CF",
    "DAE": "DAE",
    "VAE": "VAE",
    "CF": "CF",
}


def _get_fancyimpute_class(method: str):
    try:
        from fancyimpute import (
            IterativeImputer,
            IterativeSVD,
            KNN,
            MatrixFactorization,
            NuclearNormMinimization,
            SimpleFill,
            SoftImpute,
        )
    except ImportError as exc:
        raise ImportError(
            "pp.impute requires fancyimpute. Install fancyimpute before using imputation."
        ) from exc

    classes = {
        "KNN": KNN,
        "NuclearNormMinimization": NuclearNormMinimization,
        "SoftImpute": SoftImpute,
        "IterativeImputer": IterativeImputer,
        "IterativeSVD": IterativeSVD,
        "MatrixFactorization": MatrixFactorization,
        "SimpleFill": SimpleFill,
    }
    if method in classes:
        return classes[method], method

    lower_lookup = {name.lower(): (cls, name) for name, cls in classes.items()}
    key = method.lower()
    if key not in lower_lookup:
        supported = ", ".join([*classes, *PIMMS_METHODS])
        raise ValueError(
            f"Unknown imputation method '{method}'. Choose from: {supported}."
        )
    return lower_lookup[key]


def _resolve_pimms_method(method: str) -> str | None:
    if method in PIMMS_METHODS:
        return PIMMS_METHODS[method]
    return PIMMS_METHODS.get(method.upper())


def _require_pimms():
    try:
        from pimmslearn.sklearn.ae_transformer import AETransformer
        from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer
    except ImportError as exc:
        raise ImportError(
            "PIMMS imputation requires pimms-learn. Install it with "
            "`pip install pimms-learn` or `pip install -e '.[pimms]'`."
        ) from exc
    return AETransformer, CollaborativeFilteringTransformer


def _run_pimms_imputer(
    x: np.ndarray,
    *,
    method: str,
    obs_names,
    var_names,
    epochs_max: int,
    cuda: bool,
    patience: int | None,
    hidden_layers: list[int] | tuple[int, ...] | None,
    latent_dim: int,
    batch_size: int,
    n_factors: int | None,
    out_folder: str,
    target_column: str,
    sample_column: str,
    item_column: str,
    **kwargs,
) -> np.ndarray:
    AETransformer, CollaborativeFilteringTransformer = _require_pimms()

    df = pd.DataFrame(
        x,
        index=pd.Index(obs_names, name=sample_column),
        columns=pd.Index(var_names, name=item_column),
    )
    resolved = _resolve_pimms_method(method)

    if resolved in {"DAE", "VAE"}:
        if hidden_layers is None:
            hidden_layers = [min(512, max(16, df.shape[1] * 2))]
        model = AETransformer(
            model=resolved,
            hidden_layers=list(hidden_layers),
            latent_dim=latent_dim,
            batch_size=batch_size,
            out_folder=out_folder,
            **kwargs,
        )
        model.fit(df, epochs_max=epochs_max, cuda=cuda, patience=patience)
        return (
            model.transform(df)
            .reindex(index=df.index, columns=df.columns)
            .to_numpy()
        )

    if resolved == "CF":
        series = df.stack()
        series.name = target_column
        model = CollaborativeFilteringTransformer(
            target_column=target_column,
            sample_column=sample_column,
            item_column=item_column,
            n_factors=latent_dim if n_factors is None else n_factors,
            batch_size=batch_size,
            out_folder=out_folder,
            **kwargs,
        )
        model.fit(series, epochs_max=epochs_max, cuda=cuda, patience=patience)
        imputed = (
            model.transform(series)
            .unstack()
            .reindex(index=df.index, columns=df.columns)
        )
        return imputed.to_numpy()

    supported = ", ".join(PIMMS_METHODS)
    raise ValueError(
        f"Unknown PIMMS imputation method '{method}'. Choose from: {supported}."
    )


def impute(
    adata,
    *,
    method: str = "KNN",
    layer: str | None = None,
    output_layer: str = "imputed",
    inplace: bool = True,
    zero_as_missing: bool = True,
    fill_method: str = "mean",
    epochs_max: int = 100,
    cuda: bool = False,
    patience: int | None = None,
    hidden_layers: list[int] | tuple[int, ...] | None = None,
    latent_dim: int = 15,
    batch_size: int = 64,
    n_factors: int | None = None,
    out_folder: str = ".",
    target_column: str = "intensity",
    sample_column: str = "Sample ID",
    item_column: str = "feature",
    **kwargs,
):
    """Impute missing values in an AnnData matrix.

    ``method`` can be a ``fancyimpute`` method or one of the PIMMS methods:
    ``"PIMMS_DAE"``, ``"PIMMS_VAE"``, or ``"PIMMS_CF"``. Short aliases
    ``"DAE"``, ``"VAE"``, and ``"CF"`` are also accepted.
    """
    ad = adata if inplace else adata.copy()
    matrix, _ = _get_matrix_from_adata(ad, layer)
    if issparse(matrix):
        x = matrix.toarray().astype(np.float64, copy=False)
    else:
        x = np.asarray(matrix, dtype=np.float64)
    x = x.copy()
    missing_mask = ~np.isfinite(x)
    if zero_as_missing:
        missing_mask |= x == 0
    x[missing_mask] = np.nan

    if _resolve_pimms_method(method) is not None:
        imputed = _run_pimms_imputer(
            x,
            method=method,
            obs_names=ad.obs_names,
            var_names=ad.var_names,
            epochs_max=epochs_max,
            cuda=cuda,
            patience=patience,
            hidden_layers=hidden_layers,
            latent_dim=latent_dim,
            batch_size=batch_size,
            n_factors=n_factors,
            out_folder=out_folder,
            target_column=target_column,
            sample_column=sample_column,
            item_column=item_column,
            **kwargs,
        )
    else:
        imputer_cls, resolved_method = _get_fancyimpute_class(method)
        if resolved_method == "SimpleFill":
            imputer = imputer_cls(fill_method=fill_method, **kwargs)
        else:
            imputer = imputer_cls(**kwargs)
        imputed = imputer.fit_transform(x)

    ad.layers[output_layer] = imputed
    if not inplace:
        return ad
    return None
