"""Validation helpers for user-supplied feature transformers.

PCAPredictor accepts a fitted-or-fittable sklearn-like transformer to
preprocess features before PCA. The class makes assumptions about that
transformer that are not part of the sklearn API (shape preservation and
per-feature independence). The check in this module rejects unsupported
transformers at fit time so problems surface immediately rather than as
silently corrupted predictions.
"""

from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray


def validate_transformer(transformer: Any, X: NDArray) -> None:
    """Validate that ``transformer`` is compatible with PCAPredictor.

    PCAPredictor relies on three properties of the transformer that are not
    expressed by sklearn's API:

    1. It must implement ``fit_transform``, ``transform`` and
       ``inverse_transform``.
    2. It must preserve shape (no dimensionality reduction or one-hot
       expansion); ``transform(X).shape == X.shape``.
    3. It must act per-feature: output column ``k`` depends only on input
       column ``k``. This is required by ``_transform_partial`` and the
       column-indexed KKT solve in ``predict``. Per-feature transformers
       include StandardScaler, MinMaxScaler, PowerTransformer and Pipelines
       composed of them. ColumnTransformer (when it reorders), PCA,
       polynomial features, and similar will silently produce wrong
       predictions otherwise, so we reject them here.

    Raises:
        TypeError: if the required methods are missing.
        ValueError: if shape is not preserved or features are not independent.
    """
    for attr in ("fit_transform", "transform", "inverse_transform"):
        if not callable(getattr(transformer, attr, None)):
            raise TypeError(
                f"transformer must implement {attr}(); got {type(transformer).__name__}."
            )

    # Use a clone for validation so we don't disturb the user-supplied
    # transformer's state (it will be fit again by the caller).
    probe = deepcopy(transformer)

    # 1. Shape preservation.
    Xt = np.asarray(probe.fit_transform(X))
    if Xt.shape != X.shape:
        raise ValueError(
            "transformer must preserve shape; got "
            f"{X.shape} -> {Xt.shape}. Dimensionality-reducing transformers "
            "(PCA, projections, one-hot encoders, ...) are not supported."
        )

    # 2. Per-feature independence: perturbing one input column must not
    # change any other output column. Use a small sample for speed.
    n_features = X.shape[1]
    sample = X[: min(5, len(X))].copy()
    Xt_full = np.asarray(probe.transform(sample))
    rng = np.random.default_rng(0)
    for j in range(n_features):
        perturbed = sample.copy()
        # Add noise scaled to the column so the perturbation is detectable
        # regardless of feature magnitude.
        scale = float(np.std(sample[:, j])) or 1.0
        perturbed[:, j] = sample[:, j] + rng.normal(scale=scale, size=sample.shape[0])
        Xt_perturbed = np.asarray(probe.transform(perturbed))
        for k in range(n_features):
            if k == j:
                continue
            if not np.allclose(Xt_full[:, k], Xt_perturbed[:, k], rtol=1e-6, atol=1e-8):
                raise ValueError(
                    "transformer must be per-feature: changing input "
                    f"feature {j} altered output feature {k}. "
                    "ColumnTransformer (with reordering), PCA, polynomial "
                    "features, and similar are not supported."
                )
