"""
ML‑PCA utilities and a simple PCA-based predictor.

This module provides:
- The PCAPredictor class that selects useful columns, fits a PCA model,
  predicts full parameter vectors from partial constraints, and converts
  between primal parameters and principal components.

Design notes
- Only numeric columns participate in PCA. Non-numeric columns are carried
  through via their mean values in predictions.
- Columns with too little absolute or relative variance are excluded from PCA.
- The prediction uses a least-squares KKT system with optional targets in
  principal-component space.
"""

from collections.abc import Mapping, Sequence
from io import BytesIO
from os import PathLike
from typing import IO
from warnings import warn

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoDataFrameT
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

# Type Alias Definition
NumericSequenceOrArray = Sequence[float | int] | NDArray[np.floating | np.integer]
StringSequenceOrArray = Sequence[str] | NDArray[np.str_]


def _make_all_columns_numeric(df: nw.DataFrame) -> nw.DataFrame:
    """Ensure all columns are numeric in a narwhals DataFrame.

    Non-numeric columns are replaced by a column of NaNs with the same name.
    This keeps the column ordering stable while allowing numeric reductions.
    """
    return df.with_columns(
        nw.lit(float("nan")).alias(col)
        for col in df.select(~nw.selectors.numeric()).columns
    )


def _dataframe_to_dict(df: IntoDataFrame | None) -> Mapping[str, float | int]:
    """Convert a 1-row DataFrame-like into a flat mapping.

    If df is None, returns an empty mapping.
    """
    if df is None:
        return {}
    out_dict = nw.from_native(df, eager_only=True).to_dict(as_series=False)
    return {k: v[0] for k, v in out_dict.items()}


class PCAPredictor:
    """PCA-based predictor over a subset of columns.

    Workflow
    1) Fit from a (pandas/polars) DataFrame or other narwhals-supported frame.
    2) Predict a full parameter mapping given partial constraints.
    3) Convert between primal parameters and principal components.

    Parameters selected for PCA are decided by two heuristics:
    - Absolute variance threshold (variance_threshold)
    - Relative standard deviation threshold (relative_variance_ratio), i.e.
      std/mean. Columns below either threshold are excluded.
    """

    def __baseinit__(
        self,
        means: NumericSequenceOrArray,
        stds: NumericSequenceOrArray,
        columns: StringSequenceOrArray,
        pca_columns: StringSequenceOrArray,
        pca_eigenvectors: NumericSequenceOrArray,
        pca_eigenvalues: NumericSequenceOrArray,
    ) -> None:
        """Internal initializer used by classmethods and __init__."""
        if isinstance(columns, np.ndarray):
            columns = columns.tolist()
        if isinstance(pca_columns, np.ndarray):
            pca_columns = pca_columns.tolist()

        self.means = np.array(means)
        self.stds = np.array(stds)
        self.columns = list(columns)  # Ensure list for index()
        self.pca_columns = list(pca_columns)  # Ensure list for index()
        self.pca_eigenvectors = np.array(pca_eigenvectors)
        self.pca_explained_variance_ratio = (
            np.array(pca_eigenvalues) / np.sum(pca_eigenvalues)
            if np.size(pca_eigenvalues) > 0
            else np.array([])
        )
        self.pca_low_variance_columns = set(self.columns).difference(self.pca_columns)

        self.pca_n_components = len(self.pca_columns)
        self._pca_means = np.array(
            [self.means[self.columns.index(col)] for col in self.pca_columns]
        )
        self._pca_stds = np.array(
            [self.stds[self.columns.index(col)] for col in self.pca_columns]
        )
        self.pca_eigenvalues = np.array(pca_eigenvalues)
        self.y_opt: NDArray | None = None

    def _pca_column_idx(self, column: str) -> int:
        """Return the index of a PCA column in the reduced space."""
        if column not in self.pca_columns:
            raise ValueError(f"Column '{column}' is not a PCA column.")
        if column not in self.columns:
            raise ValueError(f"Column '{column}' is not in original dataset.")
        return self.pca_columns.index(column)

    @classmethod
    def from_pca_data(
        cls,
        filename: str | None = None,
        means: NumericSequenceOrArray | None = None,
        stds: NumericSequenceOrArray | None = None,
        columns: StringSequenceOrArray | None = None,
        pca_columns: StringSequenceOrArray | None = None,
        pca_eigenvectors: NDArray | None = None,
        pca_eigenvalues: NDArray | None = None,
    ) -> "PCAPredictor":
        """Load a saved model from a .npz file created by export_pca_data."""
        if filename is not None:
            data = np.load(filename, allow_pickle=False)
            means = data["means"]
            stds = data["stds"]
            columns = data["columns"]
            pca_columns = data["pca_columns"]
            pca_eigenvectors = data["pca_eigenvectors"]
            pca_eigenvalues = data["pca_eigenvalues"]
        if means is None or stds is None or columns is None or pca_columns is None:
            raise ValueError("Missing required PCA data.")

        instance = cls.__new__(cls)
        instance.__baseinit__(
            means=means,
            stds=stds,
            columns=columns,
            pca_columns=pca_columns,
            pca_eigenvectors=pca_eigenvectors,
            pca_eigenvalues=pca_eigenvalues,
        )
        return instance

    def __init__(
        self,
        data: IntoDataFrameT,
        n_components: float | int | None = 0.99,
        svd_solver: str = "auto",
        variance_threshold: float = 1e-8,
        relative_variance_ratio: float = 1e-3,
    ) -> None:
        """Fit a PCA model on columns that pass simple variance screening."""
        df = nw.from_native(data, eager_only=True)

        columns = np.array(df.columns)
        df_numeric = _make_all_columns_numeric(df)
        meanvalues = (
            df_numeric.select(nw.all().mean().fill_null(float("nan")))
            .to_numpy()
            .flatten()
        )
        stdvalues = df_numeric.select(nw.all().std().fill_null(0)).to_numpy().flatten()
        variances = df_numeric.select(nw.all().var().fill_null(0)).to_numpy().flatten()
        _relative_ratios = abs(stdvalues / (meanvalues + 1e-12))

        pca_columns = columns[
            np.logical_and(
                variances >= variance_threshold,
                _relative_ratios >= relative_variance_ratio,
            )
        ]
        df_reduced = df.select(pca_columns)

        # Handle the edge-case of zero selected features gracefully
        if len(pca_columns) == 0:
            self.__baseinit__(
                means=meanvalues,
                stds=stdvalues,
                columns=columns,
                pca_columns=pca_columns,
                pca_eigenvectors=np.zeros((0, 0), dtype=float),
                pca_eigenvalues=np.zeros((0,), dtype=float),
            )
            return

        X = df_reduced.to_numpy()
        X_scaled = StandardScaler().fit_transform(X)

        # if n_components is None:
        #     # Number of non-zero components is limited by rank of X
        #     n_components = PCA(n_components='mle').fit(X_scaled).n_components_

        pca = PCA(n_components=n_components, svd_solver=svd_solver)
        pca.fit(X_scaled)

        self.__baseinit__(
            means=meanvalues,
            stds=stdvalues,
            columns=columns,
            pca_columns=pca_columns,
            pca_eigenvectors=pca.components_,
            pca_eigenvalues=pca.explained_variance_,
        )

    def _drop_parallel_constraints(
        self, B: NDArray, d: dict[str, float]
    ) -> tuple[NDArray, dict[str, float]]:
        """Drop linearly dependent constraints (parallel or anti-parallel rows).

        Rows i and j are considered collinear if |cos(theta)| ~ 1, i.e.
        |<b_i, b_j>| ≈ ||b_i||·||b_j||.
        """
        drop: list[str] = []
        for i in range(B.shape[0]):
            for j in range(i + 1, B.shape[0]):
                inner_product = np.inner(B[i], B[j])
                norm_i = np.linalg.norm(B[i])
                norm_j = np.linalg.norm(B[j])
                if norm_i == 0 or norm_j == 0:
                    continue
                # Detect both parallel and anti-parallel vectors
                if abs(abs(inner_product) - (norm_j * norm_i)) < 1e-7:
                    drop.append(list(d)[j])
        drop = list(set(drop))
        drop_indices = [list(d).index(key) for key in drop]
        B_new = np.delete(B, drop_indices, axis=0) if drop_indices else B
        d_new = {k: v for k, v in d.items() if k not in drop}
        return B_new, d_new

    def predict(
        self,
        constraints: Mapping[str, float | int] | IntoDataFrame | None = None,
        target_pcs: NDArray | None = None,
    ) -> dict[str, float | int]:
        """Predict a full parameter vector given partial constraints.

        constraints: mapping or 1-row DataFrame containing values for a
          subset of PCA columns. Non-PCA columns cannot be constrained.
        target_pcs: optional target values in PC space to bias the solution.
        """
        if constraints is None:
            warn("No constraints provided. Returning column means.")
            return dict(zip(self.columns, self.means.tolist()))

        if not isinstance(constraints, Mapping):
            constraints = _dataframe_to_dict(constraints)

        if not constraints:
            warn(
                "No constraints provided. Result is the mean value of the PCA columns."
            )
            constraints = {self.pca_columns[0]: self._pca_means[0]}

        low_variance_constraints = [
            col for col in constraints if col in self.pca_low_variance_columns
        ]
        if low_variance_constraints:
            raise ValueError(
                f"Constraint cannot be applied to excluded low-variance columns: {low_variance_constraints}"
            )

        constraint_indices = np.array(
            [
                self._pca_column_idx(str(var))
                for var in constraints
                if var in self.pca_columns
            ]
        )

        standardized_constraints: dict[str, float] = {}
        for var in constraints:
            var = str(var)
            if var not in self.pca_columns:
                raise ValueError(
                    f"Constraint variable '{var}' is not part of the PCA columns."
                )
            idx = self._pca_column_idx(var)
            standardized_constraints[var] = (
                float(constraints[var]) - float(self._pca_means[idx])
            ) / float(self._pca_stds[idx])

        B = self.pca_eigenvectors.T[constraint_indices, :]
        d = {i: standardized_constraints[i] for i in standardized_constraints}

        B, d = self._drop_parallel_constraints(B, d)
        p = B.shape[0]
        m = self.pca_eigenvectors.T.shape[1] if self.pca_eigenvectors.size else 0

        if m == 0:
            # No PCA features: return means
            return dict(zip(self.columns, self.means.tolist()))

        if target_pcs is None:
            target_pcs = np.zeros(m)

        rhs = np.zeros(m + p)
        rhs[m:] = np.array(list(d.values())) - (B @ target_pcs)

        K = np.zeros((m + p, m + p))
        K[range(m), range(m)] = 1.0 / self.pca_eigenvalues
        K[:m, m:] = B.T
        K[m:, :m] = B

        sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
        y_opt = sol[:m] + target_pcs  # Bias by target PCs

        x_hat_standardized = self.pca_eigenvectors.T @ y_opt
        x_hat_original = x_hat_standardized * self._pca_stds + self._pca_means
        predicted_reduced = dict(zip(self.pca_columns, x_hat_original))

        self.y_opt = y_opt

        full_prediction = dict(zip(self.columns, self.means.tolist()))
        for col in self.pca_columns:
            full_prediction[col] = predicted_reduced[col]

        return full_prediction

    def parameters_to_components(
        self, parameters: Mapping[str, float | int] | IntoDataFrame
    ) -> list[float | int]:
        """Return principal components for a given set of primal parameters."""
        if not isinstance(parameters, Mapping):
            param = _dataframe_to_dict(parameters)
        else:
            param = parameters

        normalized_params = np.zeros_like(self._pca_means, dtype=float)
        for i, col in enumerate(self.pca_columns):
            if col not in param:
                raise ValueError(f"Parameter '{col}' is missing from input data.")
            normalized_params[i] = (
                float(param[col]) - float(self._pca_means[i])
            ) / float(self._pca_stds[i])

        pcs = np.dot(self.pca_eigenvectors.T, normalized_params)
        return pcs.tolist()

    def components_to_parameters(
        self, principal_components: NumericSequenceOrArray
    ) -> dict[str, float | int]:
        """Return primal parameters from a set of principal components."""

        if not isinstance(principal_components, np.ndarray):
            principal_components = np.array(principal_components, dtype=float)

        if len(principal_components) != self.pca_n_components:
            raise ValueError(
                f"Wrong number of pca modes. System has {self.pca_n_components} modes. "
            )

        reduced_params = (
            self.pca_eigenvectors @ principal_components
        ) * self._pca_stds + self._pca_means
        full_params = dict(zip(self.columns, self.means.tolist()))
        for col, val in zip(self.pca_columns, reduced_params.tolist()):
            full_params[col] = val

        return full_params

    def export_pca_data(
        self, filename: str | PathLike | None = None
    ) -> None | IO[bytes]:
        """Save the model to a .npz file, which can later be loaded with
        `sillywalk.PCAPredictor.from_pca_data(filename)`.

        If filename is None, return an in-memory buffer.
        """

        if filename is None:
            fh = BytesIO()

            np.savez_compressed(
                fh,
                means=self.means,
                stds=self.stds,
                columns=self.columns,
                pca_columns=self.pca_columns,
                pca_eigenvectors=self.pca_eigenvectors,
                pca_eigenvalues=self.pca_eigenvalues,
            )
            fh.seek(0)
            return fh
        else:
            np.savez_compressed(
                filename,
                means=self.means,
                stds=self.stds,
                columns=self.columns,
                pca_columns=self.pca_columns,
                pca_eigenvectors=self.pca_eigenvectors,
                pca_eigenvalues=self.pca_eigenvalues,
            )
