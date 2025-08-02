"""
Created on May 31 2025

This is John Rasmussen and Morten Enemark Lund's implementation of the sillywalk algorithm.

@author: John Rasmussen and Morten Enemark Lund
"""

from collections.abc import Mapping, Sequence
from os import PathLike
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


def make_all_columns_numeric(df: nw.DataFrame) -> nw.DataFrame:
    return df.with_columns(
        nw.lit(float("nan")).alias(col)
        for col in df.select(~nw.selectors.numeric()).columns
    )


def _dataframe_to_dict(df) -> Mapping[str, float | int]:
    out_dict = nw.from_native(df, eager_only=True).to_dict(as_series=False)
    return {k: v[0] for k, v in out_dict.items()}


class PCAPredictor:
    def __baseinit__(
        self,
        means: NumericSequenceOrArray,
        stds: NumericSequenceOrArray,
        columns: StringSequenceOrArray,
        pca_columns: StringSequenceOrArray,
        pca_eigenvectors: NumericSequenceOrArray,
        pca_eigenvalues: NumericSequenceOrArray,
    ):
        if isinstance(columns, np.ndarray):
            columns = columns.tolist()
        if isinstance(pca_columns, np.ndarray):
            pca_columns = pca_columns.tolist()

        self.means = np.array(means)
        self.stds = np.array(stds)
        self.columns = list(columns)  # Ensure list for index()
        self.pca_columns = list(pca_columns)  # Ensure list for index()
        self.pca_eigenvectors = np.array(pca_eigenvectors)
        self.pca_explained_variance_ratio = np.array(pca_eigenvalues) / np.sum(
            pca_eigenvalues
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
        """Returns the index of a PCA column."""
        if column not in self.pca_columns:
            raise ValueError(f"Column '{column}' is not a PCA column.")
        if column not in self.columns:
            raise ValueError(f"Column '{column}' is not in original dataset.")
        return self.pca_columns.index(column)

    @classmethod
    def from_npz(
        cls,
        filename: PathLike,
    ):
        data = np.load(filename, allow_pickle=False)
        instance = cls.__new__(cls)
        instance.__baseinit__(
            means=data["means"],
            stds=data["stds"],
            columns=data["columns"],
            pca_columns=data["pca_columns"],
            pca_eigenvectors=data["pca_eigenvectors"],
            pca_eigenvalues=data["pca_eigenvalues"],
        )
        return instance

    @classmethod
    def from_pca_values(
        cls,
        means: NumericSequenceOrArray,
        stds: NumericSequenceOrArray,
        columns: StringSequenceOrArray,
        pca_columns: StringSequenceOrArray,
        pca_eigenvectors: NDArray,
        pca_eigenvalues: NDArray,
    ):
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
        n_components: int | None = None,
        variance_threshold: float = 1e-8,
        relative_variance_ratio: float = 1e-3,
    ):
        df = nw.from_native(data, eager_only=True)

        # Select all numeric columns
        # df = df.select(nw.selectors.numeric())

        # original_columns = df.columns
        # low_variance_cols = []
        # mean_fill_values = {}

        columns = np.array(df.columns)
        df_numeric = make_all_columns_numeric(df)
        meanvalues = (
            df_numeric.select(nw.all().mean().fill_null(float("nan")))
            .to_numpy()
            .flatten()
        )
        stdvalues = df_numeric.select(nw.all().std().fill_null(0)).to_numpy().flatten()
        variances = (
            df_numeric.select(nw.all().var().fill_null(0)).to_numpy().flatten()
        )  # variances = df.var().with_columns((~nw.selectors.numeric()).str.to_integer()).fill_null(0).to_numpy().flatten()
        _relative_ratios = stdvalues / (meanvalues + 1e-12)

        pca_columns = columns[
            np.logical_and(
                variances >= variance_threshold,
                _relative_ratios >= relative_variance_ratio,
            )
        ]
        df_reduced = df.select(pca_columns)

        X_scaled = StandardScaler().fit_transform(df_reduced)

        if n_components is None:
            n_components = PCA().fit(X_scaled).n_components_

        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)

        self.__baseinit__(
            means=meanvalues,
            stds=stdvalues,
            columns=columns,
            pca_columns=pca_columns,
            pca_eigenvectors=pca.components_,
            pca_eigenvalues=pca.explained_variance_,
        )

    def _drop_parallel_constraints(self, B: NDArray, d: dict[str, float]) -> tuple:
        drop: list[str] = []
        for i in range(B.shape[0]):
            for j in range(i + 1, B.shape[0]):
                inner_product = np.inner(B[i], B[j])
                norm_i = np.linalg.norm(B[i])
                norm_j = np.linalg.norm(B[j])
                if np.abs(inner_product - norm_j * norm_i) < 1e-7:
                    drop.append(list(d)[j])
        drop = list(set(drop))
        drop_indices = [list(d).index(key) for key in drop]
        B_new = np.delete(B, drop_indices, axis=0)
        d_new = {k: v for k, v in d.items() if k not in drop}
        return B_new, d_new

    def predict(
        self,
        constraints: Mapping[str, float | int] | IntoDataFrame | None = None,
        target_pcs: NDArray | None = None,
    ) -> dict[str, float | int]:
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

        # constraint_indices = [self.pca_columns.get_loc(var) for var in constraint_map.keys()]
        constraint_indices = np.array(
            [
                self._pca_column_idx(var)
                for var in constraints
                if var in self.pca_columns
            ]
        )

        standardized_constraints = {}
        for var in constraints:
            if var not in self.pca_columns:
                raise ValueError(
                    f"Constraint variable '{var}' is not part of the PCA columns."
                )
            idx = self._pca_column_idx(var)
            standardized_constraints[var] = (
                constraints[var] - self._pca_means[idx]
            ) / self._pca_stds[idx]

        # standardized_constraints = [
        #     (val - self._mean_vec[i]) / self._std_vec[i] for i, val in zip(constraint_indices, constraint_map.values())
        # ]

        B = self.pca_eigenvectors.T[constraint_indices, :]
        d = {i: standardized_constraints[i] for i in standardized_constraints}

        B, d = self._drop_parallel_constraints(B, d)
        p = B.shape[0]
        m = self.pca_eigenvectors.T.shape[1]

        if target_pcs is None:
            target_pcs = np.zeros(m)

        rhs = np.zeros(m + p)
        rhs[m:] = np.array(list(d.values())) - (B @ target_pcs)

        K = np.zeros((m + p, m + p))
        K[range(m), range(m)] = 1.0 / self.pca_eigenvalues
        K[:m, m:] = B.T
        K[m:, :m] = B

        sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
        y_opt = sol[:m] + target_pcs  # TODO understand why we add the target_pcs here

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
        """Returns the principal components associated with a set of primal parameters"""
        if not isinstance(parameters, Mapping):
            param = _dataframe_to_dict(parameters)
        else:
            param = parameters

        normalized_params = np.zeros_like(self._pca_means)
        for i, col in enumerate(self.pca_columns):
            if col not in param:
                raise ValueError(f"Parameter '{col}' is missing from input data.")
            normalized_params[i] = (param[col] - self._pca_means[i]) / self._pca_stds[i]

        pcs = np.dot(self.pca_eigenvectors.T, normalized_params)
        return pcs.tolist()

    def components_to_parameters(
        self, principal_components: NumericSequenceOrArray
    ) -> dict[str, float | int]:
        """Return the primal parameters from a set of principal components."""

        if not isinstance(principal_components, np.ndarray):
            principal_components = np.array(principal_components)

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

    def save_npz(self, filename: PathLike):
        """Save the model to a .npz file."""
        np.savez(
            filename,
            means=self.means,
            stds=self.stds,
            columns=self.columns,
            pca_columns=self.pca_columns,
            pca_eigenvectors=self.pca_eigenvectors,
            pca_eigenvalues=self.pca_eigenvalues,
            allow_pickle=False,
        )
