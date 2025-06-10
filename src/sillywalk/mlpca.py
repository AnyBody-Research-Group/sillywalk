"""
Created on May 31 2025

This is John's implementation of the sillywalk algorithm.

@author: jr
"""

from collections.abc import Mapping, Sequence

import narwhals as nw
import numpy as np
from narwhals.typing import IntoFrame
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

# Type Alias Definition
NumericSequenceOrArray = Sequence[float | int] | NDArray[np.floating | np.integer]
StringSequenceOrArray = Sequence[str] | NDArray[np.str_]


class PCAPredictor:
    def __baseinit__(
        self,
        all_means: NumericSequenceOrArray,
        all_stds: NumericSequenceOrArray,
        all_columns: StringSequenceOrArray,
        pca_columns: StringSequenceOrArray,
        pca_components: NumericSequenceOrArray,
        pca_eigenvalues: NumericSequenceOrArray,
    ):
        if isinstance(all_columns, np.ndarray):
            all_columns = all_columns.tolist()
        if isinstance(pca_columns, np.ndarray):
            pca_columns = pca_columns.tolist()

        self.all_means = dict(zip(all_columns, np.array(all_means)))
        self.all_stds = dict(zip(all_columns, np.array(all_stds)))
        self.all_columns = all_columns

        self.pca_columns = dict(zip(pca_columns, range(len(pca_columns))))
        self.pca_components = np.array(pca_components)

        self._low_variance_columns = set(all_columns).difference(pca_columns)
        self._mean_vec = np.array([self.all_means[col] for col in self.pca_columns])
        self._std_vec = np.array([self.all_stds[col] for col in self.pca_columns])
        self._eigenvalues = np.array(pca_eigenvalues)
        self._U_k = self.pca_components.T
        self.y_opt: NDArray | None = None

    @classmethod
    def from_pca(
        cls,
        all_means: NumericSequenceOrArray,
        all_stds: NumericSequenceOrArray,
        all_columns: StringSequenceOrArray,
        pca_columns: StringSequenceOrArray,
        pca_components: NDArray,
        pca_eigenvalues: NDArray,
    ):
        instance = cls.__new__(cls)
        instance.__baseinit__(
            all_means=all_means,
            all_stds=all_stds,
            all_columns=all_columns,
            pca_columns=pca_columns,
            pca_components=pca_components,
            pca_eigenvalues=pca_eigenvalues,
        )
        return instance

    def __init__(
        self,
        df_native: IntoFrame,
        n_components: int | None = None,
        variance_threshold: float = 1e-8,
        relative_variance_ratio: float = 1e-3,
    ):
        df = nw.from_native(df_native)

        # original_columns = df.columns
        # low_variance_cols = []
        # mean_fill_values = {}

        all_columns = np.array(df.columns)
        variances = df.select(nw.all().var()).to_numpy().flatten()
        means = df.select(nw.all().mean()).to_numpy().flatten()
        stds = df.select(nw.all().std()).to_numpy().flatten()
        relative_ratios = stds / (means + 1e-12)

        pca_columns = all_columns[
            np.logical_and(
                variances >= variance_threshold,
                relative_ratios >= relative_variance_ratio,
            )
        ]
        df_reduced = df.select(pca_columns)

        X_scaled = StandardScaler().fit_transform(df_reduced)

        if n_components is None:
            n_components = PCA().fit(X_scaled).n_components_

        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)

        self.__baseinit__(
            all_means=means,
            all_stds=stds,
            all_columns=df.columns,
            pca_columns=pca_columns,
            pca_components=pca.components_,
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
        constraints: Mapping[str, float | int],
        target_pcs: NDArray | None = None,
    ) -> dict[str, float | int]:
        low_variance_constraints = [
            col for col in constraints if col in self._low_variance_columns
        ]
        if low_variance_constraints:
            raise ValueError(
                f"Constraint cannot be applied to excluded low-variance columns: {low_variance_constraints}"
            )

        # constraint_indices = [self.pca_columns.get_loc(var) for var in constraints.keys()]
        constraint_indices = np.array(
            [self.pca_columns[var] for var in constraints if var in self.pca_columns]
        )

        standardized_constraints = {}
        for var in constraints:
            if var not in self.pca_columns:
                raise ValueError(
                    f"Constraint variable '{var}' is not part of the PCA columns."
                )
            idx = self.pca_columns[var]
            standardized_constraints[var] = (
                constraints[var] - self._mean_vec[idx]
            ) / self._std_vec[idx]

        # standardized_constraints = [
        #     (val - self._mean_vec[i]) / self._std_vec[i] for i, val in zip(constraint_indices, constraints.values())
        # ]

        B = self._U_k[constraint_indices, :]
        d = {i: standardized_constraints[i] for i in standardized_constraints}

        B, d = self._drop_parallel_constraints(B, d)
        p = B.shape[0]
        m = self._U_k.shape[1]

        if target_pcs is None:
            target_pcs = np.zeros(m)

        rhs = np.zeros(m + p)
        rhs[m:] = np.array(list(d.values())) - (B @ target_pcs)

        K = np.zeros((m + p, m + p))
        K[range(m), range(m)] = 1.0 / self._eigenvalues
        K[:m, m:] = B.T
        K[m:, :m] = B

        sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
        y_opt = sol[:m] + target_pcs

        x_hat_standardized = self._U_k @ y_opt
        x_hat_original = x_hat_standardized * self._std_vec + self._mean_vec
        predicted_reduced = dict(zip(self.pca_columns, x_hat_original))

        full_prediction = dict()
        for col in self.all_columns:
            if col in self.pca_columns:
                full_prediction[col] = predicted_reduced[col]
            else:
                full_prediction[col] = self.all_means[col]

        self.y_opt = y_opt

        return full_prediction
