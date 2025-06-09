# -*- coding: utf-8 -*-
"""
Created on May 31 2025

This is John's implementation of the sillywalk algorithm.

@author: jr
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Union, Optional

import narwhals as nw
from narwhals.typing import IntoFrame


class PCAPredictor:

    def __baseinit__(
        self,
        all_means: np.ndarray,
        all_stds: np.ndarray,
        all_columns: list[str],
        pca_columns: list[str],
        pca_components: np.ndarray,
        pca_eigenvalues: np.ndarray,
    ):
        self.all_means = dict(zip(all_columns, all_means))
        self.all_stds = dict(zip(all_columns, all_stds))
        self.all_columns = all_columns

        self.pca_columns = pca_columns
        self.pca_components = pca_components
        self.pca_eigenvalues = pca_eigenvalues

        self._mean_vec = np.array([self.all_means[col] for col in self.pca_columns])
        self._std_vec = np.array([self.all_stds[col] for col in self.pca_columns])
        self._eigenvalues = pca_eigenvalues
        self._U_k = pca_components.T
        self.y_opt = None

    @classmethod
    def from_pca(
        cls,
        all_means: np.ndarray,
        all_stds: np.ndarray,
        all_columns: list[str],
        pca_columns: list[str],
        pca_components: np.ndarray,
        pca_eigenvalues: np.ndarray,
    ):
        instance = cls.__new__(cls, all_means, all_stds, all_columns, pca_columns, pca_components, pca_eigenvalues)
        instance.__baseinit__(
            all_means=all_means,
            all_stds=all_stds,
            all_columns=all_columns,
            pca_columns=pca_columns,
            pca_components=pca_components,
            pca_eigenvalues=pca_eigenvalues
        )
        return instance


    def __init__(
        self,
        df_native: IntoFrame,
        n_components: Union[int, None] = None,
        variance_threshold: float = 1e-8,
        relative_variance_ratio: float = 1e-3
    ):
        df = nw.from_native(df_native)

        original_columns = df.columns
        low_variance_cols = []
        mean_fill_values = {}

        variances = df.var()
        means = df.mean().abs()
        stds = np.sqrt(variances)
        relative_ratios = stds / (means + 1e-12)

        low_variance_cols = [
            col for col in df.columns
            if variances[col] < variance_threshold or relative_ratios[col] < relative_variance_ratio
        ]

        for col in low_variance_cols:
            mean_fill_values[col] = df[col].mean()

        df_reduced = df.drop(columns=low_variance_cols)
        reduced_columns = df_reduced.columns

        if n_components is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_reduced)
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            n_components = pca_temp.n_components_

        pca = PCA(n_components=n_components)
        
        self.__baseinit__(
            all_means=means,
            all_stds=stds,
            all_columns=df.columns,
            pca_columns=reduced_columns,
            pca_components=pca.components_,
            pca_eigenvalues=pca.explained_variance_
        )

    def _drop_parallel_constraints(self, B: np.ndarray, d: pd.Series) -> tuple:
        drop = []
        for i in range(B.shape[0]):
            for j in range(i + 1, B.shape[0]):
                inner_product = np.inner(B[i], B[j])
                norm_i = np.linalg.norm(B[i])
                norm_j = np.linalg.norm(B[j])
                if np.abs(inner_product - norm_j * norm_i) < 1E-7:
                    drop.append(d.index[j])
        drop = list(set(drop))
        drop_indices = [d.index.get_loc(key) for key in drop]
        B_new = np.delete(B, drop_indices, axis=0)
        d_new = d.drop(drop)
        return B_new, d_new

    def predict(
        self,
        constraints: Dict[str, float],
        target_pcs: Optional[np.ndarray] = None
    ) -> Union[pd.Series, tuple]:
        for col in constraints:
            if col in self.low_variance_cols:
                raise ValueError(f"Constraint cannot be applied to low-variance column '{col}'")

        constraint_indices = [self.reduced_columns.get_loc(var) for var in constraints.keys()]
        constraint_vals = [constraints[var] for var in constraints.keys()]
        standardized_constraints = [
            (val - self.mean_vec[i]) / self.std_vec[i] for i, val in zip(constraint_indices, constraint_vals)
        ]

        B = self.U_k[constraint_indices, :]
        d = pd.Series(standardized_constraints, index=list(constraints.keys()))

        B, d = self._drop_parallel_constraints(B, d)
        p = B.shape[0]
        m = self.U_k.shape[1]

        if target_pcs is None:
            target_pcs = np.zeros(m)

        rhs = np.zeros(m + p)
        rhs[m:] = d.values - (B @ target_pcs)

        K = np.zeros((m + p, m + p))
        K[range(m), range(m)] = 1.0 / self.eigenvalues_
        K[:m, m:] = B.T
        K[m:, :m] = B

        sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
        y_opt = sol[:m] + target_pcs

        x_hat_standardized = self.U_k @ y_opt
        x_hat_original = x_hat_standardized * self.std_vec + self.mean_vec
        predicted_reduced = pd.Series(x_hat_original, index=self.pca_columns)

        full_prediction = dict()
        for col in self.all_columns:
            if col in self.pca_columns: 
                full_prediction[col] = predicted_reduced[col]
            else:
               full_prediction[col] = self.all_means[col]

        self.y_opt = y_opt

        return  full_prediction
    

