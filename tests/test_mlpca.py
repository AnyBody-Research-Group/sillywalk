import numpy as np
import pandas as pd
import pytest

from sillywalk import PCAPredictor


def make_dataset(n=50, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=n)
    b = 0.5 * a + rng.normal(0.0, 0.5, size=n)  # correlated with a
    c = rng.normal(5.0, 2.0, size=n)
    d = np.full(n, 5.0)  # zero variance -> should be excluded
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    return df


def test_fit_excludes_low_variance_and_builds_model():
    df = make_dataset()
    model = PCAPredictor(df)

    # d should be excluded from PCA columns
    assert "d" not in model.pca_columns
    assert "d" in model.pca_low_variance_columns

    # At least one component should exist
    assert model.pca_n_components >= 1


def test_predict_respects_constraints_and_fills_with_means():
    df = make_dataset()
    model = PCAPredictor(df)

    constraints = {"a": 1.23}
    pred = model.predict(constraints)

    # All columns present
    assert set(pred.keys()) == set(model.columns)

    # Excluded column equals its mean
    d_mean = float(np.mean(df["d"]))
    assert np.isclose(pred["d"], d_mean)

    # Constrained variable should match approximately
    assert np.isclose(pred["a"], constraints["a"], atol=1e-6)


def test_parameters_components_roundtrip():
    df = make_dataset()
    model = PCAPredictor(df)

    # Take a sample row to build parameters on PCA columns
    row = df.iloc[0].to_dict()
    params = {col: float(row[col]) for col in model.pca_columns}

    pcs = model.parameters_to_components(params)
    back = model.components_to_parameters(pcs)

    for col in model.pca_columns:
        assert np.isfinite(back[col])
        assert np.isclose(back[col], params[col], rtol=1e-5, atol=1e-5)


def test_save_and_load_npz(tmp_path):
    df = make_dataset()
    model = PCAPredictor(df)

    file = tmp_path / "model.npz"
    model.export_pca_data(file)

    loaded = PCAPredictor.from_pca_data(file)
    constraints = {"a": -0.5}
    p1 = model.predict(constraints)
    p2 = loaded.predict(constraints)

    for k in model.columns:
        assert np.isclose(p1[k], p2[k], rtol=1e-10, atol=1e-10)


def test_predict_without_constraints_returns_means():
    df = make_dataset()
    model = PCAPredictor(df)

    pred = model.predict(None)
    assert set(pred.keys()) == set(model.columns)
    # Means
    for i, col in enumerate(model.columns):
        assert np.isclose(pred[col], model.means[i])


def test_wrong_components_length_raises():
    df = make_dataset()
    model = PCAPredictor(df)

    wrong = [0.0] * (model.pca_n_components + 1)
    with pytest.raises(ValueError):
        model.components_to_parameters(wrong)


def test_missing_parameter_raises():
    df = make_dataset()
    model = PCAPredictor(df)

    # Drop one required parameter
    row = df.iloc[0].to_dict()
    params = {col: float(row[col]) for col in model.pca_columns}
    if model.pca_columns:
        params.pop(model.pca_columns[0])
        with pytest.raises(ValueError):
            model.parameters_to_components(params)


def test_drop_parallel_constraints_internal():
    # Exercise the collinearity remover with synthetic data
    df = make_dataset()
    model = PCAPredictor(df)

    m = model.pca_n_components
    if m == 0:
        pytest.skip("No PCA components to test collinearity")

    # Two identical rows in B
    b1 = np.ones(m)
    B = np.stack([b1, 2.0 * b1, -b1])  # parallel and anti-parallel
    d = {"k1": 0.1, "k2": 0.2, "k3": -0.1}

    B2, d2 = model._drop_parallel_constraints(B, d)  # type: ignore[attr-defined]
    # Expect to drop some rows
    assert B2.shape[0] < B.shape[0]
    assert len(d2) == B2.shape[0]
