import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sillywalk import PCAPredictor


def test_custom_transformer_minmax():
    # correlated data
    df = pl.DataFrame(
        {
            "a": [0, 10, 20, 30, 40],
            "b": [0, 20, 40, 60, 80],  # b = 2*a
        }
    )

    # Use MinMaxScaler (feature range 0-1)
    # Note: For PCAPredictor to work perfectly with linear data + linear scaler,
    # it should recover exact values.
    transformer = MinMaxScaler()
    model = PCAPredictor(
        df, transformer=transformer, n_components=1, relative_variance_ratio=0.0
    )

    # Predict: if a=20, b should be 40.
    res = model.predict({"a": 20})
    assert res["b"] == pytest.approx(40, rel=1e-4)
    assert res["a"] == pytest.approx(20, rel=1e-4)  # Consistency check

    # Verify transformer was stored
    assert isinstance(model.transformer, MinMaxScaler)


def test_pipeline_transformer():
    np.random.seed(42)
    # Linear relation
    a = np.linspace(0, 10, 50)
    b = 2 * a + 1
    df = pl.DataFrame({"a": a, "b": b})

    # Pipeline with StandardScaler
    pipeline = make_pipeline(StandardScaler())
    model = PCAPredictor(df, transformer=pipeline, relative_variance_ratio=0.0)

    res = model.predict({"a": 5})
    assert res["b"] == pytest.approx(11, rel=1e-1)


def test_save_load_custom_transformer(tmpdir):
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    transformer = MinMaxScaler()
    model = PCAPredictor(df, transformer=transformer, relative_variance_ratio=0.0)

    path = str(tmpdir.join("model.npz"))
    model.export_pca_data(path)

    loaded = PCAPredictor.from_pca_data(path)
    assert isinstance(loaded.transformer, MinMaxScaler)
    # Check scale_ attribute or similar match
    assert np.allclose(loaded.transformer.scale_, model.transformer.scale_)

    pred = loaded.predict({"a": 2.0})
    assert pred["b"] == pytest.approx(4.0, rel=1e-4)
