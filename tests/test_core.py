import numpy as np
import polars as pl


def test_hard():
    import sillywalk  # noqa

    version = sillywalk.__version__
    assert version == "1.0.0a0", f"Expected version to be '1.0.0a0', got {version!r}"

    df = pl.read_csv("tests/students.csv").drop("Subject")

    model = sillywalk.PCAPredictor(df)

    f = {
        "Sex": 2,
        "Stature": 180,
        "Bodyweight": 65,
    }
    row = model.predict(f)
    values = np.array(list(row.values()))
    expected = np.array([2.0, 25.39551207, 180.0, 65.0, 39.53788392])
    np.testing.assert_array_almost_equal(values, expected, decimal=5)
