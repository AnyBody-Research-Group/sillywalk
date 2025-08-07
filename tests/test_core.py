import math

import numpy as np
import polars as pl


def test_hard():
    import sillywalk  # noqa

    version = sillywalk.__version__
    assert version == "1.0.0a1", f"Expected version to be '1.0.0a1', got {version!r}"

    df = pl.read_csv("tests/students.csv").drop("Subject")

    model = sillywalk.PCAPredictor(df)

    constraints = {
        "Sex": 2,
        "Stature": 180,
        "Bodyweight": 65,
    }
    result = model.predict(constraints)
    for key, value in constraints.items():
        assert math.isclose(result[key], value), (
            f"Expected {key} to be {value}, got {result[key]}"
        )

    np.testing.assert_array_almost_equal(result["Age"], 25.39551, decimal=5)
    np.testing.assert_array_almost_equal(result["Shoesize"], 39.53788392, decimal=5)
