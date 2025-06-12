import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import math

    import pandas as pd
    import polars as pl

    import sillywalk

    return math, pd, pl, sillywalk


@app.cell
def _(pl):
    df_polars = pl.read_csv("tests/students.csv").drop("Subject")
    df_polars
    return


@app.cell
def collection_of_tests(math, pd, pl, sillywalk):
    def test_input_polars():
        sillywalk.PCAPredictor(pl.read_csv("tests/students.csv"))

    def test_input_pandas():
        sillywalk.PCAPredictor(pd.read_csv("tests/students.csv"))

    def test_simple_constraints():
        df_polars = pl.read_csv("tests/students.csv").drop("Subject")
        model = sillywalk.PCAPredictor(df_polars)

        constraints = {
            "Sex": 2,
            "Stature": 180,
            "Bodyweight": 65,
        }
        result = model.predict(constraints)
        # Ensure constraints are met
        assert math.isclose(result["Sex"], constraints["Sex"])
        assert math.isclose(result["Stature"], constraints["Stature"])
        assert math.isclose(result["Bodyweight"], constraints["Bodyweight"])

        # Assert predicted values
        assert math.isclose(result["Age"], 25.39551, abs_tol=0.0001)
        assert math.isclose(result["Shoesize"], 39.53788, abs_tol=0.0001)

    return


if __name__ == "__main__":
    app.run()
