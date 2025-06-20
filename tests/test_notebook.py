import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import math

    import pandas as pd
    import polars as pl
    import pytest

    import sillywalk

    return math, pd, pl, pytest, sillywalk


@app.cell
def _(pl):
    # Create a test dataset with single zero variance column
    df_students = (
        pl.read_csv("tests/students.csv")
        .drop("Subject")
        .with_columns(
            pl.lit(1.0).alias("Zero Variance Test"),
            pl.lit("String dummy input").alias("String Column"),
        )
    )
    df_students
    return (df_students,)


@app.cell
def collection_of_tests(df_students, math, pd, pl, pytest, sillywalk):
    def test_input_polars():
        sillywalk.PCAPredictor(pl.read_csv("tests/students.csv"))

    def test_input_pandas():
        sillywalk.PCAPredictor(pd.read_csv("tests/students.csv"))

    def test_simple_constraints():
        model = sillywalk.PCAPredictor(df_students)
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

    def test_means_and_zero_pca_components():
        """Test that PCA components for mean values are zero."""
        model = sillywalk.PCAPredictor(df_students)
        components = model.parameters_to_components(df_students.mean())
        assert components == [0.0] * model.pca_n_components

    def test_roundtrip_pca_components():
        """Test that converting parameters to PCA components and back gives the original parameters."""
        model = sillywalk.PCAPredictor(df_students)
        # Select data from a single student and check roundtrip
        param_in = df_students.to_dicts()[3]
        pca_components = model.parameters_to_components(param_in)
        param_out = model.components_to_parameters(pca_components)

        # Input string values will be predicted as NaN
        assert param_in.pop("String Column") == "String dummy input"
        assert param_out.pop("String Column") == pytest.approx(
            float("nan"), nan_ok=True
        )

        assert param_in == pytest.approx(param_out)

    def test_initialize_model_from_pca_values():
        """Test that the model can be initialized from a subset"""
        model1 = sillywalk.PCAPredictor(df_students)
        model2 = sillywalk.PCAPredictor.from_pca_values(
            means=model1.means,
            stds=model1.stds,
            columns=model1.columns,
            pca_columns=model1.pca_columns,
            pca_eigenvectors=model1.pca_eigenvectors,
            pca_eigenvalues=model1.pca_eigenvalues,
        )
        constr = {"Age": 26, "Shoesize": 40}
        assert model1.predict(constr) == pytest.approx(
            model2.predict(constr), nan_ok=True
        )

    def test_save_reload_model():
        model1 = sillywalk.PCAPredictor(df_students)
        model1.save_npz("model1.npz")
        model2 = sillywalk.PCAPredictor.from_npz("model1.npz")
        constr = {"Age": 24, "Shoesize": 43}

        assert model1.predict(constr) == pytest.approx(
            model2.predict(constr), nan_ok=True
        )

    return


@app.cell
def _(pl, sillywalk):
    data = pl.read_csv("tests/Fourier.csv")

    # Select all numeric columns
    # data.select(pl.selectors.numeric())
    model_gait = sillywalk.PCAPredictor(data)

    model_gait.save_npz("gaitdata.npz")
    print(len(model_gait.pca_columns))
    print(len(model_gait.columns))

    model_gait.columns
    return (data,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(data):
    data
    return


if __name__ == "__main__":
    app.run()
