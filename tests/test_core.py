import polars as pl

def test_hard():
    import sillywalk  # noqa

    version = sillywalk.__version__
    assert version == "unknown" or isinstance(version, str), f"Expected version to be 'unknown' or a string, got {version!r}"

    df = pl.read_csv("tests/students.csv")

    model = sillywalk.PCAPredictor(df)

    f = {
        'Sex': 2,
        'Stature': 180,
        'Bodyweight': 65,
    }
    row = model.predict(f)
    print(row)

