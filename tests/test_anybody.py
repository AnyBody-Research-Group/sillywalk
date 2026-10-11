import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import sillywalk.anybody as anybody


def test_linear_correction_makes_ends_equal():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    y2 = anybody.linear_correction(y)
    assert np.isclose(y2[0], y2[-1])
    assert np.isclose(y2[0], 0.5 * (y[0] + y[-1]))
    assert y2.shape == y.shape


def test_windowed_correction_preserves_middle_and_fixes_ends():
    N = 100
    y = np.linspace(0.0, 10.0, N)
    eps = 0.1
    y2 = np.asarray(anybody.windowed_correction(y, eps=eps))

    # Ends should be equal
    assert np.isclose(y2[0], y2[-1])
    # Middle region should be almost unchanged
    wl = int(round(N * eps))
    mid = slice(wl, N - wl)
    assert np.allclose(y2[mid], y[mid])


def test_anybody_fft_cosine_and_sine_components():
    N = 1000
    n = np.arange(N)
    k = 3
    cos_sig = np.cos(2 * np.pi * k * n / N)
    sin_sig = np.sin(2 * np.pi * k * n / N)

    a_cos, b_cos = anybody._anybody_fft(cos_sig)
    assert np.isclose(a_cos[k], 1.0, atol=1e-6)
    assert np.isclose(b_cos[k], 0.0, atol=1e-6)

    a_sin, b_sin = anybody._anybody_fft(sin_sig)
    assert np.isclose(a_sin[k], 0.0, atol=1e-6)
    assert np.isclose(b_sin[k], 1.0, atol=1e-6)


def test_compute_fourier_coefficients_dataframe():
    # Build a simple dataframe with one cosine column
    N = 200
    idx = np.arange(N)
    k = 2
    cos_sig = np.cos(2 * np.pi * k * idx / N)
    df = pl.DataFrame({"sig": cos_sig})

    coeffs = anybody.compute_fourier_coefficients(df, n_modes=6)
    assert isinstance(coeffs, pl.DataFrame)
    assert coeffs.shape == (1, 11)
    # Expect columns sig_a0, sig_a1..a5, sig_b1..b5
    expected_cols = (
        ["sig_a0"]
        + [f"sig_a{j}" for j in range(1, 6)]
        + [f"sig_b{j}" for j in range(1, 6)]
    )
    for c in expected_cols:
        assert c in coeffs.columns
    # Check dominant cosine at k=2
    assert math.isclose(coeffs[0, f"sig_a{k}"], 1.0, rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.parametrize(
    "backend", [pl.DataFrame, pd.DataFrame], ids=["polars", "pandas"]
)
def test_fourier_result_has_one_row_and_preserves_backend(backend):
    phase = np.arange(128) / 128
    signal = (
        0.4
        + 0.8 * np.cos(2 * np.pi * phase)
        - 0.3 * np.cos(4 * np.pi * phase)
        + 0.6 * np.sin(2 * np.pi * phase)
        + 0.2 * np.sin(4 * np.pi * phase)
    )
    data = backend({"sig": signal, "shifted": 2.0 * signal + 1.0})
    coefficients = anybody.compute_fourier_coefficients(data, n_modes=3)

    assert isinstance(coefficients, backend)
    assert coefficients.shape == (1, 10)
    assert list(coefficients.columns) == [
        "sig_a0",
        "sig_a1",
        "sig_a2",
        "sig_b1",
        "sig_b2",
        "shifted_a0",
        "shifted_a1",
        "shifted_a2",
        "shifted_b1",
        "shifted_b2",
    ]
    np.testing.assert_allclose(
        coefficients.to_numpy(),
        [[0.4, 0.8, -0.3, 0.6, 0.2, 1.8, 1.6, -0.6, 1.2, 0.4]],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "backend", [pl.DataFrame, pd.DataFrame], ids=["polars", "pandas"]
)
@pytest.mark.parametrize("signal", [[], [2.5]], ids=["empty-signal", "single-sample"])
def test_fourier_result_keeps_one_row_with_only_available_modes(backend, signal):
    coefficients = anybody.compute_fourier_coefficients(
        backend({"sig": signal}), n_modes=6
    )

    assert isinstance(coefficients, backend)
    assert coefficients.shape == (1, 1)
    assert list(coefficients.columns) == ["sig_a0"]
    np.testing.assert_allclose(
        coefficients.to_numpy(), [[2.5 if signal else 0.0]], rtol=0, atol=1e-12
    )


def test_prepare_template_data_and_coefficient_insertion():
    data = {
        "omega": 2.0 * np.pi,
        "DOF:Main.HumanModel.BodyModel.Interface.Right.ShoulderArm.Jnt.ElbowFlexion.Pos[0]_a1": 0.3,
        "DOF:Main.HumanModel.BodyModel.Interface.Right.ShoulderArm.Jnt.ElbowFlexion.Pos[0]_b2": -0.1,
    }
    tpl = anybody._prepare_template_data(data)

    # Scalar data propagated
    assert tpl["scalar_data"]["omega"] == data["omega"]
    # Fourier data grouped and renamed
    # Group key should have dots and brackets replaced with underscores
    key = "Right_ShoulderArm_Jnt_ElbowFlexion_Pos_0"
    assert key in tpl["fourier_data"]
    grp = tpl["fourier_data"][key]
    assert grp["prefix"] == "DOF:"
    assert grp["measure"].endswith("ShoulderArm.Jnt.ElbowFlexion")
    assert grp["index"] == 0
    assert grp["a"][1] == 0.3
    assert grp["b"][2] == -0.1
    # Zero-filled a[0] and b[0]
    assert grp["a"][0] == 0
    assert grp["b"][0] == 0


def test_write_anyscript_renders_driver_block(tmp_path: Path):
    data = {
        "omega": 2.0 * np.pi,
        "DOF:Main.HumanModel.BodyModel.Interface.Right.ShoulderArm.Jnt.ElbowFlexion.Pos[0]_a0": 0.5,
        "DOF:Main.HumanModel.BodyModel.Interface.Right.ShoulderArm.Jnt.ElbowFlexion.Pos[0]_a1": 0.3,
        "DOF:Main.HumanModel.BodyModel.Interface.Right.ShoulderArm.Jnt.ElbowFlexion.Pos[0]_b1": -0.2,
    }
    outfile = tmp_path / "trialdata.any"
    anybody.write_anyscript(data, targetfile=outfile)

    text = outfile.read_text(encoding="utf-8")
    assert "AnyKinEqFourierDriver" in text
    assert "Right_ShoulderArm_Jnt_ElbowFlexion_Pos_0" in text
    assert "Freq = ..Freq;" in text or "Freq =" in text
