import importlib.util
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sillywalk import PCAPredictor

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_workflow.py"
MEASURE = "DOF:Interface.Right.KneeFlexion.Pos[0]"
COEFFICIENT_COLUMNS = [
    f"{MEASURE}_a0",
    f"{MEASURE}_a1",
    f"{MEASURE}_a2",
    f"{MEASURE}_b1",
    f"{MEASURE}_b2",
]
TRAINING_COLUMNS = ["Height", "Weight", "freq", *COEFFICIENT_COLUMNS]
ARTIFACT_NAMES = {
    "training_data": "training_data.csv",
    "signals": "signals.csv",
    "coefficients": "fourier_coefficients.csv",
    "model": "synthetic_model.npz",
    "prediction": "prediction.json",
    "anyscript": "predicted_motion.any",
    "report": "report.json",
}


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    def refuse_network(*args, **kwargs):
        raise AssertionError("The synthetic workflow must not use the network.")

    def refuse_process(*args, **kwargs):
        raise AssertionError(
            "The synthetic workflow must not start external processes."
        )

    monkeypatch.setattr(socket, "create_connection", refuse_network)
    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_network)
    monkeypatch.setattr(subprocess, "run", refuse_process)
    monkeypatch.setattr(subprocess, "Popen", refuse_process)
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("synthetic_workflow", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert list(tmp_path.iterdir()) == []
    return module


@pytest.fixture
def outputs(workflow, tmp_path):
    return workflow.run(tmp_path / "generated")


def reference_training():
    latent = np.random.default_rng(2026).uniform(-1.0, 1.0, size=(48, 2))
    size, motion = latent.T
    values = np.column_stack(
        [
            1.72 + 0.08 * size,
            72.0 + 6.0 * size + 3.0 * motion,
            1.2 + 0.1 * motion,
            0.4 + 0.025 * size,
            0.08 + 0.015 * motion,
            0.03 + 0.007 * size + 0.003 * motion,
            0.2 + 0.025 * size - 0.01 * motion,
            0.02 + 0.005 * motion,
        ]
    )
    return values


def test_workflow_returns_exact_artifacts_and_seeded_training(outputs):
    assert set(outputs) == set(ARTIFACT_NAMES)
    for key, filename in ARTIFACT_NAMES.items():
        path = outputs[key]
        assert isinstance(path, Path)
        assert path.name == filename
        assert path.is_file()
        assert path.stat().st_size > 0
    assert {p.name for p in outputs["report"].parent.iterdir()} == set(
        ARTIFACT_NAMES.values()
    )

    training = pl.read_csv(outputs["training_data"])
    assert training.shape == (48, 8)
    assert training.columns == TRAINING_COLUMNS
    np.testing.assert_allclose(
        training.to_numpy(), reference_training(), rtol=0, atol=1e-12
    )


def test_prediction_constraints_and_persisted_two_component_model(outputs):
    training = pl.read_csv(outputs["training_data"])
    prediction = json.loads(outputs["prediction"].read_text(encoding="utf-8"))
    assert set(prediction) == set(TRAINING_COLUMNS)
    assert all(np.isfinite(list(prediction.values())))
    assert prediction["Height"] == pytest.approx(1.76, rel=0, abs=1e-9)
    assert prediction["freq"] > 0

    fitted = PCAPredictor(training, n_components=2)
    loaded = PCAPredictor.from_pca_data(str(outputs["model"]))
    assert loaded.columns == fitted.columns == TRAINING_COLUMNS
    assert loaded.pca_columns == fitted.pca_columns == TRAINING_COLUMNS
    assert loaded.pca_n_components == fitted.pca_n_components == 2
    assert loaded.pca_low_variance_columns == set()
    np.testing.assert_allclose(loaded.means, fitted.means, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        loaded.pca_explained_variance_ratio,
        fitted.pca_explained_variance_ratio,
        rtol=0,
        atol=1e-12,
    )

    for constraints in ({"Height": 1.76}, {"Height": 1.69, "freq": 1.24}):
        before = fitted.predict(constraints)
        after = loaded.predict(constraints)
        np.testing.assert_allclose(
            [after[k] for k in TRAINING_COLUMNS],
            [before[k] for k in TRAINING_COLUMNS],
            rtol=0,
            atol=1e-12,
        )
        for key, value in constraints.items():
            assert after[key] == pytest.approx(value, rel=0, abs=1e-9)
        if constraints == {"Height": 1.76}:
            np.testing.assert_allclose(
                [prediction[k] for k in TRAINING_COLUMNS],
                [after[k] for k in TRAINING_COLUMNS],
                rtol=0,
                atol=1e-12,
            )


def test_fourier_amplitudes_and_phase_reconstruction(outputs):
    coefficients = pl.read_csv(outputs["coefficients"])
    assert coefficients.shape == (1, 5)
    assert coefficients.columns == COEFFICIENT_COLUMNS
    expected = reference_training()[0, 3:]
    actual = coefficients.to_numpy().flatten()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    signals = pl.read_csv(outputs["signals"])
    assert signals.shape == (128, 2)
    assert signals.columns == ["phase", "knee_angle_rad"]
    phase = np.arange(128) / 128
    np.testing.assert_allclose(signals["phase"].to_numpy(), phase, rtol=0, atol=0)
    assert signals["phase"].to_numpy().max() < 1
    reconstructed = (
        actual[0]
        + actual[1] * np.cos(2 * np.pi * phase)
        + actual[2] * np.cos(4 * np.pi * phase)
        + actual[3] * np.sin(2 * np.pi * phase)
        + actual[4] * np.sin(4 * np.pi * phase)
    )
    np.testing.assert_allclose(
        reconstructed, signals["knee_angle_rad"].to_numpy(), rtol=0, atol=1e-12
    )


def test_anyscript_has_matching_frequency_measure_and_coefficients(outputs):
    text = outputs["anyscript"].read_text(encoding="utf-8")
    prediction = json.loads(outputs["prediction"].read_text(encoding="utf-8"))
    assert text.count("AnyKinEqFourierDriver") == 1
    assert "AnyKinEqFourierDriver Interface_Right_KneeFlexion_Pos_0" in text
    assert "Type = CosSin;" in text
    assert "Freq = 1/..Period;" in text
    assert "CType = {Hard};" in text
    assert "Reaction.Type = {Off};" in text
    assert "MeasureOrganizer = {0};" in text
    assert (
        "AnyKinMeasure &m = Main.HumanModel.BodyModel.Interface.Right.KneeFlexion;"
        in text
    )
    assert "Main.HumanModel.BodyModel.Main.HumanModel.BodyModel" not in text
    assert not re.search(r"\b(?:nan|inf)\b", text, flags=re.IGNORECASE)
    period = re.search(r"AnyVar Period\s*=\s*DesignVar\(1/([^)]+)\);", text)
    assert period is not None
    assert float(period[1]) == pytest.approx(prediction["freq"], rel=0, abs=1e-12)
    assert float(period[1]) > 0

    cosine = re.search(r"\bA\s*=\s*\{\{(.*?)\}\};", text, flags=re.DOTALL)
    sine = re.search(r"\bB\s*=\s*\{\{(.*?)\}\};", text, flags=re.DOTALL)
    assert cosine is not None and sine is not None
    a_fields = [v.strip() for v in cosine[1].split(",") if v.strip()]
    b_fields = [v.strip() for v in sine[1].split(",") if v.strip()]
    assert len(a_fields) == len(b_fields) == 3
    assert a_fields[0].endswith(" + a0_offset")
    a_values = [
        float(a_fields[0].removesuffix(" + a0_offset")),
        *[float(v) for v in a_fields[1:]],
    ]
    b_values = [float(v) for v in b_fields]
    np.testing.assert_allclose(
        a_values,
        [prediction[f"{MEASURE}_a{j}"] for j in range(3)],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        b_values,
        [0, prediction[f"{MEASURE}_b1"], prediction[f"{MEASURE}_b2"]],
        rtol=0,
        atol=1e-12,
    )


def test_report_records_schema_tolerances_versions_and_verification_boundary(outputs):
    report = json.loads(outputs["report"].read_text(encoding="utf-8"))
    assert set(report) == {
        "data_kind",
        "seed",
        "training_rows",
        "phase_samples",
        "n_modes",
        "n_components",
        "pca_columns",
        "excluded_columns",
        "constraints",
        "constraint_max_abs_error",
        "persistence_max_abs_error",
        "fourier_max_abs_error",
        "reconstruction_max_abs_error",
        "units",
        "versions",
        "anybody_executed",
        "verification_boundary",
    }
    assert report["data_kind"] == "synthetic demonstration, not measured human data"
    assert report["seed"] == 2026
    assert report["training_rows"] == 48
    assert report["phase_samples"] == 128
    assert report["n_modes"] == 3
    assert report["n_components"] == 2
    assert report["pca_columns"] == TRAINING_COLUMNS
    assert report["excluded_columns"] == []
    assert report["constraints"] == {"Height": 1.76}
    for key, tolerance in (
        ("constraint_max_abs_error", 1e-9),
        ("persistence_max_abs_error", 1e-12),
        ("fourier_max_abs_error", 1e-12),
        ("reconstruction_max_abs_error", 1e-12),
    ):
        assert np.isfinite(report[key])
        assert 0 <= report[key] <= tolerance
    assert report["units"] == {
        "Height": "m",
        "Weight": "kg",
        "freq": "Hz (cycles/s)",
        "knee_angle": "rad",
        "phase": "dimensionless, [0, 1)",
    }
    assert set(report["versions"]) == {
        "python",
        "sillywalk",
        "numpy",
        "polars",
        "narwhals",
        "scikit-learn",
    }
    assert all(
        isinstance(version, str) and version for version in report["versions"].values()
    )
    assert report["anybody_executed"] is False
    assert "not loaded or executed in AnyBody" in report["verification_boundary"]


def test_repeated_runs_reproduce_text_artifacts(workflow, outputs, tmp_path):
    repeated = workflow.run(str(tmp_path / "repeated"))
    for key in set(ARTIFACT_NAMES) - {"model"}:
        assert repeated[key].read_bytes() == outputs[key].read_bytes()


def test_cli_runs_outside_repository_without_writing_to_checkout(tmp_path):
    target = tmp_path / "cli-output"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--output-dir", str(target)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert {path.name for path in target.iterdir()} == set(ARTIFACT_NAMES.values())
    assert {path.name for path in tmp_path.iterdir()} == {"cli-output"}
    assert "Synthetic workflow complete. Report:" in result.stdout
    assert "AnyScript rendered only; AnyBody was not invoked." in result.stdout
    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert report["anybody_executed"] is False
