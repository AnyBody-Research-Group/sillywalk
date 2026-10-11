"""Offline demonstration, not measured human motion or a validated human model."""

import argparse
import json
import platform
from pathlib import Path

import narwhals
import numpy as np
import polars as pl
import sklearn

import sillywalk
from sillywalk import PCAPredictor, anybody

SEED = 2026
N_OBSERVATIONS = 48
N_SAMPLES = 128
N_MODES = 3
MEASURE = "DOF:Interface.Right.KneeFlexion.Pos[0]"


def run(output_dir: str | Path) -> dict[str, Path]:
    """Create and check a seeded synthetic workflow in ``output_dir``.

    Returns paths keyed by training_data, signals, coefficients, model,
    prediction, anyscript and report. Existing files with these names are
    overwritten. No downloads or proprietary applications are invoked.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_data": output_dir / "training_data.csv",
        "signals": output_dir / "signals.csv",
        "coefficients": output_dir / "fourier_coefficients.csv",
        "model": output_dir / "synthetic_model.npz",
        "prediction": output_dir / "prediction.json",
        "anyscript": output_dir / "predicted_motion.any",
        "report": output_dir / "report.json",
    }
    rng = np.random.default_rng(SEED)
    latent = rng.uniform(-1.0, 1.0, size=(N_OBSERVATIONS, 2))
    phase = np.arange(N_SAMPLES) / N_SAMPLES
    rows = []
    reference_signal = None
    reference_coefficients = None

    for i, (size, motion) in enumerate(latent):
        a0 = 0.4 + 0.025 * size
        a1 = 0.08 + 0.015 * motion
        a2 = 0.03 + 0.007 * size + 0.003 * motion
        b1 = 0.2 + 0.025 * size - 0.01 * motion
        b2 = 0.02 + 0.005 * motion
        signal = (
            a0
            + a1 * np.cos(2 * np.pi * phase)
            + a2 * np.cos(4 * np.pi * phase)
            + b1 * np.sin(2 * np.pi * phase)
            + b2 * np.sin(4 * np.pi * phase)
        )
        coefficients = anybody.compute_fourier_coefficients(
            pl.DataFrame({MEASURE: signal}), n_modes=N_MODES
        ).row(0, named=True)
        rows.append(
            {
                "Height": 1.72 + 0.08 * size,
                "Weight": 72.0 + 6.0 * size + 3.0 * motion,
                "freq": 1.2 + 0.1 * motion,
                **coefficients,
            }
        )
        if i == 0:
            reference_signal = signal
            reference_coefficients = np.array([a0, a1, a2, b1, b2])

    if reference_signal is None or reference_coefficients is None:
        raise RuntimeError("No synthetic observations were generated.")
    training = pl.DataFrame(rows)
    training.write_csv(paths["training_data"])
    pl.DataFrame({"phase": phase, "knee_angle_rad": reference_signal}).write_csv(
        paths["signals"]
    )
    coefficient_names = [f"{MEASURE}_a{j}" for j in range(N_MODES)] + [
        f"{MEASURE}_b{j}" for j in range(1, N_MODES)
    ]
    first_coefficients = training.select(coefficient_names).head(1)
    first_coefficients.write_csv(paths["coefficients"])
    computed = first_coefficients.to_numpy().flatten()
    np.testing.assert_allclose(computed, reference_coefficients, rtol=0, atol=1e-12)
    reconstructed = (
        computed[0]
        + computed[1] * np.cos(2 * np.pi * phase)
        + computed[2] * np.cos(4 * np.pi * phase)
        + computed[3] * np.sin(2 * np.pi * phase)
        + computed[4] * np.sin(4 * np.pi * phase)
    )
    np.testing.assert_allclose(reconstructed, reference_signal, rtol=0, atol=1e-12)

    # Two latent variables generate every column, so two components suffice.
    model = PCAPredictor(training, n_components=2)
    constraints = {"Height": 1.76}
    prediction = {k: float(v) for k, v in model.predict(constraints).items()}
    np.testing.assert_allclose(
        prediction["Height"], constraints["Height"], rtol=0, atol=1e-9
    )
    if not all(np.isfinite(list(prediction.values()))) or prediction["freq"] <= 0:
        raise RuntimeError("Prediction must be finite with a positive frequency.")
    model.export_pca_data(paths["model"])
    # Only reload the trusted model just written by this process.
    loaded = PCAPredictor.from_pca_data(str(paths["model"]))
    reloaded_prediction = loaded.predict(constraints)
    before = np.array([prediction[k] for k in model.columns])
    after = np.array([reloaded_prediction[k] for k in model.columns])
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-12)
    paths["prediction"].write_text(
        json.dumps(prediction, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    anybody.write_anyscript(prediction, targetfile=paths["anyscript"])
    rendered = paths["anyscript"].read_text(encoding="utf-8")
    if (
        "AnyKinEqFourierDriver" not in rendered
        or "Main.HumanModel.BodyModel.Interface.Right.KneeFlexion" not in rendered
    ):
        raise RuntimeError("The expected AnyScript driver was not rendered.")

    report = {
        "data_kind": "synthetic demonstration, not measured human data",
        "seed": SEED,
        "training_rows": N_OBSERVATIONS,
        "phase_samples": N_SAMPLES,
        "n_modes": N_MODES,
        "n_components": model.pca_n_components,
        "pca_columns": model.pca_columns,
        "excluded_columns": sorted(model.pca_low_variance_columns),
        "constraints": constraints,
        "constraint_max_abs_error": abs(prediction["Height"] - constraints["Height"]),
        "persistence_max_abs_error": float(np.max(np.abs(after - before))),
        "fourier_max_abs_error": float(
            np.max(np.abs(computed - reference_coefficients))
        ),
        "reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - reference_signal))
        ),
        "units": {
            "Height": "m",
            "Weight": "kg",
            "freq": "Hz (cycles/s)",
            "knee_angle": "rad",
            "phase": "dimensionless, [0, 1)",
        },
        "versions": {
            "python": platform.python_version(),
            "sillywalk": sillywalk.__version__,
            "numpy": np.__version__,
            "polars": pl.__version__,
            "narwhals": narwhals.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "anybody_executed": False,
        "verification_boundary": (
            "Checks cover synthetic coefficients, reconstruction, constrained "
            "prediction, trusted persistence and file rendering only. "
            "AnyScript was not loaded or executed in AnyBody."
        ),
    }
    paths["report"].write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return paths


def main() -> None:
    """Run the example from a source checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example-output"),
        help="Output directory (named example files are overwritten).",
    )
    args = parser.parse_args()
    paths = run(args.output_dir)
    print(f"Synthetic workflow complete. Report: {paths['report']}")
    print("AnyScript rendered only; AnyBody was not invoked.")


if __name__ == "__main__":
    main()
