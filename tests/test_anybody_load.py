"""Opt-in real console loading, not kinematics or inverse-dynamics validation."""

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sillywalk import PCAPredictor, anybody

pytestmark = pytest.mark.anybody


def test_predicted_standalone_model_loads_in_anybody(tmp_path):
    ammr_setting = os.environ.get("AMMR_PATH")
    assert ammr_setting, (
        "Set AMMR_PATH to a licensed AMMR 4 checkout containing libdef.any."
    )
    ammr = Path(ammr_setting).resolve()
    if not (ammr / "libdef.any").is_file():
        pytest.fail(f"AMMR_PATH does not contain libdef.any: {ammr}")

    console = Path(sys.prefix) / "AnyBodyCon" / "AnyBodyCon.exe"
    console_setting = os.environ.get("ANYBODYCON_PATH")
    if console_setting:
        console = Path(console_setting).resolve()
    if not console.is_file():
        pytest.fail(
            "AnyBody Console is missing. Use the Windows 'anybody' pixi environment "
            "or set ANYBODYCON_PATH to the real AnyBodyCon.exe executable."
        )

    anypytools = importlib.import_module("anypytools")
    commands = importlib.import_module("anypytools.macro_commands")
    tools = importlib.import_module("anypytools.tools")
    version = tools.get_ammr_version(str(ammr))
    if not version.startswith("4."):
        pytest.fail(
            f"The integration test requires AMMR 4, found {version!r} at {ammr}."
        )

    output_dir = Path(os.environ.get("SILLYWALK_ANYBODY_LOG_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "predicted.main.any"
    (output_dir / "libdef.any").write_text(
        f'#include "{ammr / "libdef.any"}"\n', encoding="utf-8"
    )
    measure = "DOF:Interface.Right.KneeFlexion.Pos[0]"
    training = pl.DataFrame(
        {
            "Height": [1.60, 1.68, 1.76, 1.84],
            "Weight": [60.0, 68.0, 76.0, 84.0],
            "freq": [1.0, 1.1, 1.2, 1.3],
            f"{measure}_a0": [0.2, 0.3, 0.4, 0.5],
            f"{measure}_a1": [0.08] * 4,
            f"{measure}_b1": [0.2] * 4,
        }
    )
    prediction = PCAPredictor(training, n_components=1).predict({"Height": 1.72})
    anybody.write_anyscript(
        prediction,
        targetfile=target,
        create_human_model=True,
        ammr_version=(4, 0, 0),
    )
    process = anypytools.AnyPyProcess(
        anybodycon_path=str(console),
        num_processes=1,
        timeout=180,
        keep_logfiles=True,
        silent=True,
    )
    variables = [
        "Main.Study.nStep",
        "Main.HumanModel.Anthropometrics.BodyHeight",
        "Main.HumanModel.Anthropometrics.BodyMass",
        "Main.PCA_drivers.Period",
        "Main.PCA_drivers.JointsAndDrivers.Interface_Right_KneeFlexion_Pos_0.A",
        "Main.PCA_drivers.JointsAndDrivers.Interface_Right_KneeFlexion_Pos_0.B",
    ]
    logfile = output_dir / "anybody-load.log"
    try:
        output = process.start_macro(
            [commands.Load(str(target)), *[commands.Dump(name) for name in variables]],
            folderlist=[str(output_dir)],
            logfile=str(logfile),
        )
    finally:
        # Console diagnostics may include the license endpoint or password.
        if logfile.is_file():
            log_bytes = logfile.read_bytes()
            for name in ("RLM_LICENSE", "RLM_LICENSE_PASSWORD"):
                for encoding in ("utf-8", "utf-16le"):
                    value = os.environ.get(name, "").encode(encoding)
                    if value:
                        log_bytes = log_bytes.replace(
                            value, "[redacted]".encode(encoding)
                        )
            logfile.write_bytes(log_bytes)
    assert len(output) == 1, f"Expected one console task, got {len(output)}."
    result = output[0]
    log_text = logfile.read_text(encoding="utf-8")
    console_version = re.search(r"version\s*:\s*([0-9. ]+)", log_text)
    console_build = re.search(r"Build\s*:\s*([0-9.]+)", log_text)
    assert console_version is not None and console_build is not None, (
        "Console version/build evidence is missing from the local log."
    )
    ammr_commit = (
        subprocess.run(
            ["git", "-C", str(ammr), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if (ammr / ".git").exists()
        else None
    )
    dumped = {}
    for name in variables:
        if name in result:
            value = np.asarray(result[name])
            if value.dtype.kind in "iuf" and np.all(np.isfinite(value)):
                dumped[name] = value.tolist()
    report = {
        "anypytools": anypytools.__version__,
        "console_version": console_version[1].replace(" ", ""),
        "console_build": console_build[1],
        "ammr_version": version,
        "ammr_commit": ammr_commit,
        "error_count": len(result.get("ERROR", [])),
        "warning_count": len(result.get("WARNING", [])),
        "dumped": dumped,
        "dump_validation_succeeded": False,
        "boundary": "Load and Dump only; no kinematics or inverse dynamics.",
    }
    report_path = output_dir / "loader-result.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if "ERROR" in result:
        pytest.fail(
            f"AnyBody failed to load the generated model. See the local log: {logfile}."
        )
    assert all(name in dumped for name in variables), (
        f"Loaded-object Dump evidence is missing or not finite numeric data. See {logfile}."
    )
    np.testing.assert_allclose(
        [result[name] for name in variables[:4]],
        [200, 1.72, 72.0, 1 / 1.15],
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(result[variables[4]], [[0.35, 0.08]], rtol=0, atol=1e-9)
    np.testing.assert_allclose(result[variables[5]], [[0, 0.2]], rtol=0, atol=1e-9)
    report["dump_validation_succeeded"] = True
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
