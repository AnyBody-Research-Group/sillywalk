import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import narwhals as nw
import numpy as np
from jinja2 import Environment, PackageLoader, Template
from narwhals.typing import IntoDataFrameT

# Matches keys like
#   "DOF:Main.HumanModel.BodyModel.Interface.Right.ShoulderArm.Jnt.ElbowFlexion.Pos[0]_a3"
# Capturing:
# - prefix: optional namespace ending with ':' (e.g. 'DOF:')
# - group:  full measure plus optional '.Pos[index]'
# - measure: path to the measure without the '.Pos[index]' suffix
# - index: optional integer index inside Pos[<index>]
# - coef:  coefficient name like 'a0', 'a1', 'b2', ...
FOURIER_DATA_RE = re.compile(
    r"(?P<prefix>.+:)?(?P<group>(?P<measure>.+?)(\.Pos\[(?P<index>\d+)\])?)_(?P<coef>[ab]\d+)"
)

jinja_env = Environment(loader=PackageLoader("sillywalk"))


def linear_correction(y: Iterable[float] | np.ndarray) -> np.ndarray:
    """Return a linearly corrected copy of y so the first and last become equal.

    The correction is a linear ramp added to the signal such that the first and
    last samples of the returned array are both equal to the average of the
    original endpoints, i.e. (y[0] + y[-1]) / 2.

    Parameters:
        y: 1D array-like signal.

    Returns:
        A new numpy array with the same length as ``y``.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return y.copy()
    correction = np.linspace(0.5 * (y[-1] - y[0]), 0.5 * (y[0] - y[-1]), y.size)
    return y + correction


def windowed_correction(
    y: Iterable[float] | np.ndarray, eps: float = 0.1
) -> np.ndarray:
    """Return a windowed linear correction so ends coincide while preserving middle.

    A linear end-to-end correction is multiplied by a smooth window that is
    non-zero only in the first and last ``eps`` fraction of the signal, so the
    interior is left unchanged.

    Parameters:
        y: 1D array-like signal.
        eps: Fraction (0<eps<=0.5) of samples at each end to blend correction.

    Returns:
        A new numpy array with the same length as ``y``.
    """
    y = np.asarray(y, dtype=float)
    N = y.size
    if N < 2:
        return y.copy()

    # Clamp eps to (0, 0.5] and compute a reasonable window length
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        return linear_correction(y)
    max_eps = 0.5
    eps = min(eps, max_eps)

    taper_len = max(1, min(int(round(N * eps)), N // 2))
    if 2 * taper_len >= N:
        # No middle segment remains; fall back to simple linear correction
        return linear_correction(y)

    # Build an inverted taper window that equals 1 at both ends and 0 in the
    # interior, using a quarter-sine taper over taper_len samples at each end.
    window = np.zeros(N, dtype=float)
    if taper_len == 1:
        window[0] = 1.0
        window[-1] = 1.0
    else:
        i = np.arange(taper_len, dtype=float)
        left = 1.0 - np.sin(0.5 * np.pi * (i / (taper_len - 1)))  # 1 -> 0
        right = 1.0 - np.sin(0.5 * np.pi * (1 - i / (taper_len - 1)))  # 0 -> 1
        window[:taper_len] = left
        window[-taper_len:] = right

    lin_correction = np.linspace(0.5 * (y[-1] - y[0]), 0.5 * (y[0] - y[-1]), N)
    correction = lin_correction * window
    return y + correction


def _anybody_fft(
    signal: Iterable[float] | np.ndarray, n_modes: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AnyBody-style Fourier coefficients for a real signal.

    Returns arrays (A, B) suitable for AnyBody's AnyKinEqFourierDriver with
    Type=CosSin. The scaling matches AnyBody's convention with ``a0`` divided
    by 2 and other coefficients scaled by 2/N.

    Parameters:
        signal: 1D real-valued signal.
        n_modes: Requested number of modes (unused in computation but kept for
                  compatibility).

    Returns:
        (A, B) where A[j] is the cosine coefficient a_j and B[j] the sine
        coefficient b_j. Note b_0 is not used in AnyBody (always 0).
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size == 0:
        return np.zeros(1), np.zeros(1)

    y = 2.0 * np.fft.rfft(x) / x.size
    # AnyBody's fourier implementation expects a0 to be divided by 2.
    y[0] /= 2.0
    return y.real, -y.imag


def compute_fourier_coefficients(
    df_native: IntoDataFrameT, n_modes: int = 6
) -> IntoDataFrameT:
    """Compute AnyBody Fourier coefficients for each column of a dataframe.

    For each numeric column, the coefficients ``a0, a1..a[n-1], b1..b[n-1]``
    are computed and returned as a one-row dataframe with columns named
    ``<col>_a0``, ``<col>_a1``, ..., ``<col>_b1``, ..., ``<col>_b<n-1>``.

    Parameters:
        df_native: Input dataframe (Pandas/Polars supported via narwhals).
        n_modes: Number of modes to output per signal (>=1). If the FFT of a
                 column yields fewer modes than requested, only the available
                 modes are returned.

    Returns:
        A dataframe of the same backend type as ``df_native`` with a single row.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    df = nw.from_native(df_native)
    out = df.select()

    for col in df.columns:
        a, b = _anybody_fft(df[col].to_numpy())
        max_modes = min(n_modes, a.size)
        out = out.with_columns(
            nw.lit(a[0]).alias(col + "_a0"),
            *[nw.lit(a[j]).alias(f"{col}_a{j}") for j in range(1, max_modes)],
            *[nw.lit(b[j]).alias(f"{col}_b{j}") for j in range(1, max_modes)],
        )

    return out.to_native()


def _add_new_coefficient(groupdata: dict, coef: str, val: float):
    """Insert/extend coefficient lists for a Fourier group with zero-fill.

    Parameters:
        groupdata: Dict containing keys 'a' and/or 'b' with list values.
        coef: String like 'a3' or 'b2'.
        val: Numeric value to insert.
    """
    coeftype = coef[0]
    coefficient_index = int(coef[1:])

    if len(groupdata[coeftype]) < coefficient_index + 1:
        # Extend the list to accommodate the new coefficient (zero-fill)
        groupdata[coeftype].extend(
            [0] * (coefficient_index + 1 - len(groupdata[coeftype]))
        )
    groupdata[coeftype][coefficient_index] = val


def _prepare_template_data(data: dict[str, float]) -> dict[str, Any]:
    """Transform a flat dict of values into structures for the Jinja template.

    Keys matching ``FOURIER_DATA_RE`` are grouped under ``fourier_data`` while
    all other keys go to ``scalar_data``.

    Returns:
        A dict with two keys: 'fourier_data' and 'scalar_data'.
    """
    templatedata: dict[str, dict[Any, Any]] = {
        "fourier_data": defaultdict(lambda: {}),
        "scalar_data": defaultdict(lambda: {}),
    }

    for key, val in data.items():
        match = FOURIER_DATA_RE.match(key)
        if match:
            mdict = match.groupdict()
            groupname = mdict["group"].removeprefix(
                "Main.HumanModel.BodyModel.Interface."
            )
            groupname = groupname.replace(".", "_").replace("[", "_").replace("]", "")
            coef = mdict["coef"]

            if groupname not in templatedata["fourier_data"]:
                templatedata["fourier_data"][groupname] = {
                    "prefix": mdict["prefix"] or "",
                    "measure": mdict["measure"],
                    "index": int(mdict["index"]) if mdict["index"] else None,
                    "a": [0],
                    "b": [0],
                }
            _add_new_coefficient(templatedata["fourier_data"][groupname], coef, val)
        else:
            templatedata["scalar_data"][key] = val

    return templatedata


def _guess_ammr_version(data: dict[str, float]) -> tuple | None:
    if (
        "DOF:Main.HumanModel.BodyModel.Trunk.Joints.Lumbar.SacrumPelvis.Pos[0]_a0"
        in data
    ):
        return (4, 0, 0)
    elif (
        "DOF:Main.HumanModel.BodyModel.Trunk.JointsLumbar.SacrumPelvisJnt.Pos[0]_a0"
        in data
    ):
        return (3, 0, 0)
    return None


def write_anyscript(
    data: dict[str, float],
    targetfile: str | Path | None = "trialdata.any",
    template_file: str | None = None,
    prepfunc=_prepare_template_data,
    create_human_model: bool = False,
    ammr_version: tuple | None = None,
):
    """Create an AnyBody include file from a dictionary of values.

    Any keys on the form "DOF:<measure>_<a|b>#" produce AnyKinEqFourierDriver
    entries using the 'CosSin' formulation.

    Parameters:
        data: Mapping from string keys to numeric values.
        targetfile: Path to write the rendered AnyScript include. If None then
                    return the rendered template as a string.
        template_file: Optional path to a Jinja template. If omitted, the
                       built-in template is used.
        prepfunc: Function to transform ``data`` to template input structure.
        ammr_version: Optional AMMR version tuple (major, minor, patch). This
                      is passed to the template, and may be used to customize
                      output for specific AMMR versions.
        create_human_model: If True, wrap output into a minimal Main model.
    """

    if ammr_version is None:
        ammr_version = _guess_ammr_version(data)

    # Load template
    if template_file is not None:
        template = Template(Path(template_file).read_text(encoding="utf-8"))
    else:
        template = jinja_env.get_template("model.any.jinja")

    template_data = prepfunc(data)
    template_data["create_human_model"] = create_human_model
    template_data["ammr_version"] = ammr_version

    # Ensure parent directory exists
    if targetfile is None:
        return template.render(**template_data)
    else:
        targetpath = Path(targetfile)
        targetpath.parent.mkdir(parents=True, exist_ok=True)

        with open(targetpath, "w", encoding="utf-8") as fh:
            fh.write(template.render(**template_data))
