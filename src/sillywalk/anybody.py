import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import narwhals as nw
import numpy as np
from jinja2 import Environment, PackageLoader, Template
from narwhals.typing import IntoDataFrameT

FOURIER_DATA_RE = re.compile(
    r"(?P<prefix>.+:)?(?P<group>(?P<measure>.+?)(.Pos\[(?P<index>\d)\])?)_(?P<coef>[ab]\d+)"
)

jinja_env = Environment(loader=PackageLoader("sillywalk"))


def _anybody_fft(signal, n_modes=6):
    """Calculate the fourier coefficients for a given signal.
    Returns a (A,B) with values which can be used with AnyBody "AnyKinEqFourierDriver" class and the Type=CosSin setting.
    """
    y = 2 * np.fft.rfft(signal) / signal.size
    # AnyBody's fourier implementation expect a0 to be divided by 2.
    y[0] /= 2
    return y.real, -y.imag


def compute_fourier_coefficients(
    df_native: IntoDataFrameT, n_modes=6
) -> IntoDataFrameT:
    """Split a dataframe with time series into its 'AnyBody' fourier
    coefficients. I.e. A,B coefficients which can be used with the 'CosSin'
    formulation of the AnyBody's `AnyKinEqFourierDriver` class.

    For each column in the dataframe the 2*n_modes-1 coefficients are
    calculated and returned as new columns in a dataframe with just a single
    row. The new columns are postfixed with `_a0`, `_a1`, ..., `_a<n>` and `_b1`, ..., `_b<n>`.

    The `_b0` coefficient is not included, as it is always 0.

    Parameters:
        df_native: The input dataframe containing the signals.

        n_modes: The number of modes to calculate. Defaults to 6 (meaning 2*6-1=11 coefficients per column).

    Returns:
        A new dataframe with the fourier coefficients.
    """
    df = nw.from_native(df_native)
    out = df.select()

    for col in df.columns:
        a, b = _anybody_fft(df[col].to_numpy(), n_modes)
        out = out.with_columns(
            nw.lit(a[0]).alias(col + "_a0"),
            *[nw.lit(a[j]).alias(col + "_a" + str(j)) for j in range(1, n_modes)],
            *[nw.lit(b[j]).alias(col + "_b" + str(j)) for j in range(1, n_modes)],
        )

    return out.to_native()


def _add_new_coefficient(groupdata: dict, coef: str, val: float):
    coeftype = coef[0]
    coefficient_index = int(coef[1:])

    if len(groupdata[coeftype]) < coefficient_index + 1:
        # Extend the list to accommodate the new coefficient
        groupdata[coeftype].extend(
            [-1] * (coefficient_index + 1 - len(groupdata[coeftype]))
        )
    groupdata[coeftype][coefficient_index] = val


def _prepare_template_data(data: dict[str, float]) -> dict[str, Any]:
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


def create_model_file(
    data: dict[str, float],
    targetfile="trialdata.any",
    template_file: str | None = None,
    prepfunc=_prepare_template_data,
    create_human_model: bool = False,
):
    """Create an AnyBody include file from the data dictionary.

    Any keys on the form "DOF:<measure>_<a|b>#", will be created as `AnyKinEqFourierDriver` entries.


    """

    if template_file is not None:
        template = Template(Path(template_file).read_text())
    else:
        template = jinja_env.get_template("model.any.jinja")

    template_data = prepfunc(data)
    template_data["create_human_model"] = create_human_model
    with open(targetfile, "w+") as fh:
        fh.write(template.render(**template_data))
