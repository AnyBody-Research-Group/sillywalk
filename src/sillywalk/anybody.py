import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, PackageLoader, Template

FOURIER_DATA_RE = re.compile(
    r"(?P<prefix>.+:)?(?P<group>(?P<measure>.+?)(.Pos\[(?P<index>\d)\])?)_(?P<coef>[ab]\d+)"
)

jinja_env = Environment(loader=PackageLoader("sillywalk"))


def calc_anybody_fourier_coefficients(signal, n_modes=6):
    """Calculate the fourier coefficients for a given signal.
    Returns values which can be used with AnyBody "AnyKinEqFourierDriver" class and the Type=CosSin setting.
    """
    y = 2 * np.fft.rfft(signal) / signal.size
    # AnyBody's fourier implementation expect a0 to be divided by 2.
    a = y[0:n_modes].real
    b = -y[0:n_modes].imag
    a[0] /= 2  # Adjust a0 to match AnyBody's definition
    return a, b


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
):
    if template_file is not None:
        template = Template(Path(template_file).read_text())
    else:
        template = jinja_env.get_template("model.any.jinja")

    template_data = prepfunc(data)
    with open(targetfile, "w+") as fh:
        fh.write(template.render(**template_data))
