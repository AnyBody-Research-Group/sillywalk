import re
from collections import defaultdict

from jinja2 import Environment, PackageLoader

FOURIER_DRIVER_RE = re.compile(
    r"(?P<group>(?P<measure>.+?)(_(?P<index>\d))?)_(?P<coef>[ab]\d+)"
)

jinja_env = Environment(loader=PackageLoader("sillywalk", "templates"))

preprocess_templ_data = {}


def register_template(key):
    def decorator(function):
        preprocess_templ_data[key] = function
        return function

    return decorator


@register_template("fourier-model")
def _prepare_data_fourier_model(data):
    """Prepare template data for the fourier model template"""

    templatedata: dict[str, dict] = {
        "Drivers": defaultdict(lambda: {}),
        "TimeSeries": defaultdict(lambda: {}),
        "SegmentDimensions": {},
        "Other": {},
    }

    for key, val in data.items():
        match = FOURIER_DRIVER_RE.match(key)
        if match:
            elem = {}
            mdict = match.groupdict()
            groupname = mdict["group"].replace(".", "_")
            elem["measure"] = mdict["measure"]
            elem["index"] = int(mdict["index"] or 0)
            elem[mdict["coef"]] = val
            if groupname.startswith(("Trunk", "Interface")):
                templatedata["Drivers"][groupname].update(elem)
            else:
                templatedata["TimeSeries"][groupname].update(elem)
        elif ".SegmentDimensions." in key:
            _, _, key = key.partition(".SegmentDimensions.")
            templatedata["SegmentDimensions"][key] = val
        else:
            templatedata["Other"][key] = val

    return templatedata


@register_template("anyman")
def _prepare_data_anyman_file(data):
    """prepare template data to create a AMS anyman file to use with ScalingXYZ"""

    templatedata: dict[str, dict] = {
        "SegmentDimensions": {},
        "Anthropometrics": {},
    }

    for key, val in data.items():
        if ".Anthropometrics." in key:
            _, _, key = key.partition(".Anthropometrics.")
            if "SegmentDimensions." in key:
                _, _, key = key.partition("SegmentDimensions.")
                templatedata["SegmentDimensions"][key] = val
            else:
                templatedata["Anthropometrics"][key] = val

    return templatedata


def write_template(data, targetfile="trialdata.any", model="fourier-model"):
    template = jinja_env.get_template(f"{model}.any")
    template_data = preprocess_templ_data[model](data)

    with open(targetfile, "w+") as fh:
        fh.write(template.render(**template_data))
