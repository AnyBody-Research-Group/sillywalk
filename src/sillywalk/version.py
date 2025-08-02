from importlib.metadata import version

__version__ = version(__package__ or "")

if __version__.startswith("0.0.1"):
    # Probably started in development mode,
    # so we need to extract the version manually
    import pathlib
    import re

    pixi_manifest = pathlib.Path(__file__).parents[2] / "pixi.toml"
    if pixi_manifest.exists():
        content = pixi_manifest.read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            __version__ = match.group(1)
