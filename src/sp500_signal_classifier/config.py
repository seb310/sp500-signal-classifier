from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params(path=ROOT / "params.yaml"):
    """
    Load parameters from a YAML configuration file.

    Parameters
    ----------
    path : Path, optional
        Path to the YAML file. Defaults to params.yaml in the project root.

    Returns
    -------
    dict
        Dictionary containing the loaded parameters.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def path(*parts):
    """
    Construct an absolute path relative to the project root directory.

    Parameters
    ----------
    *parts : str or Path
        Path components to join to the project root.

    Returns
    -------
    Path
        Absolute path object combining ROOT with the given parts.
    """
    return ROOT.joinpath(*parts)
