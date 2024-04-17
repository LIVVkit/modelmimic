"""Utility methods for ModelMimic.
"""

from pathlib import Path

import toml


def read_config(cfg_path: Path):
    """
    Read a TOML configuration file.

    Parameters
    ----------
    cfg_path : `pathlib.Path`
        Path to configuration file.

    """
    with open(cfg_path, encoding="utf-8", mode="r") as _cin:
        config = toml.loads(_cin.read())

    return config
