"""Utility methods for ModelMimic.
"""
import toml
from pathlib import Path

def read_config(cfg_path : Path):
    """_summary_

    Parameters
    ----------
    cfg_path : `pathlib.Path`
        Path to configuration file.

    """
    with open(cfg_path, encoding="utf-8", mode="r") as _cin:
        config = toml.loads(_cin.read())

    return config