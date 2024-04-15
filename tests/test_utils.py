import os
from pathlib import Path

from modelmimic.utils import read_config


def test_read():
    cfg_test = """
[test]
size = [2, 2]
variables = ["U", "V", "T", "Q"]
ntimes = 123
ninst = 45
hist_file_fmt = "file_format_with_{curly}_{braces}"
[test.baseline]
ensemble = { seed = true }
name = "base"
[test.test]
ensemble = { seed = true }
name = "test_test"
"""
    expected_cfg = {
        "test": {
            "size": [2, 2],
            "variables": ["U", "V", "T", "Q"],
            "ntimes": 123,
            "ninst": 45,
            "hist_file_fmt": "file_format_with_{curly}_{braces}",
            "baseline": {"ensemble": {"seed": True}, "name": "base"},
            "test": {"ensemble": {"seed": True}, "name": "test_test"},
        }
    }
    cfg_file = Path("./test_cfg.toml")
    if cfg_file.exists():
        os.system(f"rm -f {cfg_file.resolve()}")

    with open(cfg_file, mode="w", encoding="utf-8") as _fout:
        _fout.write(cfg_test)

    assert cfg_file.exists()

    cfg_in = read_config(cfg_file)

    assert isinstance(cfg_in, dict)
    assert "test" in cfg_in
    assert cfg_in == expected_cfg
