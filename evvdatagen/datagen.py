"""Generate fake data for testing EVV, mimic the output of various CIME tests.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def gen_field(
    size: tuple, amplitude: tuple = None, length: tuple = None, pertlim: float = 0.0
):
    """
    Generate a semi-realistic atmosphere field.

    Parameters
    ----------
    size : tuple
        Shape for the data
    amplitude : tuple
        Amplitude parameter for each axis of data, must be same length as `size`
    length : tuple
        Length parameter for each axis of data, must be same length as `size`
    pertlim : float, optional
        Add a random normal perturbation on top of field, by default 0.0

    Returns
    -------
    test_data : array_like
        `numpy.array` of sample data

    """

    naxes = len(size)

    axes = []
    bcast = [None] * naxes

    if amplitude is None:
        amplitude = tuple([1] * naxes)
    if length is None:
        length = tuple([1] * naxes)

    assert naxes == len(length) == len(amplitude), "SIZES DO NOT MATCH"

    test_data = np.zeros(size)

    for _ix in range(naxes):
        axes.append(np.linspace(-1, 1, size[_ix]))
        _bcast = list(bcast)
        _bcast[_ix] = slice(0, None)
        test_data += amplitude[_ix] * np.sin(axes[-1] * np.pi / length[_ix])[*_bcast]

    test_data += np.random.randn(*size) * pertlim

    return test_data


class MimicModelRun:

    def __init__(
        self, name: str, variables: list[str], size: tuple = (3, 5), ninst: int = 1
    ):
        """
        Initalize a pseudo model run to mimic an EAM (or other) model run>

        Parameters
        ----------
        name : str
            Name of model run, probably BASE or TEST
        variables : list[str]
            List of variable names to generate mimic data for
        size : tuple
            Shape for the data
        ninst : int, optional
            Number of instances, by default 1

        """
        self.name = name
        self.vars = variables
        self.size = size
        self.ninst = ninst
        self.base_data = {}

        for _varix, _var in enumerate(self.vars):
            self.data[_var] = gen_field(
                self.size,
                amplitude=tuple([_varix + 1 / len(variables)] * len(self.size))
            )


    def __repr__(self):
        return f"DATAGEN CLASS\n  TEST: {self.test}\nACCEPT: {self.accept}"

    def gen_json(self):
        """Generate a JSON file to be used in the test."""

    def write_to_nc(self):
        """Write generated data to a netCDF file."""
