"""Generate fake data for testing EVV, mimic the output of various CIME tests.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def bcast(axis_data, data, axis=None):
    """
    Broadcast a 1-D array to an N-D array

    Parameters
    ----------
    axis_data : array_like
        1D array of data matching the size of one of `data` axes
    data : array_like
        ND array of data onto which `axis_data` will be broadcast
    axis : int, optional
        Axis of `data` onto which `axis_data` will be broadcast, by
        default None, auto-detected by matching shapes

    Returns
    -------
    axis_data : array_like
        ND array of broadcasted axis_data

    """
    # This handles the case where axis_data is 1D and data is N-D, if more than one
    # dimension of `data` is the same size, it will match the first dimension
    if axis is None:
        _dim = int(np.where(np.array(data.shape) == axis_data.shape[0])[0][0])
    else:
        _dim = axis

    # numpy.broadcast_to only works for the last axis of an array, swap our shape
    # around so that vertical dimension is last, broadcast vcoord to it, then swap
    # the axes back so vcoord.shape == data.shape
    data_shape = list(data.shape)
    data_shape[-1], data_shape[_dim] = data_shape[_dim], data_shape[-1]

    axis_data = np.broadcast_to(axis_data, data_shape)
    axis_data = np.swapaxes(axis_data, -1, _dim)
    return axis_data


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

    if amplitude is None:
        amplitude = tuple([1] * naxes)
    if length is None:
        length = tuple([1] * naxes)

    assert naxes == len(length) == len(amplitude), "SIZES DO NOT MATCH"

    test_data = np.zeros(size)

    for _ix in range(naxes):
        axes.append(np.linspace(-1, 1, size[_ix]))
        _axis_data = np.sin(axes[-1] * np.pi / length[_ix])
        test_data += amplitude[_ix] * bcast(_axis_data, test_data, axis=_ix)

    test_data += np.random.randn(*size) * pertlim

    return test_data


class MimicModelRun:

    def __init__(
        self, name: str, variables: list, size: tuple = (3, 5), ninst: int = 1
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
            self.base_data[_var] = gen_field(
                self.size,
                amplitude=tuple([_varix + 1 / len(variables)] * len(self.size))
            )


    def __repr__(self):
        return (
            f"### MIMIC CLASS ###\nNAME: {self.name}\n"
            f"NVARS: {len(self.vars)}\nSIZE: {self.size}\nNINST: {self.ninst}"
        )

    def gen_json(self):
        """Generate a JSON file to be used in the test."""

    def write_to_nc(self):
        """Write generated data to a netCDF file."""
