"""Generate model-like data for testing EVV, _mimic_ the output of various CIME tests.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import modelmimic


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
    size: tuple,
    amplitude: tuple = None,
    length: tuple = None,
    popvar: float = 0.0,
    seed: int = None,
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
    popvar : float, optional
        Add a random normal perturbation on top of field, by default 0.0
    seed : int, optional
        If `seed` is defined, use this to set numpy's random seed

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

    test_data = add_pert(test_data, popvar, seed=seed)
    return test_data, axes


def add_pert(
    test_data: np.array,
    popvar: float = 0.0,
    popmean: float = 0.0,
    seed: int = None,
):
    """
    Add a random normal perturbation to a field of test data.

    Parameters
    ----------
    test_data : np.array
        Data array to which the perturbation will be added
    popvar : float, optional
        Perturbation variance, by default 0.0
    popmean : float, optional
        Population mean, by default 0.0
    seed : int, optional
        If `seed` is defined, use this to set numpy's random seed, by default None

    Returns
    -------
    test_data : np.array
        Array with perturbation added

    """
    if seed is not None:
        np.random.seed(seed)

    pert_data = test_data + (np.random.randn(*test_data.shape) * popvar) + popmean

    return pert_data


class MimicModelRun:
    def __init__(
        self,
        name: str,
        variables: list,
        ntimes: int = 12,
        size: tuple = (5, 10),
        ninst: int = 1,
        dims: tuple = ("nlev", "ncol"),
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
        self.ntimes = ntimes
        self.size = size
        self.ninst = ninst
        self.dims = dims
        assert len(self.dims) == len(
            self.size
        ), f"Number of dims ({len(self.size)}) must match dim names ({len(self.dims)})"
        self.base_data = {}

        for _varix, _var in enumerate(self.vars):
            self.base_data[_var], axes = gen_field(
                (self.ntimes, *self.size),
                amplitude=tuple([_varix + 1 / len(variables)] * (len(self.size) + 1)),
            )

        self.axes = axes

    def __repr__(self):
        return (
            f"### MIMIC CLASS ###\nNAME: {self.name}\n"
            f"NVARS: {len(self.vars)}\nSIZE: {self.size}\nNINST: {self.ninst}"
        )

    def make_ensemble(
        self, popmean: float = 0.0, popvar: float = 1e-5, seed: bool = False
    ):
        """Turn base data into ensemble of data.

        Parameters
        ----------
        seed : bool, optional
            Seed to pass to numpy.random, will use each instance number so results will
            be bit-for-bit, by default False

        """
        ens_data = {}
        for iinst in range(self.ninst):
            ens_data[iinst] = {}

            if seed:
                inst_seed = iinst + 1
            else:
                inst_seed = None

            for _var in self.vars:
                ens_data[iinst][_var] = add_pert(
                    self.base_data[_var],
                    popvar=popvar,
                    popmean=popmean,
                    seed=inst_seed,
                )
        self.ens_data = ens_data

    def get_file_times(
        self,
        sim_start: str = "2000-01-01",
        timestep: str = "month",
    ):
        """Get the time strings for output files

        Parameters
        ----------
        sim_start : str, optional
            Start of the simulated simulation, by default "2000-01-01"
        timestep : str, optional
            Increment of time used, either "month" or "sec". By default "month".

        """
        if timestep.lower() == "month":
            file_times = pd.date_range(start=sim_start, periods=self.ntimes, freq="MS")
            file_times = [_time.strftime("%Y-%m") for _time in file_times]
        elif timestep.lower() == "sec":
            file_times = [f"{sim_start}-{istep:05d}" for istep in range(self.ntimes)]
        else:
            raise NotImplementedError(f"FREQ: {timestep} NOT YET IMPLEMENTED")
        return file_times

    def write_to_nc(
        self,
        out_path: Path = None,
        sim_start: str = "2000-01-01",
        timestep: str = "month",
        hist_file_pattern: str = "eam_{inst:04d}.h0.{time}",
    ):
        """Write generated data to a netCDF file."""
        # Make an xarray.Dataset for each instance so it can be written to a file.
        ens_xarray = {}
        ds_attrs = {
            "title": "MIMIC History file information",
            "source": "Mimic Atmosphere Model",
            "product": "mimic-model-output",
            "realm": "atmos",
            "case": self.name,
            "Conventions": "CF-1.7",
            "institution_id": "MODEL-MIMIC",
            "description": "Mimic a model run: NOT A REAL EARTH SYSTEM MODEL RUN!",
        }
        if out_path is None:
            out_path = Path(f"./data/{self.name}")

        if not out_path.exists():
            out_path.mkdir(parents=True)

        coords = {dim: self.axes[_ix + 1] for _ix, dim in enumerate(self.dims)}
        coords["time"] = [0]

        file_times = self.get_file_times(sim_start, timestep)
        output_files = []
        # file_encoding = {_var: {"zlib": True, "complevel": 3} for _var in self.vars}

        # TODO: Parallelize this
        for iinst in self.ens_data:
            for itime in range(self.ntimes):
                _outfile_name = hist_file_pattern.format(
                    inst=(iinst + 1), time=file_times[itime]
                )
                data_vars = {
                    _var: (self.dims, self.ens_data[iinst][_var][itime])
                    for _var in self.vars
                }
                _dset = xr.Dataset(
                    data_vars=data_vars,
                    coords=coords,
                    attrs={**ds_attrs, "inst": iinst},
                )
                ens_xarray[iinst] = _dset
                _out_file = Path(out_path, f"{self.name}.{_outfile_name}.nc")
                _dset.to_netcdf(
                    _out_file,
                    unlimited_dims="time",
                )
                output_files.append(_out_file)

        return output_files


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--version",
        action="version",
        version="ModelMimic {}".format(modelmimic.__version__),
        help="Show EVV's version number and exit",
    )

    args = parser.parse_args(args)
    return args


def main(args):
    """Interpred CL args, make some data."""
    ntimes = 12
    size = (3, 4)
    ninst = 30

    cases = ["BASE", "TEST"]
    vars_file = Path("/home/ac.mkelleher/evv4esm/evv4esm/extensions/ks_vars.json")
    with open(vars_file, "r") as _fin:
        vars = json.loads(_fin.read())["default"][:10]

    gens = [
        MimicModelRun(_case, variables=vars, ntimes=ntimes, size=size, ninst=ninst)
        for _case in cases
    ]

    gens[0].make_ensemble(popvar=1.0)
    _ = gens[0].write_to_nc()

    gens[1].make_ensemble(popmean=1.0, popvar=1.0)
    _ = gens[1].write_to_nc()


if __name__ == "__main__":
    main(args=parse_args())
