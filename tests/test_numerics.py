import shutil
from pathlib import Path

import numpy as np

from modelmimic.mimic import MimicModelRun, add_pert, bcast, gen_field


def test_field_gen():
    tol = 1e-3
    arr_size = (3, 10, 12)
    data1, axes1 = gen_field(size=arr_size)
    data2, axes2 = gen_field(size=arr_size, popvar=1e-2)
    data3, axes3 = gen_field(size=arr_size, popvar=1e-2, seed=12)
    data4, axes4 = gen_field(size=arr_size, popvar=1e-2, seed=12)

    assert data1.shape == arr_size
    assert len(axes1) == len(arr_size)
    assert np.all(axes1[0] == np.linspace(-1, 1, arr_size[0]))
    assert np.all(axes1[1] == np.linspace(-1, 1, arr_size[1]))
    assert np.all(axes1[2] == np.linspace(-1, 1, arr_size[2]))
    assert np.abs(np.mean(data1)) < tol

    assert data1.shape == data2.shape
    assert np.abs(np.sum(data1 - data2)) > tol

    assert data2.shape == data3.shape
    assert np.abs(np.sum(data3 - data2)) > tol

    assert data4.shape == data3.shape
    assert np.abs(np.sum(data3 - data4)) <= tol


def test_add_pert():
    data1 = np.ones((30, 40, 50))
    data2 = add_pert(data1, popvar=0.0, popmean=0.0, seed=5)

    data3 = add_pert(data1, popvar=1.0, popmean=3.0, seed=1)
    data4 = add_pert(data1, popvar=1.0, popmean=3.0, seed=1)

    data5 = add_pert(data1, popvar=2.5, popmean=-3.0)
    data6 = add_pert(data1, popvar=2.5, popmean=-3.0)

    assert (data1 == data2).all()
    assert (data3 == data4).all()

    assert abs(data3.mean() - 4.0) < 0.005
    assert abs(data3.std() - 1.0) < 0.05

    assert data5.mean() != data6.mean()
    assert data5.std() != data6.std()

    assert abs(data5.mean() - data6.mean()) < 1e-2


def test_mimic_run():
    gen = MimicModelRun(
        "BASE",
        variables=["T", "U", "V"],
        ntimes=12,
        size=(5, 10),
        dims=("nlev", "ncol"),
        ninst=5,
    )
    assert gen.name == "BASE"
    assert gen.vars == ["T", "U", "V"]
    assert gen.size == (5, 10)
    assert gen.ntimes == 12
    assert gen.ninst == 5
    assert gen.dims == ("nlev", "ncol")


def test_bcast():

    # Test for auto-detect axis for:
    #   - each axis having a different size
    #   - two axes matching
    shape_n = (2, 3, 4)
    shapes = [(2, 3, 4), (2, 3, 2)]

    for shape_n in shapes:
        arr_n = np.zeros(shape_n)
        arr_0 = np.ones(shape_n[0]) * 2
        arr_1 = np.ones(shape_n[1]) * 3
        arr_2 = np.ones(shape_n[2]) * 4

        assert bcast(arr_0, arr_n).shape == shape_n
        assert bcast(arr_1, arr_n).shape == shape_n
        assert bcast(arr_2, arr_n).shape == shape_n

        assert (bcast(arr_0, arr_n) + arr_n == 2 * np.ones(shape_n)).all()
        assert (bcast(arr_1, arr_n) + arr_n == 3 * np.ones(shape_n)).all()
        assert (bcast(arr_2, arr_n) + arr_n == 4 * np.ones(shape_n)).all()

    for shape_n in shapes:
        arr_n = np.zeros(shape_n)
        arr_0 = np.ones(shape_n[0]) * 2
        arr_1 = np.ones(shape_n[1]) * 3
        arr_2 = np.ones(shape_n[2]) * 4

        assert bcast(arr_0, arr_n, 0).shape == shape_n
        assert bcast(arr_1, arr_n, 1).shape == shape_n
        assert bcast(arr_2, arr_n, 2).shape == shape_n

        assert (bcast(arr_0, arr_n, 0) + arr_n == 2 * np.ones(shape_n)).all()
        assert (bcast(arr_1, arr_n, 1) + arr_n == 3 * np.ones(shape_n)).all()
        assert (bcast(arr_2, arr_n, 2) + arr_n == 4 * np.ones(shape_n)).all()

    arr_n = np.zeros(shapes[0])

    try:
        bcast(arr_0, arr_n, axis=2)
    except ValueError as _err:
        assert "operands could not be broadcast together" in str(_err)
    else:
        raise


def test_ensemble():
    gen = MimicModelRun("BASE", variables=["T", "U", "V"], size=(5, 10), ninst=10)
    gen.make_ensemble()
    assert len(gen.ens_data) == 10
    assert np.all(gen.ens_data[0]["T"] != gen.ens_data[1]["T"])
    assert np.all(gen.ens_data[0]["U"] != gen.ens_data[0]["T"])


def test_file_times():
    gen = MimicModelRun(
        "BASE", variables=["T", "U", "V"], ntimes=12, size=(5, 10), ninst=10
    )
    gen.make_ensemble()
    _times = gen.get_file_times("0001-01-01", timestep="month")
    assert _times == [
        "0001-01",
        "0001-02",
        "0001-03",
        "0001-04",
        "0001-05",
        "0001-06",
        "0001-07",
        "0001-08",
        "0001-09",
        "0001-10",
        "0001-11",
        "0001-12",
    ]

    _times = gen.get_file_times("0001-01-01", timestep="sec")
    step_times = [
        "0001-01-01-00000",
        "0001-01-01-00001",
        "0001-01-01-00002",
        "0001-01-01-00003",
        "0001-01-01-00004",
        "0001-01-01-00005",
        "0001-01-01-00006",
        "0001-01-01-00007",
        "0001-01-01-00008",
        "0001-01-01-00009",
        "0001-01-01-00010",
        "0001-01-01-00011",
    ]
    assert _times == step_times

    _times = gen.get_file_times("0001-01-01", timestep="sec", step_mult=10)
    step_times = [
        "0001-01-01-00000",
        "0001-01-01-00010",
        "0001-01-01-00020",
        "0001-01-01-00030",
        "0001-01-01-00040",
        "0001-01-01-00050",
        "0001-01-01-00060",
        "0001-01-01-00070",
        "0001-01-01-00080",
        "0001-01-01-00090",
        "0001-01-01-00100",
        "0001-01-01-00110",
    ]
    assert _times == step_times


def test_to_netcdf():
    ntimes = 12
    size = (5, 10)
    ninst = 2
    gen = MimicModelRun(
        "BASE", variables=["T", "U", "V"], ntimes=ntimes, size=size, ninst=ninst
    )
    gen.make_ensemble()
    _files = gen.write_to_nc()
    assert len(_files) == (ntimes * ninst)
    assert all([_file.exists() for _file in _files])
    assert _files[0].parent == Path("./data/BASE")
    if _files[0].parent == Path("./data/BASE"):
        shutil.rmtree(_files[0].parent.parent)
