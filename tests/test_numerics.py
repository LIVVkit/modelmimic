import evvdatagen
from evvdatagen.datagen import gen_field, bcast, MimicModelRun
import numpy as np

def test_field_gen():
    tol = 1e-15
    data1 = gen_field(size=(3, 10, 10))

    assert data1.shape == (3, 10, 10)
    assert np.abs(np.mean(data1)) < tol

    data2 = gen_field(size=(3, 10, 10), pertlim=1e-2)
    assert data1.shape == data2.shape
    assert np.abs(np.sum(data1 - data2)) > tol


def test_mimic_run():
    gen = MimicModelRun("BASE", variables=["T", "U", "V"], size=(3, 5, 10), ninst=5)
    assert gen.name == "BASE"
    assert gen.vars == ["T", "U", "V"]
    assert gen.size == (3, 5, 10)
    assert gen.ninst == 5


def test_bcast():

    # Test for auto-detect axis for:
    #   - each axis having a different size
    #   - two axes matching
    shape_n = (2, 3, 4)
    shapes = [
        (2, 3, 4),
        (2, 3, 2)
    ]

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