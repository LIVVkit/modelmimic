import evvdatagen
from evvdatagen.datagen import gen_field, MimicModelRun
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
    breakpoint()