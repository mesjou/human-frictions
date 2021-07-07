import numpy as np


def utility(labor: float, consumption: float):
    assert labor >= 0, "Labor must not be negative"
    assert consumption >= 0, "Labor must not be negative"
    return np.log(consumption) - labor
