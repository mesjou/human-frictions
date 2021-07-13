import numpy as np


def utility(labor: float, consumption: float):
    assert labor >= 0, "Labor must not be negative"
    assert consumption >= 0, "Consumption must not be negative"
    if consumption > 0:
        return np.log(consumption) - labor
    else:
        return -labor
