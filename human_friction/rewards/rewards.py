import numpy as np


def utility(labor: float, consumption: float, labor_coefficient: float = 0.0):
    """
    Household utility or reward, concave increasing in consumption and linear decreasing in labor.

    Args:
        labor (float): amount of labor worked in the period.
        consumption (float): amount of real valued consumption.
        labor_coefficient (float): how much does negative utility from working weight in comparison to consumption.

    Returns:
        float: The utility.
    """
    assert labor >= 0, "Labor must not be negative"
    assert consumption >= 0, "Consumption must not be negative"
    assert labor_coefficient >= 0

    util_c = np.log(consumption + 1.0)
    util_l = labor * labor_coefficient

    return util_c - util_l
