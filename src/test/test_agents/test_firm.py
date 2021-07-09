import pytest

import numpy as np
from agents.firm import Firm


def test_hire_worker():
    """Test firm for 3 agents"""
    firm = Firm(init_labor_demand=3.0)
    labor_demand = firm.hire_worker({"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0})
    assert labor_demand == {"agent-0": 1.0, "agent-1": 1.0, "agent-2": 1.0}

    firm = Firm(init_labor_demand=2.0)
    labor_demand = firm.hire_worker({"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0})
    assert labor_demand == {"agent-0": 1.0, "agent-1": 1.0, "agent-2": 0.0}

    firm = Firm(init_labor_demand=1.5)
    labor_demand = firm.hire_worker({"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0})
    assert labor_demand == {"agent-0": 1.0, "agent-1": 0.5, "agent-2": 0.0}

    firm = Firm(init_labor_demand=0.9)
    labor_demand = firm.hire_worker({"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0})
    assert labor_demand == {"agent-0": 0.9, "agent-1": 0.0, "agent-2": 0.0}


def test_produce():
    firm = Firm(init_labor_demand=3.0, technology=0.5, alpha=0.0)
    occupation = {"agent-0": 1.0, "agent-1": 0.4, "agent-2": 0.0}
    firm.produce(occupation)
    assert firm.production == 0.7

    firm = Firm(init_labor_demand=3.0, technology=0.5, alpha=1.0)
    firm.produce(occupation)
    assert firm.production == 0.5

    occupation = {"agent-0": 0.0, "agent-1": 0.0, "agent-2": 0.0}
    with pytest.raises(Exception):
        firm.produce(occupation)


def test_set_price():
    firm = Firm(init_labor_demand=3.0, markup=0.2, alpha=0.5)
    firm.production = 1.0
    firm.price = 1.0
    occupation = {"agent-0": 1.0, "agent-1": 0.4, "agent-2": 0.0}
    wages = {"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0}
    inflation = firm.set_price(occupation, wages)
    assert firm.price == 4.32
    assert np.round(inflation, 6) == 3.32

    firm = Firm(init_labor_demand=3.0, markup=0.2, alpha=0.5)
    firm.production = 2.0
    firm.price = 1.0
    occupation = {"agent-0": 1.0, "agent-1": 0.4, "agent-2": 0.0}
    wages = {"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0}
    inflation = firm.set_price(occupation, wages)
    assert firm.price == 2.16
    assert np.round(inflation, 6) == 1.16

    with pytest.raises(Exception):
        firm.set_price(occupation, {"agent-0": 0.0, "agent-1": 0.0, "agent-2": 0.0})


if __name__ == "__main__":
    test_hire_worker()
    test_produce()
    test_set_price()
