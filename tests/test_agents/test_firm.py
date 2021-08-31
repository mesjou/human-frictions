import pytest

import numpy as np
from human_friction.agents.learningfirm import LearningFirm


def test_hire_worker():
    """Test firm for 3 agents"""
    firm = LearningFirm()
    firm.labor_demand = 3.0
    labor_demand = firm.hire_worker({"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0})
    assert labor_demand == {"agent-0": 1.0, "agent-1": 1.0, "agent-2": 1.0}
    assert firm.profit == -6.0

    firm = LearningFirm()
    firm.labor_demand = 2.0
    labor_demand = firm.hire_worker({"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0})
    assert labor_demand == {"agent-0": 1.0, "agent-1": 1.0, "agent-2": 0.0}
    assert firm.profit == -3.0

    firm = LearningFirm()
    firm.labor_demand = 1.5
    labor_demand = firm.hire_worker({"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0})
    assert labor_demand == {"agent-0": 1.0, "agent-1": 0.5, "agent-2": 0.0}
    assert firm.profit == -2.0

    firm = LearningFirm()
    firm.labor_demand = 0.9
    labor_demand = firm.hire_worker({"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0})
    assert labor_demand == {"agent-0": 0.9, "agent-1": 0.0, "agent-2": 0.0}
    assert firm.profit == -0.9


def test_produce():
    firm = LearningFirm(technology=0.5, alpha=0.0)
    firm.labor_demand = 3.0
    occupation = {"agent-0": 1.0, "agent-1": 0.4, "agent-2": 0.0}
    firm.produce(occupation)
    assert firm.production == 0.7

    firm = LearningFirm(technology=0.5, alpha=1.0)
    firm.labor_demand = 3.0
    firm.produce(occupation)
    assert firm.production == 0.5

    occupation = {"agent-0": 0.0, "agent-1": 0.0, "agent-2": 0.0}
    with pytest.raises(Exception):
        firm.produce(occupation)


def test_set_price():
    firm = LearningFirm(markup=0.2, alpha=0.5)
    firm.labor_demand = 3.0
    firm.production = 1.0
    firm.price = 1.0
    occupation = {"agent-0": 1.0, "agent-1": 0.4, "agent-2": 0.0}
    wages = {"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0}
    inflation = firm.set_price(occupation, wages)
    assert firm.price == 4.32
    assert np.round(inflation, 6) == 3.32

    firm = LearningFirm(markup=0.2, alpha=0.5)
    firm.labor_demand = 3.0
    firm.production = 2.0
    firm.price = 1.0
    occupation = {"agent-0": 1.0, "agent-1": 0.4, "agent-2": 0.0}
    wages = {"agent-0": 1.0, "agent-1": 2.0, "agent-2": 3.0}
    inflation = firm.set_price(occupation, wages)
    assert firm.price == 2.16
    assert np.round(inflation, 6) == 1.16

    with pytest.raises(Exception):
        firm.set_price(occupation, {"agent-0": 0.0, "agent-1": 0.0, "agent-2": 0.0})


def test_sell_goods():
    firm = LearningFirm(markup=0.2, alpha=0.5)
    firm.labor_demand = 3.0
    firm.production = 2.0
    demand = {"agent-0": 1.0, "agent-1": 0.4, "agent-2": 0.1}
    consumption = firm.sell_goods(demand)
    assert consumption == demand

    firm = LearningFirm(markup=0.2, alpha=0.5)
    firm.labor_demand = 3.0
    firm.production = 0.0
    demand = {"agent-0": 1.0, "agent-1": 0.4, "agent-2": 0.1}
    consumption = firm.sell_goods(demand)
    assert consumption == {"agent-0": 0.0, "agent-1": 0.0, "agent-2": 0.0}

    firm = LearningFirm(markup=0.2, alpha=0.5)
    firm.labor_demand = 3.0
    firm.production = 2.0
    demand = {"agent-0": 0.0, "agent-1": 0.0, "agent-2": 0.0}
    consumption = firm.sell_goods(demand)
    assert consumption == demand

    firm = LearningFirm(markup=0.2, alpha=0.5)
    firm.labor_demand = 0.0
    firm.production = 2.0
    demand = {"agent-0": 1.1, "agent-1": 0.5, "agent-2": 0.6}
    consumption = firm.sell_goods(demand)
    assert sum([c for c in consumption.values()]) == 2.0
    assert pytest.approx(consumption) == {"agent-0": 0.9, "agent-1": 0.5, "agent-2": 0.6}


def test_earn_profits():
    firm = LearningFirm(markup=0.2, alpha=0.5)
    firm.labor_demand = 0.0
    firm.price = 1.0
    consumption = {"agent-0": 1.1, "agent-1": 0.5, "agent-2": 0.6}
    firm.earn_profits(consumption)
    assert firm.profit == 2.2

    firm = LearningFirm(markup=0.2, alpha=0.5)
    firm.labor_demand = 0.0
    firm.price = 1.0
    consumption = {"agent-0": 0.0, "agent-1": 0.0, "agent-2": 0.0}
    firm.earn_profits(consumption)
    assert firm.profit == 0.0

    firm.price = 0.0
    with pytest.raises(Exception):
        firm.earn_profits(consumption)


def test_learn():
    firm = LearningFirm(learning_rate=0.5)
    firm.labor_demand = 3.0
    firm.profit = 1.0
    firm.average_profit = 0.0
    firm.learn(max_labor=3.0)
    assert firm.labor_demand == 3.0

    firm = LearningFirm(learning_rate=0.5)
    firm.labor_demand = 3.0
    firm.profit = 1.0
    firm.average_profit = 0.0
    firm.learn(max_labor=2.0)
    assert firm.labor_demand == 2.0

    firm = LearningFirm(learning_rate=0.5)
    firm.labor_demand = 1.0
    firm.profit = 0.9
    firm.average_profit = 1.0
    firm.learn(max_labor=3.0)
    assert firm.labor_demand == 0.5

    firm = LearningFirm(learning_rate=0.5)
    firm.labor_demand = 1.0
    firm.profit = 1.1
    firm.average_profit = 1.0
    firm.learn(max_labor=3.0)
    assert firm.labor_demand == 1.5

    firm = LearningFirm(learning_rate=0.5)
    firm.labor_demand = 2.5
    firm.profit = 1.1
    firm.average_profit = 1.0
    firm.learn(max_labor=3.0)
    assert firm.labor_demand == 3.0

    with pytest.raises(Exception):
        firm.learn(max_labor=0.0)


def test_update_average_profit():
    firm = LearningFirm(memory=0.5)
    firm.labor_demand = 2.0
    firm.profit = 1.0
    firm.average_profit = 0.0
    firm.update_average_profit()
    assert firm.average_profit == 0.5

    firm = LearningFirm(memory=0.5)
    firm.labor_demand = 2.0
    firm.profit = 0.0
    firm.average_profit = 0.0
    firm.update_average_profit()
    assert firm.average_profit == 0.0

    firm = LearningFirm(memory=0.0)
    firm.labor_demand = 2.0
    firm.profit = 10.0
    firm.average_profit = 0.0
    firm.update_average_profit()
    assert firm.average_profit == 10.0
