import pytest

from agents.household import HouseholdAgent


def test_earn():
    agent = HouseholdAgent(agent_id="1", budget=0.0)
    agent.earn(hours_worked=1.0, wage=0.5)
    assert agent.labor == 1.0
    assert agent.budget == 0.5

    with pytest.raises(Exception):
        agent.earn(hours_worked=-1.0, wage=0.5)

    with pytest.raises(Exception):
        agent.earn(hours_worked=1.0, wage=-0.5)


def test_consume():
    agent = HouseholdAgent(agent_id="1", budget=0.0)
    agent.consume(consumption=3.0, price=1.0)
    assert agent.budget == -3.0

    agent = HouseholdAgent(agent_id="1", budget=0.0)
    agent.consume(consumption=0.0, price=1.0)
    assert agent.budget == 0.0

    agent.consume(consumption=1.0, price=1.0)
    agent.consume(consumption=1.0, price=2.0)
    assert agent.budget == -3.0

    with pytest.raises(Exception):
        agent.consume(consumption=3.0, price=0.0)


if __name__ == "__main__":
    test_earn()
    test_consume()
