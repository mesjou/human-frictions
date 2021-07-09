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


if __name__ == "__main__":
    test_earn()
