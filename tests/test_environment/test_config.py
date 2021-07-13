import pytest

from human_friction.environment.new_keynes import NewKeynesMarket


def test_init_nk_env():
    config = {
        "n_agents": 3,
        "episode_length": 15,
        "init_budget": 0.0,
        "init_unemployment": 0.0,
        "init_inflation": 0.02,
        "init_interest": 1.02,
        "technology": 0.5,
        "alpha": 0.25,
        "learning_rate": 0.01,
        "markup": 0.1,
        "memory": 0.45,
        "inflation_target": 0.02,
        "natural_unemployment": 0.0,
        "natural_interest": 0.0,
        "phi_unemployment": 0.1,
        "phi_inflation": 0.2,
    }
    env = NewKeynesMarket(config)
    env.reset()
    assert env.episode_length == 15
    assert env.unemployment == 0.0
    assert env.inflation == 0.02
    assert env.interest == 1.02
    assert env.firm.technology == 0.5
    assert env.firm.alpha == 0.25
    assert env.firm.learning_rate == 0.01
    assert env.firm.markup == 0.1
    assert env.firm.memory == 0.45
    assert env.central_bank.inflation_target == 0.02
    assert env.central_bank.natural_unemployment == 0.0
    assert env.central_bank.natural_interest == 0.0
    assert env.central_bank.phi_unemployment == 0.1
    assert env.central_bank.phi_inflation == 0.2
    assert len(list(env.agents.keys())) == 3
    assert list(env.agents.values())[0].budget == 0.0

    config_2 = {
        "n_agents": 2,
        "episode_length": 10,
    }
    env = NewKeynesMarket(config_2)
    env.reset()
    assert env.episode_length == 10
    assert env.unemployment == 0.0
    assert env.inflation == 0.02
    assert env.interest == 1.02
    assert env.firm.technology == 0.5
    assert env.firm.alpha == 0.25
    assert env.firm.learning_rate == 0.01
    assert env.firm.markup == 0.1
    assert env.firm.memory == 0.45
    assert env.central_bank.inflation_target == 0.02
    assert env.central_bank.natural_unemployment == 0.0
    assert env.central_bank.natural_interest == 0.0
    assert env.central_bank.phi_unemployment == 0.1
    assert env.central_bank.phi_inflation == 0.2
    assert len(list(env.agents.keys())) == 2
    assert list(env.agents.values())[0].budget == 0.0

    config_3 = {
        "n_agents": 3,
        "episode_length": 15,
        "init_budget": 0.1,
        "init_unemployment": 0.1,
        "init_inflation": 0.03,
        "init_interest": 1.03,
        "technology": 0.6,
        "alpha": 0.26,
        "learning_rate": 0.02,
        "markup": 0.2,
        "memory": 0.46,
        "inflation_target": 0.03,
        "natural_unemployment": 0.1,
        "natural_interest": 0.1,
        "phi_unemployment": 0.2,
        "phi_inflation": 0.3,
    }
    env = NewKeynesMarket(config_3)
    env.reset()
    assert env.unemployment == 0.1
    assert env.inflation == 0.03
    assert env.interest == 1.03
    assert env.firm.technology == 0.6
    assert env.firm.alpha == 0.26
    assert env.firm.learning_rate == 0.02
    assert env.firm.markup == 0.2
    assert env.firm.memory == 0.46
    assert env.central_bank.inflation_target == 0.03
    assert env.central_bank.natural_unemployment == 0.1
    assert env.central_bank.natural_interest == 0.1
    assert env.central_bank.phi_unemployment == 0.2
    assert env.central_bank.phi_inflation == 0.3
    assert list(env.agents.values())[0].budget == 0.1


def test_deny_wrong_values():
    with pytest.raises(Exception):
        NewKeynesMarket({"n_agents": 3.1, "episode_length": 15})
    with pytest.raises(Exception):
        NewKeynesMarket({"episode_length": 15})
    with pytest.raises(Exception):
        NewKeynesMarket({"n_agents": 3})
    with pytest.raises(Exception):
        NewKeynesMarket({"n_agents": 3, "episode_length": 15, "technology": -0.6})
    with pytest.raises(Exception):
        NewKeynesMarket({"n_agents": 3, "episode_length": 15, "phi_unemployment": 0.0})
    with pytest.raises(Exception):
        NewKeynesMarket({"n_agents": 3, "episode_length": 15, "phi_inflation": 0.0})
    with pytest.raises(Exception):
        NewKeynesMarket({"n_agents": 3, "episode_length": 15, "natural_unemployment": -0.1})
    with pytest.raises(Exception):
        NewKeynesMarket({"n_agents": 3, "episode_length": 15, "natural_unemployment": 1.1})
    with pytest.raises(Exception):
        NewKeynesMarket({"n_agents": 3, "episode_length": 15, "markup": -0.1})


if __name__ == "__main__":
    test_init_nk_env()
