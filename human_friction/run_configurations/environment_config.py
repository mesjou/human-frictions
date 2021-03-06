env_config = {
    "episode_length": 200,
    "n_agents": 10,
    "labor_coefficient": 0.0,
    "technology": 1.0,
    "alpha": 0.25,
    "inflation_target": 0.02,
    "natural_unemployment": 0.0,
    "natural_interest": 0.0,
    "phi_unemployment": 0.1,
    "phi_inflation": 0.2,
    "init_budget": 10.0,
    "init_wage": 0.5623413251903491,
    "init_unemployment": 0.0,
    "init_inflation": 0.02,
    "init_interest": 1.02,
    "seed": None,
}

env_config_nk = {
    **env_config,
    "learning_rate": 0.01,
    "markup": 0.1,
    "memory": 0.45,
}
