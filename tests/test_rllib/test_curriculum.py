from human_friction.rllib.curriculum import Curriculum


def test_curriculum_fn():
    train_results = {"episode_reward_mean": 0.0}
    curriculum = Curriculum()
    assert curriculum.curriculum_fn(train_results, train_results, train_results) == 10.0

    train_results["episode_reward_mean"] = 710
    assert curriculum.curriculum_fn(train_results, train_results, train_results) == 8.0

    train_results["episode_reward_mean"] = 600
    assert curriculum.curriculum_fn(train_results, train_results, train_results) == 8.0

    train_results["episode_reward_mean"] = 801
    assert curriculum.curriculum_fn(train_results, train_results, train_results) == 6.0

    train_results["episode_reward_mean"] = 400
    assert curriculum.curriculum_fn(train_results, train_results, train_results) == 6.0
