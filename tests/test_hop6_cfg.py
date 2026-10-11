from mjlab_microduck.tasks.microduck_hop5_env_cfg import make_microduck_hop5_env_cfg
from mjlab_microduck.tasks.microduck_hop6_env_cfg import (
    MicroduckHop6RlCfg,
    make_microduck_hop6_env_cfg,
)


def test_hop6_rewards_identical_to_hop5():
    a, b = make_microduck_hop5_env_cfg(), make_microduck_hop6_env_cfg()
    assert a.rewards.keys() == b.rewards.keys()
    for k in a.rewards:
        assert a.rewards[k].weight == b.rewards[k].weight


def test_hop6_own_experiment():
    assert MicroduckHop6RlCfg.experiment_name == "microduck_hop6"


def test_hop6_registered():
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Hop6-Flat-MicroDuck" in list_tasks()
