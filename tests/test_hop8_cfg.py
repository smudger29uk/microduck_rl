from mjlab_microduck.tasks.microduck_hop7_env_cfg import make_microduck_hop7_env_cfg
from mjlab_microduck.tasks.microduck_hop8_env_cfg import (
    HOP8_MAX_PAID_HEIGHT,
    HOP8_PROGRESS_WEIGHT,
    MicroduckHop8RlCfg,
    make_microduck_hop8_env_cfg,
)


def test_hop8_progress_stronger_than_hop7():
    a, b = make_microduck_hop7_env_cfg().rewards, make_microduck_hop8_env_cfg().rewards
    assert a.keys() == b.keys()
    assert b["hop_airborne_progress"].weight == HOP8_PROGRESS_WEIGHT > a["hop_airborne_progress"].weight
    assert b["hop_airborne_progress"].params["max_paid_height"] == HOP8_MAX_PAID_HEIGHT
    for k in a:
        if k != "hop_airborne_progress":
            assert a[k].weight == b[k].weight


def test_hop8_action_rate_lighter_and_negative():
    c7 = make_microduck_hop7_env_cfg().curriculum["action_rate_weight"].params["weight_stages"]
    c8 = make_microduck_hop8_env_cfg().curriculum["action_rate_weight"].params["weight_stages"]
    assert all(s["weight"] <= 0 for s in c8)
    assert c8[-1]["weight"] > c7[-1]["weight"]


def test_hop8_own_experiment_and_registered():
    assert MicroduckHop8RlCfg.experiment_name == "microduck_hop8"
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Hop8-Flat-MicroDuck" in list_tasks()
