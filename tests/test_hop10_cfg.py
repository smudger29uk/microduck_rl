from mjlab_microduck.tasks.microduck_hop9_env_cfg import make_microduck_hop9_env_cfg
from mjlab_microduck.tasks.microduck_hop10_env_cfg import (
    HOP10_FELL_OVER_RAD,
    HOP10_HSPEED_WEIGHT,
    MicroduckHop10RlCfg,
    make_microduck_hop10_env_cfg,
)


def test_hop10_only_lowers_hspeed_and_adds_termination():
    a, b = make_microduck_hop9_env_cfg().rewards, make_microduck_hop10_env_cfg().rewards
    assert set(a) == set(b)
    assert b["hop_horizontal_speed"].weight == HOP10_HSPEED_WEIGHT
    assert 0 < HOP10_HSPEED_WEIGHT < a["hop_horizontal_speed"].weight  # still positive (self-negating)
    for k in a:
        if k != "hop_horizontal_speed":
            assert a[k].weight == b[k].weight


def test_hop10_fell_over_termination():
    t = make_microduck_hop10_env_cfg().terminations
    assert t["fell_over"].params["limit_angle"] == HOP10_FELL_OVER_RAD
    assert HOP10_FELL_OVER_RAD > 0.5  # well above hop tilt (5 to 10 deg)
    assert "nan_state" in t


def test_hop10_experiment_and_registered():
    assert MicroduckHop10RlCfg.experiment_name == "microduck_hop10"
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Hop10-Flat-MicroDuck" in list_tasks()
