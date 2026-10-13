import torch
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop8_env_cfg import make_microduck_hop8_env_cfg
from mjlab_microduck.tasks.microduck_hop9_env_cfg import (
    HOP9_HSPEED_WEIGHT,
    HOP9_LAUNCH_WEIGHT,
    HOP9_MAX_VZ,
    MicroduckHop9RlCfg,
    make_microduck_hop9_env_cfg,
)


def test_hop9_changes_only_launch_and_adds_penalty():
    a, b = make_microduck_hop8_env_cfg().rewards, make_microduck_hop9_env_cfg().rewards
    assert set(b) - set(a) == {"hop_horizontal_speed"}
    assert b["hop_launch_velocity"].weight == HOP9_LAUNCH_WEIGHT
    assert b["hop_launch_velocity"].params["max_vz"] == HOP9_MAX_VZ
    for k in a:
        if k != "hop_launch_velocity":
            assert a[k].weight == b[k].weight


def test_hop9_hspeed_penalty_sign_and_weight():
    r = make_microduck_hop9_env_cfg().rewards["hop_horizontal_speed"]
    assert r.weight == HOP9_HSPEED_WEIGHT > 0  # self-negating penalty -> positive weight
    assert r.func is microduck_mdp.hop9_horizontal_speed_penalty


def test_hop9_launch_cap_allows_20mm_rise():
    assert HOP9_MAX_VZ > (2 * 9.81 * 0.020) ** 0.5


def test_hop9_hspeed_penalty_values():
    class D:  # minimal stand-ins
        root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.5], [0.3, 0.4, 0.0], [5.0, 0.0, 0.0]])
    class A: data = D()
    class E:
        scene = {"robot": A()}
    out = microduck_mdp.hop9_horizontal_speed_penalty(E())
    assert torch.allclose(out, torch.tensor([0.0, -0.5, -1.0]))


def test_hop9_experiment_and_registered():
    assert MicroduckHop9RlCfg.experiment_name == "microduck_hop9"
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Hop9-Flat-MicroDuck" in list_tasks()
