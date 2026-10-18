import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.microduck_hop12_env_cfg import make_microduck_hop12_env_cfg
from mjlab_microduck.tasks.microduck_hop14_env_cfg import make_microduck_hop14_env_cfg
from mjlab_microduck.tasks.microduck_hop15_env_cfg import (
    HOP15_PUSH_OFFSET_WEIGHT, MicroduckHop15RlCfg, make_microduck_hop15_env_cfg)
from mjlab_microduck.tasks.microduck_hop16_env_cfg import (
    HOP16_PUSH_DRIFT_WEIGHT, MicroduckHop16RlCfg, make_microduck_hop16_env_cfg)


def test_hop15_only_adds_push_offset():
    a, b = make_microduck_hop12_env_cfg().rewards, make_microduck_hop15_env_cfg().rewards
    assert set(b) - set(a) == {"hop_push_offset"}
    for k in a:
        assert a[k].weight == b[k].weight
    r = b["hop_push_offset"]
    assert r.func is m.hop15_push_offset_penalty
    assert r.weight == HOP15_PUSH_OFFSET_WEIGHT > 0


def test_hop16_is_hop14_with_lower_weight():
    a, b = make_microduck_hop14_env_cfg().rewards, make_microduck_hop16_env_cfg().rewards
    assert set(a) == set(b)
    for k in a:
        if k != "hop_push_drift":
            assert a[k].weight == b[k].weight
    assert b["hop_push_drift"].weight == HOP16_PUSH_DRIFT_WEIGHT == 2.0


def test_hop15_penalty_values(monkeypatch):
    class D:
        geom_pos_w = torch.zeros(4, 2, 3)
        root_link_pos_w = torch.tensor([[0.03, 0.04, 0.1], [0.01, 0.0, 0.1], [0.01, 0.0, 0.1], [1.0, 0.0, 0.1]])
        root_link_lin_vel_w = torch.tensor([[0, 0, 0.3], [0, 0, 0.3], [0, 0, 0.0], [0, 0, 0.3]])
    class A:
        data = D()
        geom_names = ["left_foot_collision", "right_foot_collision"]
    class E:
        num_envs = 4
        device = "cpu"
        scene = {"robot": A()}
    contact = {"L": torch.tensor([True, True, True, False]), "R": torch.tensor([True, True, True, False])}
    monkeypatch.setattr(m, "_sensor_any_contact", lambda env, name: contact[name])
    out = m.hop15_push_offset_penalty(E(), "L", "R")
    # env0 offset 0.05 clamped 0.04; env1 0.01; env2 not rising; env3 airborne
    assert torch.allclose(out, torch.tensor([-0.04, -0.01, 0.0, 0.0]))


def test_hop15_16_registered():
    assert MicroduckHop15RlCfg.experiment_name == "microduck_hop15"
    assert MicroduckHop16RlCfg.experiment_name == "microduck_hop16"
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert {"Mjlab-Hop15-Flat-MicroDuck", "Mjlab-Hop16-Flat-MicroDuck"} <= set(list_tasks())
