import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.microduck_hop12_env_cfg import make_microduck_hop12_env_cfg
from mjlab_microduck.tasks.microduck_hop14_env_cfg import (
    HOP14_PUSH_DRIFT_WEIGHT,
    MicroduckHop14RlCfg,
    make_microduck_hop14_env_cfg,
)


def test_hop14_only_adds_push_drift():
    a, b = make_microduck_hop12_env_cfg().rewards, make_microduck_hop14_env_cfg().rewards
    assert set(b) - set(a) == {"hop_push_drift"}
    for k in a:
        assert a[k].weight == b[k].weight
    r = b["hop_push_drift"]
    assert r.func is m.hop14_push_drift_penalty
    assert r.weight == HOP14_PUSH_DRIFT_WEIGHT > 0  # self-negating -> positive weight


def test_hop14_penalty_values(monkeypatch):
    class D:
        root_link_lin_vel_w = torch.tensor([
            [0.3, 0.4, 0.3],   # grounded, rising: -0.5
            [0.3, 0.4, 0.0],   # grounded, not rising: 0
            [0.3, 0.4, 0.3],   # airborne: 0
            [5.0, 0.0, 0.3],   # grounded, rising, clamped: -0.5
        ])
    class A: data = D()
    class E:
        num_envs = 4
        device = "cpu"
        scene = {"robot": A()}
    contact = {"L": torch.tensor([True, True, False, True]), "R": torch.tensor([False, True, False, True])}
    monkeypatch.setattr(m, "_sensor_any_contact", lambda env, name: contact[name])
    out = m.hop14_push_drift_penalty(E(), "L", "R")
    assert torch.allclose(out, torch.tensor([-0.5, 0.0, 0.0, -0.5]))


def test_hop14_experiment_and_registered():
    assert MicroduckHop14RlCfg.experiment_name == "microduck_hop14"
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Hop14-Flat-MicroDuck" in list_tasks()
