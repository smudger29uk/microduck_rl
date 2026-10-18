import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.microduck_hop12_env_cfg import make_microduck_hop12_env_cfg
from mjlab_microduck.tasks.microduck_hop13_env_cfg import (
    HOP13_AIRBORNE_DRIFT_WEIGHT,
    MicroduckHop13RlCfg,
    make_microduck_hop13_env_cfg,
)


def test_hop13_only_adds_airborne_drift():
    a, b = make_microduck_hop12_env_cfg().rewards, make_microduck_hop13_env_cfg().rewards
    assert set(b) - set(a) == {"hop_airborne_drift"}
    for k in a:
        assert a[k].weight == b[k].weight
    r = b["hop_airborne_drift"]
    assert r.func is m.hop13_airborne_drift_penalty
    assert r.weight == HOP13_AIRBORNE_DRIFT_WEIGHT > 0  # self-negating -> positive weight


def test_hop13_penalty_values(monkeypatch):
    class D:
        root_link_lin_vel_w = torch.tensor([[0.3, 0.4, 0.0], [0.3, 0.4, 0.0], [5.0, 0.0, 0.0]])
    class A: data = D()
    class E:
        num_envs = 3
        device = "cpu"
        scene = {"robot": A()}
    contact = {"L": torch.tensor([False, True, False]), "R": torch.tensor([False, True, False])}
    monkeypatch.setattr(m, "_sensor_any_contact", lambda env, name: contact[name])
    out = m.hop13_airborne_drift_penalty(E(), "L", "R")
    # airborne env0: -0.5; grounded env1: 0; airborne env2: clamped -0.5
    assert torch.allclose(out, torch.tensor([-0.5, 0.0, -0.5]))


def test_hop13_experiment_and_registered():
    assert MicroduckHop13RlCfg.experiment_name == "microduck_hop13"
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Hop13-Flat-MicroDuck" in list_tasks()
