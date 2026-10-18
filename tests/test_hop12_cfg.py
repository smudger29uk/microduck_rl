import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.microduck_hop11_env_cfg import (
    HOP11_SQUAT_OVERRIDES,
    make_microduck_hop11_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop12_env_cfg import (
    HOP12_FOOT_GATE_M,
    HOP12_MAX_PAID_HEIGHT,
    HOP12_PROGRESS_WEIGHT,
    HOP12_TUCK_OVERRIDES,
    MicroduckHop12RlCfg,
    make_microduck_hop12_env_cfg,
)


def test_hop12_tuck_is_mirrored_and_deeper_than_squat():
    t, s = HOP12_TUCK_OVERRIDES, HOP11_SQUAT_OVERRIDES
    for l, r in ((2, 11), (3, 12), (4, 13)):
        assert t[l] == -t[r]
        assert t[l] > s[l]  # more flexion than the crouch
    assert t[3] > 0  # same sign convention as the FK-solved squat
    assert max(abs(v) for v in t.values()) < 1.5


def test_hop12_changes_only_flight_tuck_and_progress():
    a, b = make_microduck_hop11_env_cfg().rewards, make_microduck_hop12_env_cfg().rewards
    assert a.keys() == b.keys()
    assert b["hop_airborne_progress"].weight == HOP12_PROGRESS_WEIGHT
    assert b["hop_airborne_progress"].params["max_paid_height"] == HOP12_MAX_PAID_HEIGHT
    assert b["hop_tuck"].params["tuck_overrides"] == HOP12_TUCK_OVERRIDES
    # same pay per mm of clearance as Hop11
    assert abs(HOP12_PROGRESS_WEIGHT / HOP12_MAX_PAID_HEIGHT - a["hop_airborne_progress"].weight / a["hop_airborne_progress"].params["max_paid_height"]) < 1e-6
    for k in a:
        if k not in ("hop_airborne_progress", "hop_tuck"):
            assert a[k].weight == b[k].weight
    assert make_microduck_hop11_env_cfg().rewards["hop_tuck"].params["tuck_overrides"] == {}


def test_hop12_reset_event():
    ev = make_microduck_hop12_env_cfg().events["set_hop_state"]
    assert ev.func is m.reset_hop12_state
    assert ev.params["foot_gate_m"] == HOP12_FOOT_GATE_M


def test_foot_primary_gate_math():
    # clearance scaled by smoothstep(rise / gate): 0 at no rise, full at >= gate, never capped by rise
    rise = torch.tensor([0.0, 0.003, 0.006, 0.016])
    g = m._smoothstep01(rise / HOP12_FOOT_GATE_M)
    assert g[0] == 0 and g[2] == 1 and g[3] == 1 and 0 < g[1] < 1
    clearance = torch.tensor(0.030)
    assert (clearance * g[3]) > rise[3]  # foot clearance may exceed trunk rise (tuck)


def test_hop12_experiment_and_registered():
    assert MicroduckHop12RlCfg.experiment_name == "microduck_hop12"
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Hop12-Flat-MicroDuck" in list_tasks()
