import math

import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.microduck_hop4_env_cfg import (
    HOP4_RAMP_S,
    HOP4_WINDOW_S,
    make_microduck_hop4_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop5_env_cfg import (
    HOP5_CROUCH_DEPTH,
    HOP5_HEAD_POSE_WEIGHT,
    HOP5_SQUAT_OVERRIDES,
    MicroduckHop5RlCfg,
    make_microduck_hop5_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    _LEG_JOINTS,
    HOP_CROUCH_OVERRIDES,
    STAND_Z,
)
from test_hop4_cfg import L, P, R, _Env  # tests/ is on sys.path (no __init__)

HOME14 = torch.zeros(1, 14)
HOME14[0, 2], HOME14[0, 4] = -0.46, 0.44
HOME14[0, 11], HOME14[0, 13] = 0.46, -0.44


def _patch(monkeypatch, q):
    monkeypatch.setattr(m, "_servo_default_joint_pos", lambda env, asset: HOME14.clone())
    monkeypatch.setattr(m, "_servo_joint_pos", lambda env, asset: q.clone())


def _ramp(env, **kw):
    return m.hop5_crouch_pose_ramp(
        env, L, R, STAND_Z, _LEG_JOINTS, HOP5_SQUAT_OVERRIDES,
        crouch_depth=HOP5_CROUCH_DEPTH, **{**P, **kw},
    ).item()


def _ref_at(s):
    q = HOME14.clone()
    for k, v in HOP5_SQUAT_OVERRIDES.items():
        q[0, k] = HOME14[0, k] + s * (v - HOME14[0, k])
    return q


def test_wiring_and_registration():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    assert make_microduck_hop5_env_cfg(play=True) is not None
    assert "Mjlab-Hop5-Flat-MicroDuck" in list_tasks()
    cfg = make_microduck_hop5_env_cfg()
    assert cfg.rewards["hop_crouch"].func is m.hop5_crouch_pose_ramp
    assert cfg.rewards["hop_airborne_progress"].func is m.hop4_airborne_progress  # Hop4 gates kept
    assert cfg.rewards["hop_crouch"].params["joint_indices"] == _LEG_JOINTS
    assert cfg.rewards["head_pose_l1"].weight == HOP5_HEAD_POSE_WEIGHT > 0.3
    assert cfg.rewards["hop_crouch"].weight > 0
    assert MicroduckHop5RlCfg.experiment_name == "microduck_hop5"
    keys = ("foot_rest_z", "crouch_full_depth", "min_launch_s", "crouch_tilt_ok_deg",
            "flight_tilt_full_deg", "flight_tilt_zero_deg", "min_flight_z")
    ref = {k: cfg.rewards["hop_airborne_progress"].params[k] for k in keys}
    for k in keys:
        assert cfg.rewards["hop_crouch"].params[k] == ref[k]


def test_hop4_untouched():
    c4 = make_microduck_hop4_env_cfg()
    assert c4.rewards["hop_crouch"].func is m.hop4_crouch_ramp
    assert c4.rewards["head_pose_l1"].weight == 0.3


def test_squat_target_is_mirrored_and_distinct_from_old_crouch():
    o = HOP5_SQUAT_OVERRIDES
    assert set(o) == {2, 3, 4, 11, 12, 13}  # yaw/roll NOT overridden -> HOME (0)
    for l, r in ((2, 11), (3, 12), (4, 13)):
        assert o[r] == -o[l]
    # Flat sole: ankle = knee - hip - 0.02 (left leg), the FK relation HOME obeys.
    assert abs(o[4] - (o[3] - o[2] - 0.02)) < 0.02
    # The old target has the hips far behind (-0.65); the balanced one does not.
    assert o[2] > HOP_CROUCH_OVERRIDES[2] + 0.5
    assert 0.02 < HOP5_CROUCH_DEPTH < 0.05


def test_pose_reference_perfect_pays_full(monkeypatch):
    env = _Env()
    t = HOP4_RAMP_S / 2
    z = STAND_Z - HOP5_CROUCH_DEPTH / 2
    _patch(monkeypatch, _ref_at(0.5))
    env.step(t, z, True, True, tilt=2.0)
    assert abs(_ramp(env) - 1.0) < 1e-5


def test_hip_yaw_splay_and_asymmetry_are_penalised(monkeypatch):
    env = _Env()
    t = HOP4_RAMP_S
    z = STAND_Z - HOP5_CROUCH_DEPTH
    env.step(t, z, True, True, tilt=2.0)

    _patch(monkeypatch, _ref_at(1.0))
    full = _ramp(env)

    q = _ref_at(1.0)
    q[0, 0] += 0.35   # left hip yaw splay
    q[0, 9] -= 0.35   # right hip yaw splay
    _patch(monkeypatch, q)
    splay = _ramp(env)

    q = _ref_at(1.0)
    q[0, 12] += 0.35  # right knee only half-bent (right is mirrored, so +0.35 un-bends it)
    _patch(monkeypatch, q)
    asym = _ramp(env)

    q = _ref_at(1.0)
    for k in HOP_CROUCH_OVERRIDES:
        q[0, k] = HOP_CROUCH_OVERRIDES[k]  # the old, CoM-behind-heel pose
    _patch(monkeypatch, q)
    old_pose = _ramp(env)

    assert full > 0.99
    assert splay < 0.8 * full
    assert asym < 0.9 * full
    assert old_pose < 0.5 * full


def test_ramp_closed_by_tilt_window_airborne_and_after_hop(monkeypatch):
    env = _Env()
    _patch(monkeypatch, _ref_at(0.5))
    env.step(HOP4_RAMP_S / 2, STAND_Z - HOP5_CROUCH_DEPTH / 2, True, True, tilt=30.0)
    assert _ramp(env) == 0.0                        # leaning
    env.step(HOP4_WINDOW_S + 0.1, STAND_Z - HOP5_CROUCH_DEPTH, True, True, tilt=0.0)
    assert _ramp(env) == 0.0                        # past the window
    env.step(0.3, STAND_Z - 0.01, False, False, tilt=0.0)
    assert _ramp(env) == 0.0                        # airborne
