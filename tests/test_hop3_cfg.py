from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.microduck_hop3_env_cfg import (
    HOP3_CROUCH_DEPTH,
    HOP3_ENTROPY_COEF,
    HOP3_MAX_PAID_CLEARANCE,
    HOP3_MIN_LAUNCH_S,
    HOP3_RAMP_S,
    HOP3_WINDOW_S,
    MicroduckHop3RlCfg,
    make_microduck_hop3_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop_env_cfg import STAND_Z, MicroduckHopRlCfg

L, R = "left_foot_ground_contact", "right_foot_ground_contact"
COMMON = dict(foot_rest_z=0.006, crouch_full_depth=0.03, min_launch_s=0.5)


def test_builds_and_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    assert make_microduck_hop3_env_cfg() is not None
    assert make_microduck_hop3_env_cfg(play=True) is not None
    ids = list_tasks()
    for t in ("Mjlab-Hop-Flat-MicroDuck", "Mjlab-Hop2-Flat-MicroDuck", "Mjlab-Hop3-Flat-MicroDuck"):
        assert t in ids


def test_earlier_tasks_untouched():
    from mjlab_microduck.tasks.microduck_hop2_env_cfg import make_microduck_hop2_env_cfg

    c2 = make_microduck_hop2_env_cfg()
    assert c2.rewards["hop_airborne_progress"].func is m.hop2_airborne_progress
    assert c2.rewards["hop_crouch"].func is m.hop2_crouch_depth
    assert c2.events["set_hop_state"].func is m.reset_hop2_state
    c1 = __import__(
        "mjlab_microduck.tasks.microduck_hop_env_cfg", fromlist=["x"]
    ).make_microduck_hop_env_cfg()
    assert c1.rewards["hop_airborne_progress"].func is m.hop_airborne_progress
    assert c1.events["set_hop_state"].func is m.reset_hop_state
    assert MicroduckHopRlCfg.experiment_name == "microduck_hop"
    assert MicroduckHopRlCfg.algorithm.entropy_coef == 0.01


def test_hop3_wiring():
    cfg = make_microduck_hop3_env_cfg()
    assert cfg.rewards["hop_airborne_progress"].func is m.hop3_airborne_progress
    assert cfg.rewards["hop_crouch"].func is m.hop3_crouch_ramp
    assert cfg.rewards["hop_landing_composite"].func is m.hop3_landing_composite
    assert cfg.rewards["hop_stand_tax"].func is m.hop3_stand_tax
    assert cfg.events["set_hop_state"].func is m.reset_hop3_state
    assert cfg.rewards["hop_airborne_progress"].params["max_paid_height"] == HOP3_MAX_PAID_CLEARANCE
    assert cfg.rewards["head_pose_l1"].weight > 0  # inherited head hold
    assert MicroduckHop3RlCfg.experiment_name == "microduck_hop3"
    assert MicroduckHop3RlCfg.algorithm.entropy_coef == HOP3_ENTROPY_COEF


def test_gate_params_consistent_across_terms():
    cfg = make_microduck_hop3_env_cfg()
    keys = ("foot_rest_z", "crouch_full_depth", "min_launch_s")
    ref = {k: cfg.rewards["hop_airborne_progress"].params[k] for k in keys}
    for n in ("hop_crouch", "hop_landing_composite", "hop_stand_tax"):
        for k in keys:
            assert cfg.rewards[n].params[k] == ref[k], f"{n}.{k} differs"
    for n in ("hop_crouch", "hop_landing_composite", "hop_stand_tax"):
        assert cfg.rewards[n].params["min_hop_height"] == 0.010


def test_signs_and_curricula():
    cfg = make_microduck_hop3_env_cfg()
    for n in ("hop_stand_tax", "head_pose_l1", "height_stand_l1", "gentle_landing"):
        assert cfg.rewards[n].weight >= 0
    assert cfg.rewards["hop_crouch"].weight > 0
    for c in cfg.curriculum.values():
        name = c.params.get("reward_name")
        if name:
            assert name in cfg.rewards


def test_ramp_constants_sane():
    assert 0 < HOP3_RAMP_S < HOP3_WINDOW_S < 3.0
    assert HOP3_MIN_LAUNCH_S < HOP3_WINDOW_S
    assert 0.01 < HOP3_CROUCH_DEPTH < 0.06  # policy reached ~0.05 dynamically


# ── functional tests with a fake env ──────────────────────────────────────────

class _Env:
    dt = 0.02

    def __init__(self):
        self.num_envs, self.device = 1, "cpu"
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.step_dt = self.dt
        env = self
        self.contact = {L: torch.ones(1, dtype=torch.bool), R: torch.ones(1, dtype=torch.bool)}
        self.asset = SimpleNamespace(
            geom_names=("a", "left_foot_collision", "right_foot_collision"),
            data=SimpleNamespace(root_link_pos_w=None, geom_pos_w=None),
        )

        class Sensors(dict):
            def __contains__(s, k):
                return k in env.contact

            def __getitem__(s, k):
                return SimpleNamespace(data=SimpleNamespace(found=env.contact[k].float().view(-1, 1)))

        class Scene:
            sensors = Sensors()
            terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))

            def __getitem__(s, k):
                return env.asset

        self.scene = Scene()

    def step(self, t_s, z, left, right, foot_l=0.006, foot_r=0.006):
        self.common_step_counter += 1
        self.episode_length_buf = torch.tensor([round(t_s / self.dt)])
        self.asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, float(z)]])
        self.asset.data.geom_pos_w = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, foot_l], [0.0, 0.0, foot_r]]]
        )
        self.contact[L] = torch.tensor([left])
        self.contact[R] = torch.tensor([right])


def _progress(env):
    return m.hop3_airborne_progress(env, L, R, STAND_Z, max_paid_height=0.04, **COMMON).item()


def _charge(env, t0=0.0):
    """Crouch to 0.08 m (35 mm below standing) while grounded."""
    for i in range(10):
        env.step(t0 + i * 0.02, 0.08, True, True)
        _progress(env)


def test_charged_hop_after_min_launch_pays_foot_clearance():
    env = _Env()
    _charge(env, t0=0.4)
    env.step(0.62, 0.10, False, False, foot_l=0.026, foot_r=0.026)  # 20 mm clearance
    assert abs(_progress(env) - 0.020 / 0.04) < 1e-6
    env.step(0.64, 0.11, False, False, foot_l=0.056, foot_r=0.046)  # min foot 40 mm
    assert abs(_progress(env) - 0.5) < 1e-6
    env.step(0.66, 0.11, False, False, foot_l=0.09, foot_r=0.09)    # beyond cap
    assert _progress(env) < 1e-5


def test_one_foot_high_does_not_count():
    env = _Env()
    _charge(env, t0=0.4)
    env.step(0.62, 0.10, False, False, foot_l=0.05, foot_r=0.006)  # one foot stays low
    assert _progress(env) == 0.0


def test_no_crouch_no_pay():
    env = _Env()
    for i in range(10):
        env.step(0.5 + i * 0.02, 0.115, True, True)
        _progress(env)
    env.step(0.8, 0.13, False, False, foot_l=0.03, foot_r=0.03)
    assert _progress(env) == 0.0


def test_early_launch_before_min_time_no_pay():
    env = _Env()
    _charge(env, t0=0.0)             # crouched by t=0.18 s
    env.step(0.30, 0.10, False, False, foot_l=0.03, foot_r=0.03)  # < 0.5 s
    assert _progress(env) == 0.0
    env.step(0.60, 0.10, False, False, foot_l=0.03, foot_r=0.03)  # >= 0.5 s
    assert _progress(env) > 0.0


def test_shallow_crouch_pays_proportionally_less():
    env = _Env()
    for i in range(10):
        env.step(0.4 + i * 0.02, STAND_Z - 0.015, True, True)  # half charge
        _progress(env)
    env.step(0.62, 0.11, False, False, foot_l=0.046, foot_r=0.046)  # 40 mm clearance
    p = _progress(env)
    assert 0.2 < p < 0.8  # smoothstep(0.5) = 0.5 of full 1.0


def test_order_independence():
    env = _Env()
    _charge(env, t0=0.4)
    env.step(0.62, 0.10, False, False, foot_l=0.026, foot_r=0.026)
    m.hop3_completion_gate(env, L, R, STAND_Z, **COMMON)  # gate first
    assert abs(_progress(env) - 0.5) < 1e-6


def _ramp(env):
    return m.hop3_crouch_ramp(env, L, R, STAND_Z, **COMMON).item()


def test_crouch_ramp_tracks_slewed_reference():
    env = _Env()
    env.step(0.0, STAND_Z, True, True)
    assert abs(_ramp(env) - 1.0) < 1e-5            # t=0: ref == standing
    env.step(0.3, STAND_Z - HOP3_CROUCH_DEPTH / 2, True, True)
    assert abs(_ramp(env) - 1.0) < 1e-5            # on the ramp at t=0.3
    env.step(0.3, STAND_Z - HOP3_CROUCH_DEPTH, True, True)
    assert _ramp(env) < 0.15                       # diving ahead of the ramp pays little
    env.step(0.3, STAND_Z, True, True)
    assert _ramp(env) < 0.15                       # lagging also pays little
    env.step(0.9, STAND_Z - HOP3_CROUCH_DEPTH, True, True)
    assert abs(_ramp(env) - 1.0) < 1e-5            # holding the floor in the hold phase


def test_crouch_ramp_closed_outside_window_airborne_and_after_hop():
    env = _Env()
    env.step(HOP3_WINDOW_S + 0.1, STAND_Z - HOP3_CROUCH_DEPTH, True, True)
    assert _ramp(env) == 0.0                       # past the window
    env = _Env()
    env.step(0.3, STAND_Z - HOP3_CROUCH_DEPTH / 2, False, False)
    assert _ramp(env) == 0.0                       # airborne
    env = _Env()
    _charge(env, t0=0.4)
    env.step(0.62, 0.10, False, False, foot_l=0.06, foot_r=0.06)
    _progress(env)                                 # hop completes (gate fully open)
    env.step(0.7, STAND_Z - HOP3_CROUCH_DEPTH, True, True)
    assert _ramp(env) == 0.0                       # no farming after the hop
