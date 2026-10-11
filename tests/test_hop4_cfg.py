import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.microduck_hop3_env_cfg import (
    HOP3_CROUCH_DEPTH,
    MicroduckHop3RlCfg,
    make_microduck_hop3_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop4_env_cfg import (
    HOP4_MIN_LAUNCH_S,
    HOP4_RAMP_S,
    HOP4_WINDOW_S,
    MicroduckHop4RlCfg,
    make_microduck_hop4_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop_env_cfg import STAND_Z

L, R = "left_foot_ground_contact", "right_foot_ground_contact"
P = dict(
    foot_rest_z=0.006, crouch_full_depth=0.03, min_launch_s=HOP4_MIN_LAUNCH_S,
    crouch_tilt_ok_deg=20.0, flight_tilt_full_deg=20.0, flight_tilt_zero_deg=40.0,
    min_flight_z=0.09,
)
T0 = HOP4_MIN_LAUNCH_S  # charge for ~0.4 s before this


def test_builds_registered_and_earlier_tasks_untouched():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    assert make_microduck_hop4_env_cfg() is not None
    assert make_microduck_hop4_env_cfg(play=True) is not None
    for t in ("Mjlab-Hop-Flat-MicroDuck", "Mjlab-Hop2-Flat-MicroDuck",
              "Mjlab-Hop3-Flat-MicroDuck", "Mjlab-Hop4-Flat-MicroDuck"):
        assert t in list_tasks()
    c3 = make_microduck_hop3_env_cfg()
    assert c3.rewards["hop_airborne_progress"].func is m.hop3_airborne_progress
    assert c3.rewards["hop_crouch"].func is m.hop3_crouch_ramp
    assert c3.events["set_hop_state"].func is m.reset_hop3_state
    assert MicroduckHop3RlCfg.experiment_name == "microduck_hop3"
    assert MicroduckHop4RlCfg.experiment_name == "microduck_hop4"


def test_hop4_wiring_and_signs():
    cfg = make_microduck_hop4_env_cfg()
    assert cfg.rewards["hop_airborne_progress"].func is m.hop4_airborne_progress
    assert cfg.rewards["hop_crouch"].func is m.hop4_crouch_ramp
    assert cfg.rewards["hop_landing_composite"].func is m.hop4_landing_composite
    assert cfg.rewards["hop_stand_tax"].func is m.hop4_stand_tax
    assert cfg.events["set_hop_state"].func is m.reset_hop4_state
    for n in ("hop_stand_tax", "head_pose_l1", "height_stand_l1", "gentle_landing"):
        assert cfg.rewards[n].weight >= 0
    assert cfg.rewards["hop_crouch"].weight > 0
    keys = ("foot_rest_z", "crouch_full_depth", "min_launch_s", "crouch_tilt_ok_deg",
            "flight_tilt_full_deg", "flight_tilt_zero_deg", "min_flight_z")
    ref = {k: cfg.rewards["hop_airborne_progress"].params[k] for k in keys}
    for n in ("hop_crouch", "hop_landing_composite", "hop_stand_tax"):
        for k in keys:
            assert cfg.rewards[n].params[k] == ref[k], f"{n}.{k} differs"
    for c in cfg.curriculum.values():
        name = c.params.get("reward_name")
        if name:
            assert name in cfg.rewards


def test_charge_up_is_slower_than_hop3():
    assert HOP4_RAMP_S >= 0.8 and HOP4_MIN_LAUNCH_S >= HOP4_RAMP_S
    assert HOP4_WINDOW_S > HOP4_MIN_LAUNCH_S


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
            data=SimpleNamespace(root_link_pos_w=None, geom_pos_w=None, projected_gravity_b=None),
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

    def step(self, t_s, z, left, right, foot=0.006, tilt=0.0):
        self.common_step_counter += 1
        self.episode_length_buf = torch.tensor([round(t_s / self.dt)])
        self.asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, float(z)]])
        self.asset.data.geom_pos_w = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, foot], [0.0, 0.0, foot]]]
        )
        a = math.radians(tilt)
        self.asset.data.projected_gravity_b = torch.tensor([[math.sin(a), 0.0, -math.cos(a)]])
        self.contact[L] = torch.tensor([left])
        self.contact[R] = torch.tensor([right])


def _progress(env):
    return m.hop4_airborne_progress(env, L, R, STAND_Z, max_paid_height=0.04, **P).item()


def _charge(env, tilt=0.0, z=0.08):
    for i in range(10):
        env.step(T0 - 0.3 + i * 0.02, z, True, True, tilt=tilt)
        _progress(env)


def test_upright_charged_hop_pays_clearance():
    env = _Env()
    _charge(env)
    env.step(T0 + 0.1, 0.10, False, False, foot=0.026, tilt=5.0)  # 20 mm clearance
    assert abs(_progress(env) - 0.5) < 1e-6


def test_lying_on_back_feet_up_pays_zero():
    """The iteration-1000 Hop3 hack: trunk on the floor, feet in the air."""
    env = _Env()
    _charge(env)
    env.step(T0 + 0.1, 0.03, False, False, foot=0.06, tilt=80.0)
    assert _progress(env) == 0.0
    # Same height gate even if (implausibly) upright:
    env.step(T0 + 0.12, 0.03, False, False, foot=0.06, tilt=0.0)
    assert _progress(env) == 0.0


def test_leaning_dive_is_not_a_crouch():
    env = _Env()
    _charge(env, tilt=35.0)  # low trunk, but leaning hard
    assert env._hop4_crouch_min.item() == 9.0
    env.step(T0 + 0.1, 0.10, False, False, foot=0.026, tilt=5.0)
    assert _progress(env) == 0.0


def test_flight_tilt_fades_clearance():
    pays = []
    for tilt in (10.0, 30.0, 45.0):
        env = _Env()
        _charge(env)
        env.step(T0 + 0.1, 0.10, False, False, foot=0.046, tilt=tilt)
        pays.append(_progress(env))
    assert pays[0] > pays[1] > 0.0
    assert pays[2] == 0.0


def test_early_launch_and_no_crouch_pay_zero():
    env = _Env()
    _charge(env)
    env.step(T0 - 0.2, 0.10, False, False, foot=0.03, tilt=0.0)
    assert _progress(env) == 0.0
    env = _Env()
    for i in range(10):
        env.step(T0 + i * 0.02, STAND_Z, True, True)
        _progress(env)
    env.step(T0 + 0.3, 0.13, False, False, foot=0.03)
    assert _progress(env) == 0.0


def test_order_independence():
    env = _Env()
    _charge(env)
    env.step(T0 + 0.1, 0.10, False, False, foot=0.026, tilt=5.0)
    m.hop4_completion_gate(env, L, R, STAND_Z, **P)
    assert abs(_progress(env) - 0.5) < 1e-6


def _ramp(env):
    return m.hop4_crouch_ramp(env, L, R, STAND_Z, **P).item()


def test_crouch_ramp_tracks_and_fades_with_tilt():
    env = _Env()
    t = HOP4_RAMP_S / 2
    env.step(t, STAND_Z - HOP3_CROUCH_DEPTH / 2, True, True, tilt=2.0)
    assert abs(_ramp(env) - 1.0) < 1e-5
    env.step(t, STAND_Z - HOP3_CROUCH_DEPTH / 2, True, True, tilt=16.0)
    mid = _ramp(env)
    assert 0.0 < mid < 1.0
    env.step(t, STAND_Z - HOP3_CROUCH_DEPTH / 2, True, True, tilt=30.0)
    assert _ramp(env) == 0.0


def test_crouch_ramp_closed_outside_window_airborne_and_after_hop():
    env = _Env()
    env.step(HOP4_WINDOW_S + 0.1, STAND_Z - HOP3_CROUCH_DEPTH, True, True)
    assert _ramp(env) == 0.0
    env = _Env()
    env.step(0.3, STAND_Z - 0.01, False, False)
    assert _ramp(env) == 0.0
    env = _Env()
    _charge(env)
    env.step(T0 + 0.1, 0.10, False, False, foot=0.06, tilt=0.0)
    _progress(env)
    env.step(T0 + 0.2, STAND_Z - HOP3_CROUCH_DEPTH, True, True)
    assert _ramp(env) == 0.0
