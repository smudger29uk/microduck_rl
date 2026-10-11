from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.microduck_hop2_env_cfg import (
    HOP2_HEAD_JOINTS,
    HOP2_MAX_PAID_HEIGHT,
    HOP2_MIN_HEIGHT,
    make_microduck_hop2_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    MAX_PAID_HOP_HEIGHT,
    STAND_Z,
    make_microduck_hop_env_cfg,
)

L, R = "left_foot_ground_contact", "right_foot_ground_contact"


def test_builds_and_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    assert make_microduck_hop2_env_cfg() is not None
    assert make_microduck_hop2_env_cfg(play=True) is not None
    assert "Mjlab-Hop2-Flat-MicroDuck" in list_tasks()
    assert "Mjlab-Hop-Flat-MicroDuck" in list_tasks()


def test_original_task_untouched():
    cfg = make_microduck_hop_env_cfg()
    assert cfg.rewards["hop_airborne_progress"].func is m.hop_airborne_progress
    assert cfg.rewards["hop_crouch"].func is m.hop_crouch_reward
    assert cfg.rewards["height_stand_l1"].weight == 0.6
    assert "head_pose_l1" not in cfg.rewards
    assert cfg.events["set_hop_state"].func is m.reset_hop_state
    assert MAX_PAID_HOP_HEIGHT == 0.035


def test_hop2_uses_new_funcs_and_ceilings():
    cfg = make_microduck_hop2_env_cfg()
    assert cfg.rewards["hop_airborne_progress"].func is m.hop2_airborne_progress
    assert cfg.rewards["hop_crouch"].func is m.hop2_crouch_depth
    assert cfg.rewards["hop_landing_composite"].func is m.hop2_landing_composite
    assert cfg.rewards["hop_stand_tax"].func is m.hop2_stand_tax
    assert cfg.events["set_hop_state"].func is m.reset_hop2_state
    assert cfg.rewards["hop_airborne_progress"].params["max_paid_height"] == HOP2_MAX_PAID_HEIGHT > 0.035
    assert cfg.rewards["hop_landing_composite"].params["min_hop_height"] == HOP2_MIN_HEIGHT
    assert cfg.rewards["height_stand_l1"].weight == 0.2


def test_head_penalty_sign_and_joints():
    cfg = make_microduck_hop2_env_cfg()
    t = cfg.rewards["head_pose_l1"]
    assert t.func is m.pose_l1_penalty
    assert t.params["joint_indices"] == HOP2_HEAD_JOINTS == [5, 6, 7, 8]
    assert t.weight > 0  # self-negating penalty -> positive weight


def test_self_negating_weights_nonnegative_and_curricula_resolve():
    cfg = make_microduck_hop2_env_cfg()
    for n in ("hop_stand_tax", "head_pose_l1", "height_stand_l1", "gentle_landing"):
        assert cfg.rewards[n].weight >= 0
    for c in cfg.curriculum.values():
        name = c.params.get("reward_name")
        if name:
            assert name in cfg.rewards


def test_obs_dims_and_runner_names():
    from mjlab_microduck.tasks.microduck_hop2_env_cfg import MicroduckHop2RlCfg
    from mjlab_microduck.tasks.microduck_hop_env_cfg import MicroduckHopRlCfg

    assert MicroduckHop2RlCfg.experiment_name == "microduck_hop2"
    assert MicroduckHopRlCfg.experiment_name == "microduck_hop"  # not mutated


# ── functional tests with a fake env ──────────────────────────────────────────

class _Env:
    def __init__(self, n=1):
        self.num_envs, self.device = n, "cpu"
        self.common_step_counter = 0
        self.z = torch.full((n,), STAND_Z)
        self.contact = {L: torch.ones(n, dtype=torch.bool), R: torch.ones(n, dtype=torch.bool)}
        env = self
        asset = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=None))
        self.asset = asset

        class Sensors(dict):
            def __contains__(s, k):
                return k in env.contact

            def __getitem__(s, k):
                return SimpleNamespace(data=SimpleNamespace(found=env.contact[k].float().view(-1, 1)))

        class Scene:
            sensors = Sensors()
            terrain = SimpleNamespace(env_origins=torch.zeros(n, 3))

            def __getitem__(s, k):
                return asset

        self.scene = Scene()

    def step(self, z, left, right):
        self.common_step_counter += 1
        self.z = torch.tensor([float(z)])
        self.asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, float(z)]])
        self.contact[L] = torch.tensor([left])
        self.contact[R] = torch.tensor([right])


def _progress(env):
    return m.hop2_airborne_progress(env, L, R, max_paid_height=0.06).item()


def test_straightening_with_feet_down_pays_nothing():
    env = _Env()
    env.step(0.09, True, True)
    _progress(env)
    for _ in range(5):
        env.step(0.13, True, True)  # rises well above STAND_Z, but grounded
        assert _progress(env) == 0.0
    assert env._hop2_peak.item() == 0.0


def test_single_foot_clearance_pays_nothing():
    env = _Env()
    env.step(0.10, True, True)
    _progress(env)
    env.step(0.14, True, False)
    assert _progress(env) == 0.0


def test_flight_measured_from_takeoff_not_stand_z():
    env = _Env()
    env.step(0.10, True, True)       # crouched takeoff z = 0.10
    _progress(env)
    env.step(0.12, False, False)     # airborne, rise 0.02 (old metric: 0.005)
    assert abs(_progress(env) - 0.02 / 0.06) < 1e-6
    env.step(0.16, False, False)     # peak rise 0.06 -> capped, pays the rest
    assert abs(_progress(env) - 0.04 / 0.06) < 1e-6
    env.step(0.20, False, False)     # beyond cap pays nothing
    assert _progress(env) < 1e-5
    env.step(0.10, True, True)       # land: peak is kept, nothing more paid
    assert _progress(env) < 1e-5
    assert env._hop2_peak.item() > 0.09


def test_progress_independent_of_term_evaluation_order():
    env = _Env()
    env.step(0.10, True, True)
    _progress(env)
    env.step(0.12, False, False)
    # gate evaluated BEFORE progress in the same step must not eat the delta
    m.hop2_completion_gate(env, L, R)
    assert abs(_progress(env) - 0.02 / 0.06) < 1e-6


def test_gate_and_crouch_fade():
    env = _Env()
    env.step(0.075, True, True)      # 0.04 below STAND_Z -> full depth
    crouch = m.hop2_crouch_depth(env, L, R, STAND_Z, full_depth=0.04).item()
    assert abs(crouch - 1.0) < 1e-5
    assert m.hop2_completion_gate(env, L, R).item() == 0.0
    env.step(0.20, False, False)     # big flight -> gate fully open
    assert m.hop2_completion_gate(env, L, R).item() == 1.0
    env.step(0.075, True, True)
    assert m.hop2_crouch_depth(env, L, R, STAND_Z, full_depth=0.04).item() == 0.0


def test_crouch_zero_when_standing_or_airborne():
    env = _Env()
    env.step(STAND_Z, True, True)
    assert m.hop2_crouch_depth(env, L, R, STAND_Z).item() == 0.0
    env.step(0.08, False, False)
    assert m.hop2_crouch_depth(env, L, R, STAND_Z).item() == 0.0
