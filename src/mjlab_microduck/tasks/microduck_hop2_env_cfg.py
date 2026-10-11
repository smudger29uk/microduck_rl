"""Microduck hop, attempt 2 (task Mjlab-Hop2-Flat-MicroDuck).

Built on make_microduck_hop_env_cfg so DR / obs / sensors / events stay in
sync with the original; the original task is left untouched for comparison.
Obs (61D actor / 74D critic) and actions (14) are unchanged, so original
checkpoints are shape-compatible, but the intent is a from-scratch run (the
original's action std collapsed to 0.17).

Changes vs Mjlab-Hop-Flat-MicroDuck (diagnosis in HANDOFF.md):
  1. Flight height is trunk rise above the TAKEOFF z (last grounded step)
     while BOTH feet are off the ground, not trunk z minus fixed STAND_Z.
  2. Higher ceilings: paid cap 0.035 -> 0.06; gate opens at 0.015, full at 0.025.
  3. hop_crouch (hard z<0.095 gate, logged 0.0) -> dense clamped depth reward,
     faded out once the hop completes; height_stand_l1 0.6 -> 0.2 so it no
     longer fights the crouch.
  4. Head/neck (joints 5-8) held at HOME by an L1 pose penalty (head drift).
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: F401

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    _LEG_JOINTS,
    STAND_Z,
    MicroduckHopRlCfg,
    make_microduck_hop_env_cfg,
)

# ── Tuned constants ───────────────────────────────────────────────────────────
HOP2_MAX_PAID_HEIGHT = 0.06
HOP2_MIN_HEIGHT = 0.015
HOP2_GATE_WIDTH = 0.01
HOP2_CROUCH_FULL_DEPTH = 0.04   # trunk this far below STAND_Z = full crouch reward
HOP2_CROUCH_WEIGHT = 1.5
HOP2_HEIGHT_STAND_L1_WEIGHT = 0.2
HOP2_HEAD_JOINTS = [5, 6, 7, 8]  # neck_pitch, head_pitch, head_yaw, head_roll
HOP2_HEAD_POSE_WEIGHT = 0.3      # pose_l1_penalty is self-negating -> positive

_LEFT = "left_foot_ground_contact"
_RIGHT = "right_foot_ground_contact"


def make_microduck_hop2_env_cfg(play: bool = False):
    cfg = make_microduck_hop_env_cfg(play=play)

    cfg.rewards["hop_airborne_progress"] = RewardTermCfg(
        func=microduck_mdp.hop2_airborne_progress,
        weight=10.0,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "max_paid_height": HOP2_MAX_PAID_HEIGHT,
        },
    )

    cfg.rewards["hop_crouch"] = RewardTermCfg(
        func=microduck_mdp.hop2_crouch_depth,
        weight=HOP2_CROUCH_WEIGHT,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "standing_height": STAND_Z,
            "full_depth": HOP2_CROUCH_FULL_DEPTH,
            "min_hop_height": HOP2_MIN_HEIGHT,
            "gate_width": HOP2_GATE_WIDTH,
        },
    )

    cfg.rewards["hop_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.hop2_landing_composite,
        weight=4.0,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "min_hop_height": HOP2_MIN_HEIGHT,
            "gate_width": HOP2_GATE_WIDTH,
            "target_overrides": None,
        },
    )

    # Weight stays 0 here; the inherited hop_stand_tax_weight curriculum ramps it.
    cfg.rewards["hop_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.hop2_stand_tax,
        weight=0.0,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "target_height": STAND_Z,
            "min_hop_height": HOP2_MIN_HEIGHT,
            "gate_width": HOP2_GATE_WIDTH,
        },
    )

    cfg.rewards["height_stand_l1"].weight = HOP2_HEIGHT_STAND_L1_WEIGHT

    cfg.rewards["head_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=HOP2_HEAD_POSE_WEIGHT,
        params={"joint_indices": HOP2_HEAD_JOINTS, "target_overrides": None},
    )

    # Spawn event: same standing spawn, plus hop2 state reset.
    ev = cfg.events["set_hop_state"]
    ev.func = microduck_mdp.reset_hop2_state

    return cfg


MicroduckHop2RlCfg = deepcopy(MicroduckHopRlCfg)
MicroduckHop2RlCfg.experiment_name = "microduck_hop2"
MicroduckHop2RlCfg.run_name = "microduck_hop2"
