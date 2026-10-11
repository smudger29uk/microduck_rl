"""Microduck hop, attempt 5 (task Mjlab-Hop5-Flat-MicroDuck).

Built on Hop4 (upright gates, slower charge-up). Hop4 model_500 trace showed a
height-only crouch reward lets the policy squat badly: hip yaw splayed ~0.35 rad,
legs asymmetric, neck swung 0.68 rad as a counterweight.

Root cause found by FK (not a guess): the old crouch target (hip -0.65, knee
0.79, ankle 1.44) is flat-soled but puts the CoM ~5 mm BEHIND the heel, so any
policy tracking it has to throw the head forward. The balanced flat-footed
squat (sole flat, trunk upright, CoM-to-foot offset equal to HOME's) is
solved per knee angle:

    knee 1.10, hip_pitch +0.16, ankle 0.92  ->  32 mm trunk drop

(left leg; right mirrored). Hop4 model_1000's left leg had found almost exactly
this by itself (hip 0.25 / knee 1.12 / ankle 0.93).

Changes vs Hop4:
  1. hop_crouch tracks a slewed joint-POSE reference (HOME -> balanced squat,
     all 10 leg joints incl. hip yaw/roll -> 0, both legs symmetric) as well as
     the height ramp (depth 0.032 to match the pose's drop).
  2. head_pose_l1 weight 0.3 -> 1.0 (the balanced squat does not need the head
     as a counterweight, so the penalty no longer fights physics).
Unchanged: all Hop4 gates, tuck, curricula, entropy_coef 0.02, obs/action dims.
UNVERIFIED in closed loop: whether the policy can hold this pose dynamically.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop3_env_cfg import (
    HOP3_CROUCH_STD,
    HOP3_CROUCH_WEIGHT,
)
from mjlab_microduck.tasks.microduck_hop4_env_cfg import (
    _COMMON,
    _GATE,
    _LEFT,
    _RIGHT,
    HOP4_RAMP_S,
    HOP4_RAMP_TILT_FULL_DEG,
    HOP4_RAMP_TILT_ZERO_DEG,
    HOP4_WINDOW_S,
    MicroduckHop4RlCfg,
    make_microduck_hop4_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop_env_cfg import _LEG_JOINTS, STAND_Z

# Balanced flat-footed squat (left leg; right mirrored). Solved by FK, see above.
HOP5_SQUAT_OVERRIDES = {
    2:  0.16,   # left  hip_pitch
    3:  1.10,   # left  knee
    4:  0.92,   # left  ankle
    11: -0.16,  # right hip_pitch
    12: -1.10,  # right knee
    13: -0.92,  # right ankle
}
HOP5_CROUCH_DEPTH = 0.032
HOP5_POSE_STD = 0.3
HOP5_HEAD_POSE_WEIGHT = 1.0


def make_microduck_hop5_env_cfg(play: bool = False):
    cfg = make_microduck_hop4_env_cfg(play=play)

    cfg.rewards["hop_crouch"] = RewardTermCfg(
        func=microduck_mdp.hop5_crouch_pose_ramp,
        weight=HOP3_CROUCH_WEIGHT,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "standing_height": STAND_Z,
            "joint_indices": _LEG_JOINTS,
            "squat_overrides": HOP5_SQUAT_OVERRIDES,
            "crouch_depth": HOP5_CROUCH_DEPTH,
            "ramp_s": HOP4_RAMP_S,
            "window_s": HOP4_WINDOW_S,
            "height_std": HOP3_CROUCH_STD,
            "pose_std": HOP5_POSE_STD,
            "tilt_full_deg": HOP4_RAMP_TILT_FULL_DEG,
            "tilt_zero_deg": HOP4_RAMP_TILT_ZERO_DEG,
            **_COMMON,
            **_GATE,
        },
    )
    cfg.rewards["head_pose_l1"].weight = HOP5_HEAD_POSE_WEIGHT
    return cfg


MicroduckHop5RlCfg = deepcopy(MicroduckHop4RlCfg)
MicroduckHop5RlCfg.experiment_name = "microduck_hop5"
MicroduckHop5RlCfg.run_name = "microduck_hop5"
