"""Microduck hop, attempt 3 (task Mjlab-Hop3-Flat-MicroDuck).

Built on the Hop2 cfg (itself built on the original), so DR / obs / sensors
stay in sync and both earlier tasks remain reproducible.

Why (headless trace of Hop2 model_3999, one episode): feet left the ground by
only ~13 mm for ~0.1 s, the "30 mm trunk rise" was inflated by the legs
unfolding in the air, the crouch was a ~0.15 s fold, and the robot then
stood still for 87% of the episode. Intended behaviour: slow visible charge-up
crouch -> push up -> lift legs -> land -> stand.

Changes vs Hop2:
  1. Flight = foot clearance (min of both foot geoms above rest height while
     both feet are off the ground), not trunk rise.
  2. Hop only counts if the trunk was crouched (>= ~25 mm below standing) at
     some grounded point earlier AND episode time >= 0.5 s (hard gates).
  3. hop_crouch = slewed charge-up reference (0.6 s descent, hold to 1.2 s),
     paid only pre-hop, grounded, inside the window.
  4. Higher exploration: entropy_coef 0.01 -> 0.02 (UNTESTED; Hop2's action std
     collapsed to 0.21).
Unchanged: tuck target (kinematically verified: feet 21 mm below trunk vs
109 mm at HOME), head hold, curricula, obs/action dims.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop2_env_cfg import (
    MicroduckHop2RlCfg,
    make_microduck_hop2_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop_env_cfg import _LEG_JOINTS, STAND_Z

# ── Tuned constants ───────────────────────────────────────────────────────────
HOP3_MAX_PAID_CLEARANCE = 0.04   # foot clearance (m) at which progress saturates
HOP3_MIN_HOP_HEIGHT = 0.010      # landing gate opens
HOP3_GATE_WIDTH = 0.008          # ... full at 0.018
HOP3_FOOT_REST_Z = 0.006         # foot-geom centre z at rest (measured 5.0-5.9 mm)
HOP3_CROUCH_FULL_DEPTH = 0.03    # grounded trunk this far below STAND_Z = full charge
HOP3_MIN_LAUNCH_S = 0.5          # hops before this don't count
HOP3_CROUCH_DEPTH = 0.035        # slewed reference floor: z = 0.080 m
HOP3_RAMP_S = 0.6
HOP3_WINDOW_S = 1.2
HOP3_CROUCH_STD = 0.012
HOP3_CROUCH_WEIGHT = 3.0
HOP3_ENTROPY_COEF = 0.02

_LEFT = "left_foot_ground_contact"
_RIGHT = "right_foot_ground_contact"
_COMMON = {
    "foot_rest_z": HOP3_FOOT_REST_Z,
    "crouch_full_depth": HOP3_CROUCH_FULL_DEPTH,
    "min_launch_s": HOP3_MIN_LAUNCH_S,
}
_GATE = {"min_hop_height": HOP3_MIN_HOP_HEIGHT, "gate_width": HOP3_GATE_WIDTH}


def make_microduck_hop3_env_cfg(play: bool = False):
    cfg = make_microduck_hop2_env_cfg(play=play)

    cfg.rewards["hop_airborne_progress"] = RewardTermCfg(
        func=microduck_mdp.hop3_airborne_progress,
        weight=10.0,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "standing_height": STAND_Z,
            "max_paid_height": HOP3_MAX_PAID_CLEARANCE,
            **_COMMON,
        },
    )

    cfg.rewards["hop_crouch"] = RewardTermCfg(
        func=microduck_mdp.hop3_crouch_ramp,
        weight=HOP3_CROUCH_WEIGHT,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "standing_height": STAND_Z,
            "crouch_depth": HOP3_CROUCH_DEPTH,
            "ramp_s": HOP3_RAMP_S,
            "window_s": HOP3_WINDOW_S,
            "std": HOP3_CROUCH_STD,
            **_COMMON,
            **_GATE,
        },
    )

    cfg.rewards["hop_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.hop3_landing_composite,
        weight=4.0,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
            **_COMMON,
            **_GATE,
        },
    )

    # Weight 0 here; the inherited hop_stand_tax_weight curriculum ramps it.
    cfg.rewards["hop_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.hop3_stand_tax,
        weight=0.0,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "target_height": STAND_Z,
            **_COMMON,
            **_GATE,
        },
    )

    cfg.events["set_hop_state"].func = microduck_mdp.reset_hop3_state
    return cfg


MicroduckHop3RlCfg = deepcopy(MicroduckHop2RlCfg)
MicroduckHop3RlCfg.experiment_name = "microduck_hop3"
MicroduckHop3RlCfg.run_name = "microduck_hop3"
MicroduckHop3RlCfg.algorithm.entropy_coef = HOP3_ENTROPY_COEF
