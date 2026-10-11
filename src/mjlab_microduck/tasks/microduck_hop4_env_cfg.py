"""Microduck hop, attempt 4 (task Mjlab-Hop4-Flat-MicroDuck).

Built on Hop3. Hop3's final checkpoint (4000 iters, 256-env eval) hops for
real: mean foot clearance 40 mm (p90 52, max 64), 88% of episodes >= 30 mm,
every episode ends upright. Remaining faults, all measured:
  • forward lean of ~16 deg (p90 21) during the crouch — the crouch reward only
    checked trunk height, so dipping by leaning paid as well as squatting;
  • ~27 deg tilt in flight;
  • shallow crouch (min trunk z 87.5 mm vs the 80 mm target) and liftoff at
    0.46 s, quicker than the 0.6 s ramp;
  • the iteration-1000 checkpoint farmed "lying on its back, feet up" as flight.

Changes vs Hop3:
  1. Crouch memory counts only grounded steps with tilt < 20 deg.
  2. Flight clearance needs trunk z > 90 mm and is faded by tilt (full at
     20 deg, zero at 40 deg).
  3. Slewed crouch reward faded by tilt (full at 8 deg, zero at 25 deg).
  4. Slower charge-up: ramp 0.8 s, window 1.6 s, hops before 0.8 s don't count.
Unchanged: tuck, head hold, curricula, entropy_coef 0.02, obs/action dims.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop3_env_cfg import (
    HOP3_CROUCH_DEPTH,
    HOP3_CROUCH_FULL_DEPTH,
    HOP3_CROUCH_STD,
    HOP3_CROUCH_WEIGHT,
    HOP3_FOOT_REST_Z,
    HOP3_GATE_WIDTH,
    HOP3_MAX_PAID_CLEARANCE,
    HOP3_MIN_HOP_HEIGHT,
    MicroduckHop3RlCfg,
    make_microduck_hop3_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop_env_cfg import _LEG_JOINTS, STAND_Z

# ── Tuned constants ───────────────────────────────────────────────────────────
HOP4_MIN_LAUNCH_S = 0.8
HOP4_RAMP_S = 0.8
HOP4_WINDOW_S = 1.6
HOP4_CROUCH_TILT_OK_DEG = 20.0
HOP4_FLIGHT_TILT_FULL_DEG = 20.0
HOP4_FLIGHT_TILT_ZERO_DEG = 40.0
HOP4_MIN_FLIGHT_Z = 0.09
HOP4_RAMP_TILT_FULL_DEG = 8.0
HOP4_RAMP_TILT_ZERO_DEG = 25.0

_LEFT = "left_foot_ground_contact"
_RIGHT = "right_foot_ground_contact"
_COMMON = {
    "foot_rest_z": HOP3_FOOT_REST_Z,
    "crouch_full_depth": HOP3_CROUCH_FULL_DEPTH,
    "min_launch_s": HOP4_MIN_LAUNCH_S,
    "crouch_tilt_ok_deg": HOP4_CROUCH_TILT_OK_DEG,
    "flight_tilt_full_deg": HOP4_FLIGHT_TILT_FULL_DEG,
    "flight_tilt_zero_deg": HOP4_FLIGHT_TILT_ZERO_DEG,
    "min_flight_z": HOP4_MIN_FLIGHT_Z,
}
_GATE = {"min_hop_height": HOP3_MIN_HOP_HEIGHT, "gate_width": HOP3_GATE_WIDTH}


def make_microduck_hop4_env_cfg(play: bool = False):
    cfg = make_microduck_hop3_env_cfg(play=play)

    cfg.rewards["hop_airborne_progress"] = RewardTermCfg(
        func=microduck_mdp.hop4_airborne_progress,
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
        func=microduck_mdp.hop4_crouch_ramp,
        weight=HOP3_CROUCH_WEIGHT,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "standing_height": STAND_Z,
            "crouch_depth": HOP3_CROUCH_DEPTH,
            "ramp_s": HOP4_RAMP_S,
            "window_s": HOP4_WINDOW_S,
            "std": HOP3_CROUCH_STD,
            "tilt_full_deg": HOP4_RAMP_TILT_FULL_DEG,
            "tilt_zero_deg": HOP4_RAMP_TILT_ZERO_DEG,
            **_COMMON,
            **_GATE,
        },
    )
    cfg.rewards["hop_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.hop4_landing_composite,
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
        func=microduck_mdp.hop4_stand_tax,
        weight=0.0,
        params={
            "left_sensor_name": _LEFT,
            "right_sensor_name": _RIGHT,
            "target_height": STAND_Z,
            **_COMMON,
            **_GATE,
        },
    )
    cfg.events["set_hop_state"].func = microduck_mdp.reset_hop4_state
    return cfg


MicroduckHop4RlCfg = deepcopy(MicroduckHop3RlCfg)
MicroduckHop4RlCfg.experiment_name = "microduck_hop4"
MicroduckHop4RlCfg.run_name = "microduck_hop4"
