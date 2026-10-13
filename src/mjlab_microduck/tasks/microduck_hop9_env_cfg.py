"""Microduck hop, attempt 9 (task Mjlab-Hop9-Flat-MicroDuck).

Hop8 (warm2, model_15995, headless 256-env eval): crouch, hop and head are
good. Trunk rise 15.8 mm mean, foot clearance 19.3 mm, but the landing is
poor: net horizontal drift 512 mm over the episode (p90 872), peak speed after
the flight 0.33 m/s (p90 0.53), 9.4 deg tilt after the flight. Watching it, the
duck stumbles forward on landing.

Why height is stuck: hop_launch_velocity is capped at max_vz 0.6 m/s, but a
20 mm rise needs ~0.63 m/s at takeoff (v = sqrt(2 g h)), so the cap stops
paying right where more height starts.

Changes vs Hop8 (everything else identical):
  1. hop_launch_velocity max_vz 0.6 -> 0.85 and weight 2 -> 3.
  2. New hop_horizontal_speed penalty (-|v_xy|, positive weight 3.0): a vertical
     hop that stays put is free, stumbling or walking off costs.
Warm start from Hop8 model_15995. UNVERIFIED in closed loop.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop8_env_cfg import (
    MicroduckHop8RlCfg,
    make_microduck_hop8_env_cfg,
)

HOP9_MAX_VZ = 0.85
HOP9_LAUNCH_WEIGHT = 3.0
HOP9_HSPEED_WEIGHT = 3.0


def make_microduck_hop9_env_cfg(play: bool = False):
    cfg = make_microduck_hop8_env_cfg(play=play)
    launch = cfg.rewards["hop_launch_velocity"]
    launch.weight = HOP9_LAUNCH_WEIGHT
    launch.params["max_vz"] = HOP9_MAX_VZ
    cfg.rewards["hop_horizontal_speed"] = RewardTermCfg(
        func=microduck_mdp.hop9_horizontal_speed_penalty,
        weight=HOP9_HSPEED_WEIGHT,  # POSITIVE: the function is self-negating
        params={},
    )
    return cfg


MicroduckHop9RlCfg = deepcopy(MicroduckHop8RlCfg)
MicroduckHop9RlCfg.experiment_name = "microduck_hop9"
MicroduckHop9RlCfg.run_name = "microduck_hop9"
