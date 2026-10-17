"""Microduck hop, attempt 10 (task Mjlab-Hop10-Flat-MicroDuck).

Hop9 (long run) landed best (0.11 m/s post-landing speed) but trunk rise sat at
14.7 mm, 1 to 3 mm below Hop8, and the run later collapsed onto its back with
feet up (iterations 26000 to 26500, cause not found).

Changes vs Hop9 (everything else identical):
  1. hop_horizontal_speed weight 3.0 -> 1.0, so the landing stays tidy but the
     penalty no longer taxes the push for extra height.
  2. New fell_over termination (bad_orientation, 1.2 rad ~ 69 deg on the trunk).
     Hops tilt 5 to 10 deg, so this never fires on a real hop; it ends episodes
     that lie on the back, which removes the chance to farm that pose.
Warm start from Hop9 model_22000 (obs/action dims unchanged). UNVERIFIED in
closed loop. Run short (about 2000 its) and eval often with hop_eval.py.
"""

from copy import deepcopy

from mjlab.envs import mdp as base_mdp
from mjlab.managers import SceneEntityCfg, TerminationTermCfg

from mjlab_microduck.tasks.microduck_hop9_env_cfg import (
    MicroduckHop9RlCfg,
    make_microduck_hop9_env_cfg,
)

HOP10_HSPEED_WEIGHT = 1.0
HOP10_FELL_OVER_RAD = 1.2


def make_microduck_hop10_env_cfg(play: bool = False):
    cfg = make_microduck_hop9_env_cfg(play=play)
    cfg.rewards["hop_horizontal_speed"].weight = HOP10_HSPEED_WEIGHT
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=base_mdp.bad_orientation,
        params={
            "limit_angle": HOP10_FELL_OVER_RAD,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    return cfg


MicroduckHop10RlCfg = deepcopy(MicroduckHop9RlCfg)
MicroduckHop10RlCfg.experiment_name = "microduck_hop10"
MicroduckHop10RlCfg.run_name = "microduck_hop10"
