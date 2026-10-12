"""Microduck hop, attempt 8 (task Mjlab-Hop8-Flat-MicroDuck).

Hop7 (warm run, model_9997, headless 256-env eval): crouch, hop and head are
good. Trunk rise above standing 14.8 mm mean (p90 17.7, max 20.1), foot
clearance 15.5 mm mean, 4.4 airborne steps, 83 mm crouch, 5.6 deg tilt in
flight. Target: about +5 mm of trunk and foot height, nothing else changed.

Changes vs Hop7 (rewards/gates/crouch/head otherwise identical):
  1. hop_airborne_progress weight 10 -> 25 and cap 0.03 -> 0.04, so each extra
     mm of rise pays ~1.9x more and the payout does not saturate near 20 mm.
  2. Action-rate penalty final stage -0.3 -> -0.15 (stages 0 / -0.05 / -0.10 /
     -0.15), since a harder push needs bigger, faster joint changes.
Warm start from Hop7 model_9997 (obs/action dims unchanged). UNVERIFIED in
closed loop: whether landings and the level head survive the stronger push.
"""

from copy import deepcopy

from mjlab_microduck.tasks.microduck_hop7_env_cfg import (
    MicroduckHop7RlCfg,
    make_microduck_hop7_env_cfg,
)

HOP8_PROGRESS_WEIGHT = 25.0
HOP8_MAX_PAID_HEIGHT = 0.04
HOP8_ACTION_RATE_STAGES = [
    {"step": 0,          "weight": -0.05},
    {"step": 1500 * 24,  "weight": -0.10},
    {"step": 3000 * 24,  "weight": -0.15},
]


def make_microduck_hop8_env_cfg(play: bool = False):
    cfg = make_microduck_hop7_env_cfg(play=play)
    prog = cfg.rewards["hop_airborne_progress"]
    prog.weight = HOP8_PROGRESS_WEIGHT
    prog.params["max_paid_height"] = HOP8_MAX_PAID_HEIGHT
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = deepcopy(HOP8_ACTION_RATE_STAGES)
    return cfg


MicroduckHop8RlCfg = deepcopy(MicroduckHop7RlCfg)
MicroduckHop8RlCfg.experiment_name = "microduck_hop8"
MicroduckHop8RlCfg.run_name = "microduck_hop8"
