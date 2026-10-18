"""Microduck hop, attempt 16 (task Mjlab-Hop16-Flat-MicroDuck).

Hop14 with the push-drift weight lowered 6.0 -> 2.0. Hop14 at 6.0 cost 8 mm of
foot clearance for a 13 mm drift gain; this tests whether a lighter tax keeps
most of the height. Identical to Hop14 otherwise (built on Hop12).
Warm start from Hop12 model_37747. UNVERIFIED in closed loop.
"""

from copy import deepcopy

from mjlab_microduck.tasks.microduck_hop14_env_cfg import (
    MicroduckHop14RlCfg,
    make_microduck_hop14_env_cfg,
)

HOP16_PUSH_DRIFT_WEIGHT = 2.0


def make_microduck_hop16_env_cfg(play: bool = False):
    cfg = make_microduck_hop14_env_cfg(play=play)
    cfg.rewards["hop_push_drift"].weight = HOP16_PUSH_DRIFT_WEIGHT
    return cfg


MicroduckHop16RlCfg = deepcopy(MicroduckHop14RlCfg)
MicroduckHop16RlCfg.experiment_name = "microduck_hop16"
MicroduckHop16RlCfg.run_name = "microduck_hop16"
