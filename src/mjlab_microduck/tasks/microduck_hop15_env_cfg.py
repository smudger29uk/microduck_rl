"""Microduck hop, attempt 15 (task Mjlab-Hop15-Flat-MicroDuck).

Hop14 (speed penalty in the push, weight 6) cut backwards travel only
-44 -> -31 mm and cost 8 mm foot clearance (35.6 -> 27.3); the push still threw
the trunk -20 mm relative to the feet. Hop15 penalises that OFFSET directly.

Change vs Hop12 (everything else identical):
  New hop_push_offset penalty: -|trunk xy - feet midpoint xy| (clamped 40 mm)
  while a foot is grounded and the trunk is rising, POSITIVE weight 100
  (self-negating; ~2.0 per step at a 20 mm offset, similar mass to Hop14).
Warm start from Hop12 model_37747. UNVERIFIED in closed loop. Success: trunk
rel. feet at liftoff -20 mm -> ~0 (scripts/hop_push.py), foot clearance ~35 mm,
flight >= 96%.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop12_env_cfg import (
    MicroduckHop12RlCfg,
    make_microduck_hop12_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop4_env_cfg import _LEFT, _RIGHT

HOP15_PUSH_OFFSET_WEIGHT = 100.0


def make_microduck_hop15_env_cfg(play: bool = False):
    cfg = make_microduck_hop12_env_cfg(play=play)
    cfg.rewards["hop_push_offset"] = RewardTermCfg(
        func=microduck_mdp.hop15_push_offset_penalty,
        weight=HOP15_PUSH_OFFSET_WEIGHT,  # POSITIVE: the function is self-negating
        params={"left_sensor_name": _LEFT, "right_sensor_name": _RIGHT},
    )
    return cfg


MicroduckHop15RlCfg = deepcopy(MicroduckHop12RlCfg)
MicroduckHop15RlCfg.experiment_name = "microduck_hop15"
MicroduckHop15RlCfg.run_name = "microduck_hop15"
