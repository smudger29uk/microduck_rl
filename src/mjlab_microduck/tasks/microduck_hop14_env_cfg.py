"""Microduck hop, attempt 14 (task Mjlab-Hop14-Flat-MicroDuck).

Built on Hop12 (NOT Hop13). Hop12 model_37747: foot clearance 35.6 mm, but it
jumps backwards. scripts/hop_push.py: feet stay planted through the ~11-step
push, the body is thrown backwards over them (trunk -20 mm vs feet, trunk vx at
liftoff -0.41 m/s vs +0.6 m/s up). Hop13 penalised drift only in the air, after
direction was already set, and flying less was the cheapest way to cut it.

Change vs Hop12 (everything else identical):
  New hop_push_drift penalty: -|v_xy| (clamped 0.5 m/s) while a foot is on the
  ground and the trunk is rising (vz > 0.05 m/s), POSITIVE weight 6.0
  (self-negating). Zero in flight and in the crouch, so it cannot be dodged by
  flying less.
Warm start from Hop12 model_37747. UNVERIFIED in closed loop. Success:
liftoff vx -0.41 -> ~0 (scripts/hop_push.py), foot clearance ~35 mm, flight
>= 96%, forward displacement (scripts/hop_eval.py) from -44 mm towards 0.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop12_env_cfg import (
    MicroduckHop12RlCfg,
    make_microduck_hop12_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop4_env_cfg import _LEFT, _RIGHT

HOP14_PUSH_DRIFT_WEIGHT = 6.0


def make_microduck_hop14_env_cfg(play: bool = False):
    cfg = make_microduck_hop12_env_cfg(play=play)
    cfg.rewards["hop_push_drift"] = RewardTermCfg(
        func=microduck_mdp.hop14_push_drift_penalty,
        weight=HOP14_PUSH_DRIFT_WEIGHT,  # POSITIVE: the function is self-negating
        params={"left_sensor_name": _LEFT, "right_sensor_name": _RIGHT},
    )
    return cfg


MicroduckHop14RlCfg = deepcopy(MicroduckHop12RlCfg)
MicroduckHop14RlCfg.experiment_name = "microduck_hop14"
MicroduckHop14RlCfg.run_name = "microduck_hop14"
