"""Microduck hop, attempt 13 (task Mjlab-Hop13-Flat-MicroDuck).

Hop12 (warm run, model_37747, headless 256-env eval): foot clearance 35.6 mm
mean (p90 42.1), trunk rise 14.3 mm, 96% of episodes fly. The user: "it jumps
backwards a bit rather than straight up". Measured (scripts/hop_drift.py style,
body-frame along the initial heading): net -45 mm at the end (p10 -78, p90
-3), -18 mm already at first liftoff and ~-20 mm more during flight. model_36750
is the same (-44 mm), so it is the learned launch, not a late drift.

The existing hop_horizontal_speed penalty (weight 1.0, every step, symmetric)
is too small a part of the stack to shape the launch direction.

Change vs Hop12 (everything else identical):
  New hop_airborne_drift penalty: -|v_xy| (clamped 0.5 m/s), only while both
  feet are off the ground, POSITIVE weight 6.0 (self-negating). Horizontal
  speed at takeoff carries the trunk sideways through the whole flight, so
  this taxes a leaning launch directly and a vertical one is free.
Warm start from Hop12 model_37747 (obs/action dims unchanged). UNVERIFIED in
closed loop. Watch: foot clearance must not fall back (Hop12 35.6 mm), flight
rate must stay ~96%+, and the body-frame forward displacement at touchdown
(scripts/hop_eval.py now prints it) should move from -39 mm towards 0.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop12_env_cfg import (
    MicroduckHop12RlCfg,
    make_microduck_hop12_env_cfg,
)
from mjlab_microduck.tasks.microduck_hop4_env_cfg import _LEFT, _RIGHT

HOP13_AIRBORNE_DRIFT_WEIGHT = 6.0


def make_microduck_hop13_env_cfg(play: bool = False):
    cfg = make_microduck_hop12_env_cfg(play=play)
    cfg.rewards["hop_airborne_drift"] = RewardTermCfg(
        func=microduck_mdp.hop13_airborne_drift_penalty,
        weight=HOP13_AIRBORNE_DRIFT_WEIGHT,  # POSITIVE: the function is self-negating
        params={"left_sensor_name": _LEFT, "right_sensor_name": _RIGHT},
    )
    return cfg


MicroduckHop13RlCfg = deepcopy(MicroduckHop12RlCfg)
MicroduckHop13RlCfg.experiment_name = "microduck_hop13"
MicroduckHop13RlCfg.run_name = "microduck_hop13"
