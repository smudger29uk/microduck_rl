"""Microduck hop, attempt 12 (task Mjlab-Hop12-Flat-MicroDuck).

Hop11 (night run, model_34748, headless 256-env eval): trunk rise 16.3 mm mean
(p90 19.3), foot clearance 17.3 mm mean, crouch 79 mm. Height has sat at 15 to
16 mm across Hops 8 to 11 whatever the stroke or reward scale. Watching Hop11
(user): "it jumps a bit, but the feet don't leave the floor much". Focus should
be FOOT height off the floor, not final torso height; tucking the feet after
the jump is fine.

Cause in the reward: since Hop7 flight = min(foot clearance, trunk rise above
standing), so foot height can never exceed the trunk's rise, and tucking
cannot pay (a tuck lifts the feet relative to the trunk, which min() ignores).

Changes vs Hop11 (everything else identical):
  1. Flight = foot clearance x smoothstep(trunk rise / 6 mm). The trunk only
     has to have genuinely lifted 6 mm for full credit (Hop11 already does
     16 mm), so a heel kick with no hop still pays nothing, but foot height is
     no longer capped by trunk height. reset_hop12_state sets the flag.
  2. hop_tuck target = the FK-solved balanced squat at knee 1.40 (hip 0.367,
     ankle 1.033; feet ~44 mm closer to the trunk than HOME). Same pose family
     as the crouch, so the landing from it is soft and symmetric. NOT the old
     HOP_TUCK_OVERRIDES (knee -1.3: different sign convention, never FK-checked).
  3. hop_airborne_progress cap 0.04 -> 0.06 and weight 25 -> 37.5 (same pay
     per mm, but it no longer saturates at 40 mm of foot clearance).
Warm start from Hop11 model_34748 (obs/action dims unchanged). UNVERIFIED in
closed loop. Watch for: heel kick / toes up (trunk gate should stop it), head
pitching down in the tuck (head_pose_l1 weight 4 is unchanged), landing from the
tuck. Eval with scripts/hop_eval.py: foot clearance mean/p90 is the headline.
"""

from copy import deepcopy

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop11_env_cfg import (
    MicroduckHop11RlCfg,
    make_microduck_hop11_env_cfg,
)

HOP12_FOOT_GATE_M = 0.006
HOP12_TUCK_OVERRIDES = {
    2:  0.367,   # left  hip_pitch
    3:  1.40,    # left  knee
    4:  1.033,   # left  ankle
    11: -0.367,  # right hip_pitch
    12: -1.40,   # right knee
    13: -1.033,  # right ankle
}
HOP12_MAX_PAID_HEIGHT = 0.06
HOP12_PROGRESS_WEIGHT = 37.5


def make_microduck_hop12_env_cfg(play: bool = False):
    cfg = make_microduck_hop11_env_cfg(play=play)
    cfg.events["set_hop_state"].func = microduck_mdp.reset_hop12_state
    cfg.events["set_hop_state"].params["foot_gate_m"] = HOP12_FOOT_GATE_M
    cfg.rewards["hop_tuck"].params["tuck_overrides"] = dict(HOP12_TUCK_OVERRIDES)
    prog = cfg.rewards["hop_airborne_progress"]
    prog.weight = HOP12_PROGRESS_WEIGHT
    prog.params["max_paid_height"] = HOP12_MAX_PAID_HEIGHT
    return cfg


MicroduckHop12RlCfg = deepcopy(MicroduckHop11RlCfg)
MicroduckHop12RlCfg.experiment_name = "microduck_hop12"
MicroduckHop12RlCfg.run_name = "microduck_hop12"
