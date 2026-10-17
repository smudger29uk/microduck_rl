"""Microduck hop, attempt 11 (task Mjlab-Hop11-Flat-MicroDuck).

Hop10 (long run, model_26750, headless 256-env eval): crouch, hop, landing and
head are good (trunk rise 16.0 mm mean, p90 19.1; foot clearance 17.4 mm; real
drift 56 mm, the old "520 mm" was the timeout-reset teleport). Height plateaued
at 15 to 16 mm across 6000 its, so more iterations of the same rewards will not
reach the 20 mm target. The launch reward is not the limit (cap 0.85 m/s, never
binding); the push STROKE is: the crouch is pinned at ~84 mm by the Hop5 squat
reference (32 mm drop) and by crouch_full_depth 0.03 (charge saturates at a
30 mm drop).

Change vs Hop10 (everything else identical):
  1. Deeper balanced squat. Solved by FK in MuJoCo (robot_groundcontact.xml,
     sole flat, trunk upright, CoM-to-foot offset equal to HOME's; the same
     solve reproduces Hop5's knee 1.10 -> hip 0.166 / ankle 0.934 / 32 mm drop):
         knee 1.30, hip_pitch +0.297, ankle 1.003  ->  ~40 mm trunk drop
     (left leg; right mirrored). hop_crouch crouch_depth 0.032 -> 0.040.
  2. crouch_full_depth 0.03 -> 0.04 in every term that carries it, so the
     flight charge keeps growing until the deeper crouch, rather than
     saturating at the old one.
Warm start from Hop10 model_26750 (obs/action dims unchanged). UNVERIFIED in
closed loop: whether the policy takes the longer stroke or just loses reward
on the tighter charge. Run short (about 2000 its) and eval every 500; compare
trunk rise mean/p90 and the crouch min trunk z (target ~75 mm).
"""

from copy import deepcopy

from mjlab_microduck.tasks.microduck_hop10_env_cfg import (
    MicroduckHop10RlCfg,
    make_microduck_hop10_env_cfg,
)

HOP11_SQUAT_OVERRIDES = {
    2:  0.297,   # left  hip_pitch
    3:  1.30,    # left  knee
    4:  1.003,   # left  ankle
    11: -0.297,  # right hip_pitch
    12: -1.30,   # right knee
    13: -1.003,  # right ankle
}
HOP11_CROUCH_DEPTH = 0.040
HOP11_CROUCH_FULL_DEPTH = 0.04


def make_microduck_hop11_env_cfg(play: bool = False):
    cfg = make_microduck_hop10_env_cfg(play=play)
    crouch = cfg.rewards["hop_crouch"]
    crouch.params["squat_overrides"] = dict(HOP11_SQUAT_OVERRIDES)
    crouch.params["crouch_depth"] = HOP11_CROUCH_DEPTH
    for term in cfg.rewards.values():
        params = term.params if isinstance(term.params, dict) else {}
        if "crouch_full_depth" in params:
            params["crouch_full_depth"] = HOP11_CROUCH_FULL_DEPTH
    return cfg


MicroduckHop11RlCfg = deepcopy(MicroduckHop10RlCfg)
MicroduckHop11RlCfg.experiment_name = "microduck_hop11"
MicroduckHop11RlCfg.run_name = "microduck_hop11"
