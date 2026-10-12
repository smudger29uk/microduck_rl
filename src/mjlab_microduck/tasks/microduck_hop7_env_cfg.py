"""Microduck hop, attempt 7 (task Mjlab-Hop7-Flat-MicroDuck).

Built on Hop5/Hop6 (balanced squat reference, upright gates). Viewing Hop6
model_5998 (headless eval + by eye): it squats to ~84 mm, then makes a ~4 step
hop with the feet kicked back and toes pointing up; trunk rise above standing
height is only 4.6 mm mean (11 mm best); the head ends pitched down.

Cause: flight is scored as FOOT clearance (min of both foot geoms). Flicking or
tucking the feet up scores without the trunk rising, hop_tuck pays the
tucked pose while airborne, and head_pose_l1 (1.0) is cheaper than the
counterweight it buys.

Changes vs Hop5/Hop6:
  1. Flight = min(foot clearance, trunk z - STAND_Z). Tucking the feet up
     cannot pay unless the trunk rises as well (reset_hop7_state sets the flag).
  2. Lower/narrower gate (0.004 + 0.006) and cap 0.03 so a 5-10 mm trunk rise
     already has gradient and more height keeps paying up to 30 mm.
  3. hop_tuck target = HOME (empty overrides): legs stay extended and under
     the body while airborne, instead of a knee-tuck / heel kick.
  4. head_pose_l1 weight 1.0 -> 4.0 (head must stay level).
Unchanged: crouch pose reference, tilt gates, curricula, obs/action dims.

Suggested runs (UNVERIFIED in closed loop):
  warm   uv run train Mjlab-Hop7-Flat-MicroDuck --env.scene.num-envs 2048 \\
             --agent.max_iterations 4000 --agent.run-name hop7-warm \\
             --agent.resume True --agent.load-run <hop6 run folder> \\
             --agent.load-checkpoint model_5998.pt   (copy it under microduck_hop7/ first)
  scratch  same without --agent.resume, --agent.max_iterations 6000.
"""

from copy import deepcopy

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop5_env_cfg import (
    MicroduckHop5RlCfg,
    make_microduck_hop5_env_cfg,
)

HOP7_MAX_PAID_HEIGHT = 0.03
HOP7_MIN_HOP_HEIGHT = 0.004
HOP7_GATE_WIDTH = 0.006
HOP7_HEAD_POSE_WEIGHT = 4.0


def make_microduck_hop7_env_cfg(play: bool = False):
    cfg = make_microduck_hop5_env_cfg(play=play)

    for term in cfg.rewards.values():
        params = term.params if isinstance(term.params, dict) else {}
        if "min_hop_height" in params:
            params["min_hop_height"] = HOP7_MIN_HOP_HEIGHT
            params["gate_width"] = HOP7_GATE_WIDTH
    cfg.rewards["hop_airborne_progress"].params["max_paid_height"] = HOP7_MAX_PAID_HEIGHT
    cfg.rewards["hop_tuck"].params["tuck_overrides"] = {}  # HOME: legs extended
    cfg.rewards["head_pose_l1"].weight = HOP7_HEAD_POSE_WEIGHT
    cfg.events["set_hop_state"].func = microduck_mdp.reset_hop7_state
    return cfg


MicroduckHop7RlCfg = deepcopy(MicroduckHop5RlCfg)
MicroduckHop7RlCfg.experiment_name = "microduck_hop7"
MicroduckHop7RlCfg.run_name = "microduck_hop7"
