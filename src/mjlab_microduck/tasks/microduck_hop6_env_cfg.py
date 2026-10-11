"""Microduck hop, attempt 6 (task Mjlab-Hop6-Flat-MicroDuck).

Hop4 learned a real hop (27 mm clearance, upright) but with a bad squat
(hip splay, asymmetric legs, neck counterweight). Hop5 learned the balanced
squat but never launches (0.6 mm clearance at iteration 3500): the squat
reward is fully paid by holding the pose. Hop6 combines them: IDENTICAL
rewards/gates to Hop5, intended to be WARM-STARTED from Hop4 model_3999 so the
policy already knows how to launch and the squat pose reference reshapes the
crouch it launches from.

Warm start (obs/action dims are unchanged):
    uv run train Mjlab-Hop6-Flat-MicroDuck --env.scene.num-envs 2048 \
        --agent.max_iterations 2000 --agent.run-name hop6-local \
        --agent.resume True --agent.load-run <hop4 run folder> \
        --agent.load-checkpoint model_3999.pt
(copy/symlink the Hop4 run folder under logs/rsl_rl/microduck_hop6/ first.)
Curricula are env-step based and restart from 0 on resume, which is benign.
UNVERIFIED in closed loop: whether the hop survives the squat reference.
"""

from copy import deepcopy

from mjlab_microduck.tasks.microduck_hop5_env_cfg import (
    MicroduckHop5RlCfg,
    make_microduck_hop5_env_cfg,
)


def make_microduck_hop6_env_cfg(play: bool = False):
    return make_microduck_hop5_env_cfg(play=play)


MicroduckHop6RlCfg = deepcopy(MicroduckHop5RlCfg)
MicroduckHop6RlCfg.experiment_name = "microduck_hop6"
MicroduckHop6RlCfg.run_name = "microduck_hop6"
