#!/bin/bash
cd ~/robotics/microduck_rl_jump
uv run train Mjlab-Hop7-Flat-MicroDuck --env.scene.num-envs 2048 --agent.max_iterations 4000 \
  --agent.run-name hop7-warm --agent.resume True \
  --agent.load-run hop6_src --agent.load-checkpoint model_5998.pt > logs/hop7_warm.out 2>&1
uv run train Mjlab-Hop7-Flat-MicroDuck --env.scene.num-envs 2048 --agent.max_iterations 6000 \
  --agent.run-name hop7-scratch > logs/hop7_scratch.out 2>&1
