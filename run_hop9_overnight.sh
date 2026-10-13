#!/bin/bash
# Waits for the running hop9-warm job to finish, then continues it for 10000 more iterations.
cd ~/robotics/microduck_rl_jump
until grep -aq "ETA: 00:00:0[0-1]" logs/hop9_warm.out; do sleep 30; done
sleep 90   # let the final checkpoint/onnx export finish
D=$(ls -d logs/rsl_rl/microduck_hop9/*hop9-warm | tail -1)
M=$(ls $D | grep -E "^model_[0-9]+\.pt$" | sort -V | tail -1)
mkdir -p logs/rsl_rl/microduck_hop9/hop9_src && cp -r $D/$M $D/params logs/rsl_rl/microduck_hop9/hop9_src/
uv run train Mjlab-Hop9-Flat-MicroDuck --env.scene.num-envs 2048 --agent.max_iterations 10000 \
  --agent.run-name hop9-long --agent.resume True \
  --agent.load-run hop9_src --agent.load-checkpoint $M > logs/hop9_long.out 2>&1
