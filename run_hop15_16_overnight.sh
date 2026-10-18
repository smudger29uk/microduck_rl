#!/bin/bash
# Sequential 5000-iteration warm runs from Hop12 model_37747 (8 GB GPU: one at a time).
cd "$(dirname "$0")"
for n in 15 16; do
  uv run train Mjlab-Hop$n-Flat-MicroDuck --env.scene.num-envs 2048 --agent.max_iterations 5000 \
    --agent.run-name hop$n-warm --agent.resume True \
    --agent.load-run prev_src --agent.load-checkpoint model_37747.pt > logs/hop$n-warm.out 2>&1
done
