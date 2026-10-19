# Handoff: Microduck hop (jump) training, local on this PC

Last updated 2026-10-18 (Claude Code session). Read this first. All training is LOCAL on this PC. Never use `--hf-jobs`.

Style: UK English, no em dashes, concise, numbered steps, one thing at a time.

## Machine and stack
- Ubuntu PC, Ryzen 7 5800X, 128 GB RAM, RTX 3070 (8 GB), `uv`, Python 3.12 venv in `.venv` (`uv sync`).
- mjlab 1.3.0 (MuJoCo Warp) + rsl_rl PPO, BAM actuator model. Working folder: `~/robotics/microduck_rl_jump`.
- Other folders in `~/robotics`: `microduck_rl` (fork on Pollen's spring_v2, not used here), `microduck` (Rust runtime for the real robot), `jump-src` (original tarball).

## CURRENT STATUS (2026-10-19, read this first)
Nothing is training. Best policy: **Hop16 warm `model_42746`** (task `Mjlab-Hop16-Flat-MicroDuck`, = Hop14 with push-drift weight 2.0, warm from Hop12 37747, 5000 its). ONNX: `checkpoints/hop16-warm-42746/policy.onnx` (also on HF `smudger29uk/microduck-hop-local-runs/hop16-warm-42746`). User watched it in the viewer and is about to try it on the REAL robot; outcome pending. Previous best Hop12 37747 is the fallback.
Published PRIVATE to HF as `smudger29uk/microduck-hop` (episodic, 3.0 s, schema-2 manifest). On the robot: `sudo robotctl policy add hop smudger29uk/microduck-hop`. Button mapping NOT checked yet (the local `microduck` checkout predates the published-policy mechanism). CPU rehearsal (`infer_policy.py --walking ... --new-cmd-obs --foot-friction 2.0 --foot-solref 0.04`) hopped; user saw it.

Hop16 42746: foot clearance 39.0 mm (p90 47, max 60), trunk rise 15.9 mm, 98% fly, crouch 78 mm, forward displacement -42 mm at end. Still jumps backwards: push throws trunk -32 mm vs feet, liftoff vx -0.43 m/s.

| Task | Checkpoint | Foot clearance | Trunk rise | Flight eps | Fwd displacement at end |
|---|---|---|---|---|---|
| Hop12 | warm 37747 | 35.6 mm | 14.3 mm | 96% | -44 mm |
| Hop13 | warm 40746 | 26.4 mm | 12.9 mm | 98% | -50 mm (abandoned) |
| Hop14 | warm 40746 | 27.3 mm | 13.0 mm | 99% | -31 mm (drift weight 6: cut drift, cost 8 mm height) |
| Hop15 | warm 42746 | 22.2 mm | 12.4 mm | 98% | -47 mm (trunk-over-feet offset penalty: feet slid back instead; abandoned) |
| **Hop16** | **warm 42746 (BEST)** | **39.0 mm** | **15.9 mm** | 98% | -42 mm |

Lesson: every attempt that reduced the backwards throw (Hop13, 14, 15) also cut height. The throw looks like a real trade-off in this recipe. Hop14 `hop14_push_drift_penalty` (grounded and rising, |v_xy|) and Hop15 `hop15_push_offset_penalty` are in mdp.py. Overnight script: `run_hop15_16_overnight.sh`.
Next ideas if the real robot shows a problem: check real-robot drift first (sim drift is only ~40 mm); otherwise try weight ~3 to 4 on Hop14 as a middle point, or a launch-direction term in the crouch-to-push transition.

### After a restart (do this first)
```
cd ~/robotics/microduck_rl_jump
uv sync
uv run --with pytest pytest tests/     # expect 303 passed, 1 skipped
```
If that passes, the stack is fine. Then resume from the "Next ideas" line in CURRENT STATUS (Hop14 to 16 are done). Nothing is training after a restart; any run must be relaunched with the commands in "How to run things".

### Backwards-jump diagnosis (measured on Hop12 37747, `scripts/hop_push.py`)
- Feet stay planted through the push (slide < 3 mm). The BODY is thrown backwards over them: during the ~11-step push (crouch bottom ~step 29 -> liftoff ~step 39) trunk moves -20 mm relative to the feet, whole-robot CoM from -4 to -20 mm vs feet, and trunk vx at liftoff is -0.41 m/s (vs ~0.6 m/s up). That carries it backwards through the flight. Pitch stays within ~2.4 deg, so it is translation, not a lean.
- Hop13 failed because it acted only in the air, after direction was set at liftoff, and flying less was the cheapest way to cut the penalty.
- CAVEAT: whole-robot CoM mass indexing was not fully verified (mass array has 17 entries for 15 bodies); trunk and CoM agree so the direction is trustworthy, exact mm are not.

### Next step (DONE as Hop14, 15, 16; kept for history): Hop14
Build on Hop12 (not Hop13). Penalise horizontal trunk speed while GROUNDED and rising (the push phase), self-negating so POSITIVE weight. Must not reward flying less. Warm start from Hop12 `model_37747`, ~3000 its, eval every 500 with `scripts/hop_eval.py` (now prints forward displacement, negative = backwards) and `scripts/hop_push.py`. Success = liftoff vx -0.41 -> ~0 with foot clearance staying ~35 mm and flight >= 96%. Alternative/companion: reward CoM staying over the feet midpoint through the push.

### Eval tooling
- `scripts/hop_eval.py <task> <ckpt>`: trunk rise, foot clearance, crouch, tilt, drift, forward displacement. Drift bug fixed 2026-10-17 (last step is the timeout-reset teleport; the old "520 mm drift" was an artifact, true drift ~60 mm).
- `scripts/hop_drift.py`: per-phase path length and body-frame direction. `scripts/hop_push.py`: push-phase trunk/feet/CoM breakdown.
- Run folders: `logs/rsl_rl/microduck_hop{10,11,12,13}/` (untracked). Hop12 run = `*hop12-warm`, Hop11 night run = `*hop11-night`.

## Earlier history (hops 1 to 9, as of 2026-10-13)
Tasks `Mjlab-Hop-Flat-MicroDuck` (hop 1, original) and `Mjlab-Hop2..Hop9-Flat-MicroDuck`, each built on the previous one, in `src/mjlab_microduck/tasks/microduck_hopN_env_cfg.py` (docstrings explain each change). Reward functions are in `tasks/mdp.py` (hop section from about line 7191, hop3 to hop5 helpers after it, `hop9_horizontal_speed_penalty` at the end).

Measured with `scripts/hop_eval.py` (256 envs, one 3 s episode each). Trunk rise = trunk z above 115 mm standing height while both feet are off the ground. Foot clearance = lowest foot above rest height.

| Run | Checkpoint | Trunk rise mean | Foot clearance mean | Post-landing speed | Verdict |
|---|---|---|---|---|---|
| Hop 6 warm | model_5998 | 4.6 mm | n/a | n/a | heel kick, toes up, head pitched down |
| Hop 7 warm | model_9997 | 14.8 mm | 15.5 mm | 0.27 m/s | head and crouch good. User: "95% there" |
| Hop 8 warm2 | model_15995 | 15.8 mm | 19.3 mm | 0.33 m/s | highest so far, but stumbles forward on landing (about 0.5 m drift per episode) |
| Hop 9 long | model_22000 | 14.7 mm | 16.6 mm | 0.11 m/s | best stable landing, about 1 to 3 mm lower than hop 8. Viewer shown, user decision pending |

User's goal: about 5 mm more body AND foot height than hop 7 (so trunk about 20 mm), keeping the crouch, hop and level head.

Failures to remember:
- Hop 7 from scratch (6000 its) never learned to hop. Hop 5 from scratch also never launched. Warm start from a policy that already hops.
- Hop 9 long run (10000 its after the first 4000) COLLAPSED between iterations 26000 and 26500: it ends lying on its back with feet up (49 mm foot clearance, 1.8 mm trunk rise). Cause not found. Do not trust the final checkpoint, nor its `.onnx` (written at the end of training). Checkpoints 20250 to 26000 are fine.
- `hop_launch_velocity` was capped at max_vz 0.6 m/s, but a 20 mm rise needs about 0.63 m/s. Hop 9 raised the cap to 0.85, but height did not improve (the speed penalty may be holding it back).

## Hop 10 (2026-10-17; drift numbers in the table are the artifact, see below)
`Mjlab-Hop10-Flat-MicroDuck` = Hop9 + horizontal-speed penalty weight 3.0 -> 1.0 + `fell_over` termination (bad_orientation, 1.2 rad). Warm start from hop 9 model_22000: 2000 its (`hop10-warm`, model_23999), then 4000 more (`hop10-long`, from 23999).

| Checkpoint | Trunk rise | Foot clearance | Post-flight speed | Drift |
|---|---|---|---|---|
| warm 23999 | 15.8 mm | 16.6 mm | 0.16 m/s | 514 mm |
| long 26750 (BEST) | 16.0 mm (p90 19.1, max 21.0) | 17.4 mm | 0.18 m/s | 523 mm |
| long 27998 (final) | 15.0 mm | 15.8 mm | 0.15 m/s | 521 mm |

- No collapse in 6000 its, but height plateaued at 15 to 16 mm (target 20). More iterations of this recipe will not get there.
- The "520 mm drift" quoted for hops 8 to 10 was a MEASUREMENT ARTIFACT: the last step of the 150-step eval is the timeout reset (teleport to a random spawn, ~516 mm). Fixed in `scripts/hop_eval.py` (uses xy[-2]). True drift for hop 10 26750 is 56 mm mean (p90 76, max 227); it sits still before the first flight (~0.78 s) and moves ~60 mm path around the hop. `scripts/hop_drift.py` gives the per-phase breakdown. Drift is NOT a problem; do not chase it.
- ONNX for the best checkpoint: `checkpoints/hop10-long-26750/policy.onnx` (with model and params, untracked). Uploaded to HF: `smudger29uk/microduck-hop-local-runs/hop10-long-26750` (model, params, onnx). Code pushed to GitHub.

## Hop 11 proposal (DONE, see CURRENT STATUS; kept for history)
1. Height: crouch bottoms out at about 84 mm. Try a deeper crouch requirement (raise `crouch_full_depth`/crouch target) so there is more stored push, and a launch reward that keeps paying above 0.63 m/s.
2. Warm start from hop 10 model_26750, 2000 its, eval every 500.

## Next options (older, from 2026-10-13)
1. Hop 10: keep the landing fix but lower the horizontal-speed penalty weight (3.0) or apply it only after flight, and add a guard against the on-back, feet-up hack (for example require trunk z and tilt in the flight score, or terminate on prolonged tilt). Warm start from hop 9 model_22000 or hop 8 model_15995. Run short (about 2000 its) and eval often.
2. Export an ONNX from the chosen checkpoint with `uv run scripts/export.py <task> --checkpoint ...` (bakes in the obs normaliser) before using it on the robot. None exists yet for model_22000.
3. Decision on moving to Windows is open (WSL2 suggested as the safest route). Back up first, then smoke test.

## How to run things
Warm start (copy the source checkpoint into a folder under the new experiment first):
```
mkdir -p logs/rsl_rl/microduck_hopN/prev_src
cp -r <run folder>/{model_X.pt,params} logs/rsl_rl/microduck_hopN/prev_src/
uv run train Mjlab-HopN-Flat-MicroDuck --env.scene.num-envs 2048 --agent.max_iterations 2000 \
    --agent.run-name hopN-warm --agent.resume True \
    --agent.load-run prev_src --agent.load-checkpoint model_X.pt
```
- About 1.5 s per iteration, about 3.6 GB VRAM, 2000 its is about 50 min. Always smoke test a new task first: 64 envs, `--agent.max_iterations 5`.
- Evaluate headless: `uv run python scripts/hop_eval.py <task> <path to model.pt>` (trunk rise, foot clearance, airborne steps, drift, post-landing speed and tilt).
- View: `uv run play <task> --num-envs 4 --viewer native --checkpoint-file <path>`. `--viewer viser` crashes. Wayland and OpenGL warnings are harmless.
- Tests (CPU): `uv run --with pytest pytest tests/` (85 pass at last run).

## Gotchas
- The last saved checkpoint is named one below the final iteration (for example model_9997 for iteration 9998). Intermediate ones save every 250.
- On resume the iteration counter carries on from the checkpoint, so env-step curricula start at their LATE stage (action-rate penalty, stand tax, etc. already at final values).
- `Episode_Reward/<term>` in logs is the WEIGHTED value. Self-negating penalties take a POSITIVE weight and must log <= 0.
- Never run `pkill -f` or `pgrep -f` with a pattern that also appears in your own command line (it kills or matches the calling shell). Use `ps -eo pid,args | grep "[v]env/bin/play"`.
- `hop3` to `hop6` score flight as foot clearance (so tucking feet counted). From hop 7 on, `reset_hop7_state` sets `env._hop4_trunk_coupled = True`, so flight = min(foot clearance, trunk rise above standing).
- Earlier docstrings quote "27 mm" or "40 mm" clearance for hop 3/4: that is foot clearance on their own measure, not comparable with trunk rise.

## Backups (as of 2026-10-18)
- Code: GitHub `smudger29uk/microduck_rl` (PUBLIC), branch `jump-hop-2026-10-11` (hops 2 to 16, tests, scripts; see git log for the last pushed commit). Older snapshot on `jump-2026-09-11`. Untracked on purpose: `checkpoints/`, `MUJOCO_LOG.TXT`, `logs/`, `wandb/`.
- Checkpoints: Hugging Face private model repo `smudger29uk/microduck-hop-local-runs`, one folder per run with the final model, `.onnx` (where trustworthy) and `params`: hop1-local to hop6-local, hop7-warm (9997), hop7-scratch (5999), hop8-warm2 (15995), hop9-long (22000, params only, no onnx), hop10-long-26750, hop12-warm-37747 and hop16-warm-42746 (model, params, onnx). NOT uploaded: hop11 night run, hop13 (abandoned), intermediate checkpoints.
- Original HF Job runs (Sept 2026) are in private repos `smudger29uk/mjlab-hop-flat-microduck-<timestamp>`, source tarball in dataset `smudger29uk/mjlab-microduck-src`.
- NOT backed up: full `logs/` and `wandb/` folders (W&B holds the curves online), intermediate checkpoints.
- Logged in on this PC: `gh` as smudger29uk, Hugging Face CLI as smudger29uk (use `uv run --no-project --with huggingface_hub python -I script.py`, there is no `hf` binary), W&B.
