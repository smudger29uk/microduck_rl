# Handoff: Microduck hop (jump) training, local on this PC

Written 2026-10-11 from a claude.ai session. Read this first. The user wants to do all training LOCALLY on this PC. Never use `--hf-jobs`.

Style: UK English, no em dashes, concise, numbered steps, one thing at a time.

## Machine
- Ubuntu PC, Ryzen 7 5800X, 128 GB RAM, RTX 3070 (8 GB VRAM), `uv` installed, Python 3.12 venvs via `uv sync`.
- Training stack: mjlab 1.3.0 (MuJoCo Warp) + rsl_rl PPO, BAM actuator model.

## Folders in ~/robotics
- `microduck_rl_jump/`  EXACT code snapshot the HF job ran (from tarball `src-20260911-072331.tar.gz`, private HF dataset `smudger29uk/mjlab-microduck-src`). This is the working folder. It contains the custom hop task. Check `git status` and `git log`: a snapshot commit was attempted, the user may or may not have completed it.
- `microduck_rl/`       fork `smudger29uk/microduck_rl`, currently on branch `my-jump` tracking `upstream/spring_v2` (Pollen's spring hop branch). NOT the user's jump. Ignore for this work.
- `microduck/`          fork of the Rust runtime for the physical robot (not needed for training).
- `jump-src/`           the downloaded tarball.

## The task
- Task id: `Mjlab-Hop-Flat-MicroDuck`
- Env config: `src/mjlab_microduck/tasks/microduck_hop_env_cfg.py`
- Reward functions: `src/mjlab_microduck/tasks/mdp.py`, hop section starts around line 7191 (`_update_hop_peak`, `hop_airborne_progress`, `hop_completion_gate`, `hop_landing_composite`, `hop_stand_tax`, `hop_tuck_reward`, `hop_crouch_reward`). `com_upward_velocity` is around line 900.
- Obs: actor 61, critic 74. Actions: 14 joints. 50 Hz, 3 s episodes (150 steps).
- Joint layout: 0-4 left leg, 5-8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll), 9-13 right leg.

## History
- Original HF Job 2026-09-11: 2000 iterations, 4096 envs, L4. W&B run `garry-smith29-hobbyist/mjlab_microduck/lno7bpp6`.
- Checkpoints: HF model repo `smudger29uk/mjlab-hop-flat-microduck-20260911-072331` (model_0 ... model_1999.pt). Downloaded to `microduck_rl_jump/checkpoints/` and copied to `logs/rsl_rl/microduck_hop/2026-09-11_06-45-00_microduck_hop/`.
- Local resume currently running in tmux session `hop` (check `tmux ls`), started about 23:48 on 2026-10-10:

```
uv run train Mjlab-Hop-Flat-MicroDuck --env.scene.num-envs 2048 --agent.max_iterations 2000 \
    --agent.run-name hop-local --agent.resume True \
    --agent.load-run 2026-09-11_06-45-00_microduck_hop --agent.load-checkpoint model_1999.pt
```
  It resumed at iteration 2000 and runs to 3999. About 1.45 s per iteration, about 3.5 GB VRAM. New checkpoints save every 250 iterations into a new folder under `logs/rsl_rl/microduck_hop/` with `hop-local` in the name. DO NOT kill it unless the user agrees.

## Viewing a policy
```
uv run play Mjlab-Hop-Flat-MicroDuck --num-envs 4 --viewer native --checkpoint-file <path to model_XXXX.pt>
```
`--viewer viser` crashes with `AssertionError: assert max >= value >= min` in mjlab's `velocity_command.py` create_gui slider (twist ranges are +/-0.01 in this task). Use `--viewer native`.

## Observed behaviour of model_1999
Small bounces or heel lifts, feet barely leave the ground. The head turns left for no reason.

## Diagnosis (from reading the code and logs)
1. Height is measured against a fixed datum. `_update_hop_peak` takes trunk z minus `STAND_Z` (0.115) while both foot sensors read no contact. That is trunk rise, not flight height, so straightening the legs plus a momentary two-foot clearance can score as a hop.
2. Everything saturates near 3 cm: `MAX_PAID_HOP_HEIGHT` = 0.035, gate opens at `MIN_HOP_HEIGHT` 0.020 and is full at 0.030. Final-iteration logs imply the average peak is about 30 mm of the 35 mm cap (assuming mjlab logs episode sums divided by episode seconds). No reward for going higher.
3. `hop_landing_composite` (weight 4.0) logged 3.56 out of a possible 4.0: the gate opens in the first fraction of each episode, then the duck stands still collecting it.
4. `hop_crouch` logs 0.0000: it only pays when trunk z < 0.095 and grounded (hard gate, no partial credit), while `height_stand_l1` (weight 0.6) penalises any deviation from 0.115 every step. No gradient into a crouch.
5. Head drift: all pose rewards use `_LEG_JOINTS = [0,1,2,3,4,9,10,11,12,13]`. Joints 5-8 have no reward, and head/body command observations are zero padding.
6. The running resume is in the curriculum's polish phase (arrival damping on from iteration 2000, stand tax to 3.0 at 2500, action rate to -0.3 at 3000). It will smooth landings but cannot raise jump height because of 1 and 2.
7. For reference, Pollen's `upstream/spring_v2` hop task independently moved to "rise above takeoff height" and "airborne means ZERO ground contact".

## Proposed next step (user said they want to do this in Claude Code; confirm before editing)
Create a NEW task id, e.g. `Mjlab-Hop2-Flat-MicroDuck`, so the original stays reproducible. Do not edit the running training's task in place (the process already imported the code, but keep the original intact for comparison).
1. Track `takeoff_z` (trunk z while any foot is grounded); peak = max of `z - takeoff_z` while both feet are off. Reset it in `reset_hop_state`.
2. Raise ceilings, for example cap about 0.06 and gate about 0.015 to 0.025 (tune).
3. Replace the hard crouch gate with a dense term: reward proportional to how far trunk z is below `STAND_Z` while grounded (clamped), and consider lowering `height_stand_l1`.
4. Add a head/neck pose penalty on joints 5-8 (`pose_l1_penalty` with `joint_indices=[5,6,7,8]`, small positive weight, it is self-negating).
5. Train from scratch (action std collapsed to 0.17, which limits exploration). Obs and action dims are unchanged, so warm starting is technically possible.
6. Add or extend CPU tests in `tests/` (the repo has config and reward regression tests), run with `uv run --with pytest pytest tests/`.
7. Launch with 2048 envs (about 48 min per 2000 iterations) or 4096 (VRAM has room, 41% used at 2048).

## Outstanding housekeeping
- GitHub backup of the snapshot is not done: `git remote add origin https://github.com/smudger29uk/microduck_rl.git` then `git push origin main:jump-2026-09-11`. `gh auth login` (device flow) was in progress, state unknown.
- Git identity was set globally (Garry Smith).
- W&B: user may be logged in, or use `WANDB_MODE=offline`. `play` can take `--checkpoint-file` so W&B is not needed for viewing.
- Hugging Face CLI is logged in as `smudger29uk`.
