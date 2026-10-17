"""Headless hop eval: python scripts/hop_eval.py <task> <checkpoint>.
Reports trunk rise above STAND_Z (0.115) and foot clearance while both feet are off the ground."""
import sys, torch
from dataclasses import asdict
import mjlab_microduck  # noqa registers tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks import mdp

task, ckpt = sys.argv[1], sys.argv[2]
dev = "cuda:0"
cfg = load_env_cfg(task, play=False); cfg.scene.num_envs = 256
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
pol = runner.get_inference_policy(device=dev)
u = env.unwrapped; robot = u.scene["robot"]
ids = mdp._hop3_foot_geom_ids(u, robot)
obs = env.get_observations()
zs, air, fz, tilt, xy, vxy = [], [], [], [], [], []
with torch.no_grad():
    for _ in range(150):
        obs, _, _, _ = env.step(pol(obs))
        l = mdp._sensor_any_contact(u, "left_foot_ground_contact"); r = mdp._sensor_any_contact(u, "right_foot_ground_contact")
        zs.append(robot.data.root_link_pos_w[:, 2].clone()); air.append((~l) & (~r))
        xy.append(robot.data.root_link_pos_w[:, :2].clone()); vxy.append(robot.data.root_link_lin_vel_w[:, :2].norm(dim=1).clone())
        fz.append(robot.data.geom_pos_w[:, ids, 2].min(1).values.clone() - 0.006); tilt.append(mdp._tilt_deg(robot))
z, a, f, t = map(torch.stack, (zs, air, fz, tilt))
xy, vxy = torch.stack(xy), torch.stack(vxy)
rise = torch.where(a, (z - 0.115).clamp(min=0), torch.zeros_like(z)).max(0).values
clear = torch.where(a, f.clamp(min=0), torch.zeros_like(f)).max(0).values
mm = 1e3
print(f"airborne steps/ep {a.float().sum(0).mean():.1f}  eps with flight {(a.any(0)).float().mean():.2f}")
print(f"trunk rise over 115mm  (mm): mean {rise.mean()*mm:.1f} median {rise.median()*mm:.1f} p90 {rise.quantile(.9)*mm:.1f} max {rise.max()*mm:.1f}")
print(f"foot clearance         (mm): mean {clear.mean()*mm:.1f} median {clear.median()*mm:.1f} p90 {clear.quantile(.9)*mm:.1f} max {clear.max()*mm:.1f}")
print(f"crouch min trunk z (mm) mean {z.min(0).values.mean()*mm:.0f}   max tilt in flight (deg) mean {torch.where(a,t,torch.zeros_like(t)).max(0).values.mean():.1f}")

# landing: steps after the first flight ends
first = torch.where(a.any(0), a.float().argmax(0), torch.full((a.shape[1],), 149, device=a.device))
idx = torch.arange(150, device=a.device).unsqueeze(1)
after = idx > (first + a.float().cumsum(0).max(0).values.long()).unsqueeze(0)
drift = (xy[-2] - xy[0]).norm(dim=1)  # [-2]: the last step is the timeout reset (teleport to a new spawn)
v_after = torch.where(after, vxy, torch.zeros_like(vxy)).max(0).values
t_after = torch.where(after, t, torch.zeros_like(t)).max(0).values
print(f"net horizontal drift over episode (mm): mean {drift.mean()*mm:.0f} p90 {drift.quantile(.9)*mm:.0f} max {drift.max()*mm:.0f}")
print(f"peak horizontal speed after flight (m/s): mean {v_after.mean():.2f} p90 {v_after.quantile(.9):.2f}   max tilt after flight (deg): mean {t_after.mean():.1f} p90 {t_after.quantile(.9):.1f}")
