import sys, torch
from dataclasses import asdict
import mjlab_microduck  # noqa
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
obs = env.get_observations()
xy, air, yaw = [], [], []
with torch.no_grad():
    for _ in range(150):
        obs, _, _, _ = env.step(pol(obs))
        l = mdp._sensor_any_contact(u, "left_foot_ground_contact"); r = mdp._sensor_any_contact(u, "right_foot_ground_contact")
        air.append((~l) & (~r)); xy.append(robot.data.root_link_pos_w[:, :2].clone())
        q = robot.data.root_link_quat_w
        yaw.append(torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2]**2+q[:,3]**2)))
xy, a, yaw = torch.stack(xy), torch.stack(air), torch.stack(yaw)
T = xy.shape[0]
first = torch.where(a.any(0), a.float().argmax(0), torch.full((a.shape[1],), T-1, device=dev))
idx = torch.arange(T, device=dev).unsqueeze(1)
step = torch.cat([torch.zeros_like(xy[:1]), xy[1:] - xy[:-1]]).norm(dim=2)  # path length per step
mm = 1e3
pre = (idx < first.unsqueeze(0))
post = ~pre
print(f"first flight step: mean {first.float().mean():.0f} (t={first.float().mean()*u.step_dt:.2f}s)")
print(f"path length before first flight (mm): mean {(step*pre).sum(0).mean()*mm:.0f}")
print(f"path length after  first flight (mm): mean {(step*post).sum(0).mean()*mm:.0f}")
net_pre = torch.stack([(xy[first[i], i]-xy[0, i]).norm() for i in range(xy.shape[1])])
net_tot = (xy[-1]-xy[0]).norm(dim=1)
print(f"net displacement to first flight (mm): mean {net_pre.mean()*mm:.0f};  total net {net_tot.mean()*mm:.0f}")
for lo, hi in [(0,25),(25,50),(50,75),(75,100),(100,125),(125,150)]:
    print(f"  steps {lo:3d}-{hi:3d}: path {step[lo:hi].sum(0).mean()*mm:5.0f} mm  frac airborne {a[lo:hi].float().mean():.2f}")
d = xy[-1]-xy[0]
dyaw = (yaw[-1]-yaw[0] + 3.14159) % 6.28318 - 3.14159
hd = torch.atan2(d[:,1], d[:,0])
rel = ((hd - yaw[0] + 3.14159) % 6.28318) - 3.14159
print(f"yaw change (deg): mean {torch.rad2deg(dyaw).mean():.1f} abs mean {torch.rad2deg(dyaw).abs().mean():.1f}")
print(f"drift direction rel. to initial heading (deg): mean {torch.rad2deg(rel).mean():.0f}, abs mean {torch.rad2deg(rel).abs().mean():.0f}  (0=forward, 180=backward)")
print(f"body-frame drift fwd {(d[:,0]*torch.cos(yaw[0])+d[:,1]*torch.sin(yaw[0])).mean()*mm:.0f} mm, lat abs {(-d[:,0]*torch.sin(yaw[0])+d[:,1]*torch.cos(yaw[0])).abs().mean()*mm:.0f} mm")
print("mean per-step path, last 6 steps (mm):", [round(float(step[i].mean()*mm),1) for i in range(T-6, T)])
print(f"drift excluding final step (xy[-2]-xy[0]) (mm): mean {(xy[-2]-xy[0]).norm(dim=1).mean()*mm:.0f} p90 {(xy[-2]-xy[0]).norm(dim=1).quantile(.9)*mm:.0f}")
print(f"peak per-env speed before final step (m/s) max over steps {((xy[1:-1]-xy[:-2]).norm(dim=2)/u.step_dt).max():.2f}")
