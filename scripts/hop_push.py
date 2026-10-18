"""Push-phase breakdown: python scripts/hop_push.py <task> <checkpoint>. Trunk / feet / whole-robot CoM along the body axis
through crouch bottom, liftoff and early flight. Forward axis is derived from the foot positions and its sign is
calibrated so net motion reads negative (matches hop_eval: negative = backwards)."""
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
ids = mdp._hop3_foot_geom_ids(u, robot)
print("body_com_pos_w", robot.data.body_com_pos_w.shape, "foot geom ids", ids)
# whole-robot CoM: mass-weighted body coms
try:
    mass = torch.as_tensor(u.sim.model.body_mass, device=dev)
    print("mass shape", mass.shape)
except Exception as e:
    mass = None; print("no mass:", e)
obs = env.get_observations()
rec = {k: [] for k in ["trunk","feet","com","air","vel","tilt","pitch"]}
with torch.no_grad():
    for _ in range(149):
        obs, _, _, _ = env.step(pol(obs))
        l = mdp._sensor_any_contact(u, "left_foot_ground_contact"); r = mdp._sensor_any_contact(u, "right_foot_ground_contact")
        d = robot.data
        rec["air"].append(((~l) & (~r)).clone())
        rec["trunk"].append(d.root_link_pos_w.clone())
        rec["feet"].append(d.geom_pos_w[:, ids, :].clone())
        rec["vel"].append(d.root_link_lin_vel_w.clone())
        rec["tilt"].append(mdp._tilt_deg(robot))
        q = d.root_link_quat_w
        # pitch about body y from gravity projection x component
        rec["pitch"].append(torch.rad2deg(torch.asin(d.projected_gravity_b[:, 0].clamp(-1, 1))))
        bc = d.body_com_pos_w
        if mass is not None and mass.shape[-1] >= bc.shape[1]:
            m = mass.reshape(-1)[-bc.shape[1]:] if mass.ndim == 1 else mass[0][-bc.shape[1]:]
            rec["com"].append((bc * m.view(1, -1, 1)).sum(1) / m.sum())
        else:
            rec["com"].append(bc[:, 0, :].clone())
S = {k: torch.stack(v) for k, v in rec.items()}
T, N = S["air"].shape
# heading at step 0 from trunk quat is unavailable; use the horizontal motion frame: project onto trunk-forward via gravity-free yaw from foot geoms (left->right axis)
f = S["feet"]  # [T,N,2,3]
lat = f[0, :, 1, :2] - f[0, :, 0, :2]; lat = lat / lat.norm(dim=1, keepdim=True)
fwd = torch.stack([-lat[:, 1], lat[:, 0]], 1)  # rotate +90 deg; sign fixed below by trunk motion convention
# fix sign: forward is the side the head faces; use initial pitch-free assumption -> pick sign so toe (foot geom far point) ... fall back: calibrate with eval metric (end displacement should be negative)
def proj(p): return (p[..., :2] * fwd.unsqueeze(0)).sum(-1)
first = torch.where(S["air"].any(0), S["air"].float().argmax(0), torch.full((N,), T-1, device=dev))
ar = torch.arange(N, device=dev)
tr = proj(S["trunk"]) - proj(S["trunk"])[0]
sign = torch.sign(tr[-1].mean()) * -1  # eval says net motion is backwards (negative)
print("axis sign flip" if sign < 0 else "axis ok")
fwd = fwd * (-sign if sign != 0 else 1.0) if False else fwd
sg = -1.0 if tr[-1].mean() > 0 else 1.0
def P(x): return sg * proj(x)
trunk = P(S["trunk"]); com = P(S["com"]); foot = P(f.mean(2)); vel = sg * (S["vel"][..., :2] * fwd.unsqueeze(0)).sum(-1)
# crouch bottom: min trunk z before first flight
z = S["trunk"][..., 2]
zmask = torch.arange(T, device=dev).unsqueeze(1) < first.unsqueeze(0)
zb = torch.where(zmask, z, torch.full_like(z, 9)).argmin(0)
def at(x, idx): return x[idx.clamp(0, T-1), ar]
mm = 1e3
print(f"first liftoff step mean {first.float().mean():.1f}; crouch-bottom step mean {zb.float().mean():.1f}")
for name, idx in [("start", torch.zeros_like(first)), ("crouch bottom", zb), ("liftoff", first), ("peak (liftoff+2)", first+2)]:
    print(f"{name:16s} trunk-x {at(trunk-trunk[0], idx).mean()*mm:7.1f} mm | feet-x {at(foot-foot[0], idx).mean()*mm:7.1f} mm (foot slide) | "
          f"trunk rel. feet {at(trunk-foot, idx).mean()*mm:7.1f} mm | com rel. feet {at(com-foot, idx).mean()*mm:7.1f} mm | "
          f"vx {at(vel, idx).mean():6.3f} m/s | pitch {at(S['pitch'], idx).mean():5.1f} deg")
# push phase: bottom -> liftoff
push = (torch.arange(T, device=dev).unsqueeze(1) >= zb.unsqueeze(0)) & (torch.arange(T, device=dev).unsqueeze(1) <= first.unsqueeze(0))
print(f"push phase length (steps) {push.float().sum(0).mean():.1f}")
print(f"mean trunk vx during push {(vel*push).sum(0).div(push.sum(0).clamp(min=1)).mean():.3f} m/s;  peak |vx| in push {(vel*push).abs().max(0).values.mean():.3f}")
print(f"trunk-x change over push {( at(trunk, first) - at(trunk, zb)).mean()*mm:.1f} mm; feet-x change over push {(at(foot, first)-at(foot, zb)).mean()*mm:.1f} mm")
print(f"before liftoff, trunk-x relative to feet: crouch bottom {at(trunk-foot, zb).mean()*mm:.1f} -> liftoff {at(trunk-foot, first).mean()*mm:.1f} mm")
