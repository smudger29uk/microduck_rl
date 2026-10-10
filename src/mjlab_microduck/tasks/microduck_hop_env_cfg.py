"""Microduck hop task — attempt 1.

Episodic policy: robot starts standing, crouches, extends explosively enough
to leave the ground with BOTH feet simultaneously, briefly airborne, then
lands and recovers to standing. Triggered like sit/standup/roulade (policy
switch = hop starts immediately; no phase clock, no reference trajectory).

This is a fresh task — there is no prior training run to report lessons
from, unlike roulade's "run 3, attempt 5" history. Design choices below are
reasoned from first principles and from what roulade/standup already proved,
not from hop-specific data. Expect the usual reward-hacking iteration once a
real run exists (see mdp.py's hop section docstring for the full reasoning).

Key differences from roulade (the closest existing template — also a
dynamic, briefly-ballistic maneuver):
  • Roulade structurally FORBIDS leaving the ground (a support-gated rotation
    accumulator — "a roulade never leaves the floor"). A hop's entire point
    IS leaving the ground. The progress signal here borrows roulade's
    potential-based accumulator shape (pay only a NEW peak, camping or
    repeated smaller hops pay nothing further) but keys off "both feet
    simultaneously unloaded" instead of a support-gated rotation angle.
  • Roulade deletes the always-on "upright" reward (the flip needs to leave
    vertical on purpose). A hop should stay upright throughout — it's a
    vertical leap, not a somersault — so this env KEEPS a modest always-on
    upright term instead of deleting it.
  • Roulade turns on impact/style shaping from step 0 because falling into a
    roll is an EASY thing to stumble into by accident (run 1 proved it).
    Coordinating both legs to extend explosively enough for real liftoff
    against these actuators (kp=0.55, forcerange +/-0.96 N*m, real XL330
    sysID) is plausibly a HARD skill to find by random exploration — so
    impact/motion-blocking regularisers here follow STANDUP's timing lesson
    (near-zero during early exploration, ramped in late by curriculum) more
    than roulade's "on from step 0" choice.
  • No reverse-curriculum mid-air spawn bucket (unlike roulade's mid-roll
    spawns, added because the second half of a roll is a face-up-recovery
    problem that gets no on-policy data otherwise). Every episode starts
    standing; if landing-phase learning stalls the way standup's did before
    reverse-curriculum spawns fixed it, that's the next thing to add here.
"""

import math
from copy import deepcopy

# Symmetry — a vertical hop is sagittal / left-right symmetric.
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to roulade/standup for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a push mid-flight is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to roulade/standup) ───────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: crouch (~0.5s) + explosive extension/flight (~0.3-0.5s for a robot
# this size) + land + recover + settle. Generous relative to the maneuver
# itself, same reasoning as roulade's EPISODE_LENGTH_S comment.
EPISODE_LENGTH_S = 3.0

# Empirically-measured standing trunk height — SAME constant roulade/standup
# use (same robot cfg family, MICRODUCK_STANDUP_ROBOT_CFG below). Don't
# re-measure/re-guess this per-task (the standup lesson this constant itself
# documents).
STAND_Z = 0.115

# Hop-completion gate: episode's peak airborne height must exceed this before
# the landing/recovery reward (and the stand-after-hop tax) turn on at all —
# otherwise standing still forever would be a free ride to the landing
# reward's payout.
#
# RUN 2 (2000 iters @ 4096 envs, 2026-09-10, wandb run z3z1xbzk): a naive
# scripted "maximum instantaneous torque" test to find a physical ceiling
# backfired informatively -- commanding an instant snap from deep crouch to
# full extension didn't launch the robot, it just knocked it over (a real
# jump needs coordinated timing, not raw torque, which is exactly the part
# RL has to discover). So there's no computed ceiling to set this against.
# Verified instead against what training actually achieved: this checkpoint
# reached ~13mm peak height / 48ms airborne. Raising the bar from 15mm to
# 20mm is a modest, proven-reachable-adjacent step (not a guess at a
# theoretical max) -- the standard incremental-curriculum move when you don't
# have a ground-truth ceiling: nudge just past the last proven point.
MIN_HOP_HEIGHT = 0.020
HOP_GATE_WIDTH = 0.01

# Tuck target for hop_tuck_reward below (left-leg convention, mirrored for
# the right leg by the same rule every other pose target in this file uses).
# Same values used in the very first kinematic-only hop preview, now given
# an actual reward reason to exist.
HOP_TUCK_OVERRIDES = {
    2:  0.3,   # left  hip_pitch
    3:  -1.3,  # left  knee
    4:  0.6,   # left  ankle
    11: -0.3,  # right hip_pitch
    12: 1.3,   # right knee
    13: -0.6,  # right ankle
}

# Crouch target for hop_crouch_reward below. RUN 3 checkpoint (2000 iters,
# wandb h846u6n1) measured a much shallower crouch than this before
# launching: hip_pitch only reached -0.61 (vs standing -0.46) and knee only
# -0.34 (vs standing -0.005) -- roughly a third of the depth below, in the
# joint-space sense of the deepest crouch this model can hold with a flat
# sole (found earlier by gradient descent on trunk height for the physical-
# ceiling experiment: hip_pitch -0.65, knee +0.79, ankle +1.44). Diagnosis:
# the always-on pose_stand/height_stand/upright_always terms cost every
# single step spent away from HOME, so the policy minimizes crouch depth and
# duration to whatever's just enough to launch -- there was never a reward
# for crouching MORE, only pressure to crouch LESS. This target gives it one,
# gated on actually being grounded and meaningfully below standing height so
# it can't fight normal standing.
HOP_CROUCH_OVERRIDES = {
    2:  -0.6507,  # left  hip_pitch
    3:  0.7880,   # left  knee
    4:  1.4375,   # left  ankle
    11: 0.6507,   # right hip_pitch
    12: -0.7880,  # right knee
    13: -1.4375,  # right ankle
}
# How far below standing height counts as "actually crouching" for the gate
# above -- loose enough to cover both the pre-launch crouch and the landing-
# absorb dip, tight enough that ordinary standing (which oscillates a few mm)
# never triggers it.
HOP_CROUCH_HEIGHT_MARGIN = 0.02

# Cap on paid airborne height for hop_airborne_progress — same role as
# roulade_progress's max_paid_rate: rewards saturate here so a policy that
# somehow found much more height doesn't get an ever-growing single-step
# jackpot on the step it happens (it still gets full credit up to the cap,
# just not unbounded credit beyond it).
#
# RUN 1 (200 iters @ 4096 envs, 2026-09-09, wandb run wno6h9g0): the policy
# converged toward standing PERFECTLY still — pose_stand/height_stand/
# upright_always all approached their max, while hop_airborne_progress and
# hop_launch_velocity both PEAKED around iter 10 and declined toward ~0 by
# iter 150. Diagnosis: the always-on standing bootstrap terms were reachable
# and nearly-maxed within ~100 iterations, worth ~2.5/step for the full
# 150-step episode just by holding HOME — a large, reliable payoff with zero
# risk. hop_airborne_progress only pays once per episode (potential-based,
# new-peak-only) and only if real liftoff happens, which a policy converging
# toward stillness stumbles into less and less by chance. The gradient
# reliably pointed toward "stand better," never toward "risk leaving the
# ground" — a real instance of the local optimum flagged when this reward set
# was designed, not a hypothetical.
#
# Rebalance for run 2: shrink MAX_PAID_HOP_HEIGHT so the SAME physical liftoff
# is worth a bigger normalized share (delta_height / cap) — 35mm instead of
# 60mm roughly doubles the payout per mm of real height achieved. See the
# matching reward-weight changes below (standing bootstrap weights cut
# ~60-70%, hop_airborne_progress and hop_launch_velocity weights raised).
MAX_PAID_HOP_HEIGHT = 0.035

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg


def make_microduck_hop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck hop environment configuration."""

    # Combined sensor — NAME IS LOAD-BEARING beyond this file: every microduck
    # task's inherited base obs terms (foot_contact_forces, foot_air_time)
    # and robot_state_is_nan's force-NaN guard look up a sensor literally
    # named "feet_ground_contact". Keep it even though the hop-specific
    # reward logic below needs the per-foot sensors alongside it.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Per-foot sensors — the combined sensor above reports "found" if EITHER
    # foot touches, which cannot distinguish a single-foot toe-off from a
    # genuine two-footed liftoff. hop_airborne_progress/hop_completion_gate
    # need both of these to be simultaneously un-contacted.
    left_foot_ground_cfg = ContactSensorCfg(
        name="left_foot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^left_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    right_foot_ground_cfg = ContactSensorCfg(
        name="right_foot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^right_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (
        feet_ground_cfg, left_foot_ground_cfg, right_foot_ground_cfg, self_collision_cfg,
    )
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # track_angular_velocity is KEPT (unlike the walking-specific terms above)
    # and re-tuned rather than dropped — run 3 (2000 iters, wandb h846u6n1)
    # measured real, uncorrected yaw drift: up to -8.9deg during the hop
    # itself, settling to a persistent -4.1deg offset (matches
    # Metrics/twist/error_vel_yaw sitting around 0.5-0.7 rad/s all through
    # training). The twist command stays near-zero here (kept only for 61D
    # obs-shape parity, see the command block below), so this reward is
    # pulling toward "don't rotate," not toward any real turn. Kept light —
    # AGENTS.md's regularizer lesson applies: yaw rate is part of what an
    # explosive liftoff/landing physically needs, so this should nudge, not
    # block, unlike a walking gait's much tighter heading requirement.
    if "track_angular_velocity" in cfg.rewards:
        cfg.rewards["track_angular_velocity"].weight = 0.5
        cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # ── Rewards: hop task set ──────────────────────────────────────────────────
    # RUN 1 REBALANCE (see the MAX_PAID_HOP_HEIGHT comment above for the full
    # diagnosis): weights on hop_airborne_progress and hop_launch_velocity
    # raised — same physical liftoff/push now worth more relative to the cut-
    # down standing bootstrap below, so exploring away from "stand still"
    # stops being strictly worse in expectation.
    #
    # The one dense "attempt/discover the jump" signal — pays only a NEW peak
    # airborne height, so camping or repeating a smaller hop earns nothing
    # further (roulade_progress's shape, applied to height instead of angle).
    cfg.rewards["hop_airborne_progress"] = RewardTermCfg(
        func=microduck_mdp.hop_airborne_progress,
        weight=10.0,
        params={
            "standing_height":   STAND_Z,
            "left_sensor_name":  left_foot_ground_cfg.name,
            "right_sensor_name": right_foot_ground_cfg.name,
            "max_paid_height":   MAX_PAID_HOP_HEIGHT,
        },
    )

    # Reward tucking the legs up WHILE airborne (per-step, gated on the
    # instantaneous both-feet-off contact state — see mdp.py's docstring for
    # the physical reasoning: a shorter effective leg length gives the feet
    # farther to fall before touching down again for the SAME ballistic
    # trajectory, so this should genuinely extend measured airborne duration,
    # not just look more like a deliberate jump).
    cfg.rewards["hop_tuck"] = RewardTermCfg(
        func=microduck_mdp.hop_tuck_reward,
        weight=1.5,
        params={
            "left_sensor_name":  left_foot_ground_cfg.name,
            "right_sensor_name": right_foot_ground_cfg.name,
            "joint_indices":     _LEG_JOINTS,
            "tuck_overrides":    HOP_TUCK_OVERRIDES,
        },
    )

    # Reward a deep crouch WHILE grounded and meaningfully below standing
    # height — the counterweight to hop_tuck above, and to the always-on
    # standing bootstrap terms further down which otherwise only ever
    # penalize time spent away from HOME (see mdp.py's docstring: run 3
    # measured a crouch only ~1/3 as deep as what's kinematically available,
    # exactly the shallow "flick" that pressure produces).
    cfg.rewards["hop_crouch"] = RewardTermCfg(
        func=microduck_mdp.hop_crouch_reward,
        weight=1.5,
        params={
            "left_sensor_name":     left_foot_ground_cfg.name,
            "right_sensor_name":    right_foot_ground_cfg.name,
            "joint_indices":        _LEG_JOINTS,
            "crouch_overrides":     HOP_CROUCH_OVERRIDES,
            "standing_height":      STAND_Z,
            "height_margin":        HOP_CROUCH_HEIGHT_MARGIN,
        },
    )

    # Explosive-extension bootstrap: reward upward CoM velocity while still
    # near/below standing height — this is the push BEFORE liftoff, gives a
    # bootstrap gradient toward "extend fast" even before any real airborne
    # height has ever been achieved (hop_airborne_progress alone has zero
    # gradient until the first successful liftoff). max_vz caps it so a more
    # violent push isn't automatically worth more (its own docstring's
    # explosive-vs-gentle-rise note). Run 1 showed this term working (it rose
    # in early iterations) then getting bid down as standing-still won out —
    # doubling the weight keeps it competitive against the cut-down bootstrap.
    cfg.rewards["hop_launch_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=2.0,
        params={"max_height": STAND_Z + 0.005, "max_vz": 0.6},
    )

    # Completion-gated standing annuity — zero until a real hop has happened
    # this episode, then every step spent back at HOME pose pays. Broad stds
    # (standup composite lesson: partial landing must score visibly).
    cfg.rewards["hop_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.hop_landing_composite,
        weight=4.0,
        params={
            "target_height":    STAND_Z,
            "height_std":       0.04,
            "upright_std":      0.40,
            "pose_std":         0.40,
            "joint_indices":    _LEG_JOINTS,
            "min_hop_height":   MIN_HOP_HEIGHT,
            "gate_width":       HOP_GATE_WIDTH,
            "target_overrides": None,
        },
    )

    # Completion-gated stand tax (the roulade/standup lesson): once a real hop
    # has happened, every step spent below STAND_Z costs — "land and crumple"
    # flips from free to net-negative. Gate closed before the hop itself, so
    # the crouch and the landing-absorb dip are never taxed for being low.
    cfg.rewards["hop_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.hop_stand_tax,
        weight=0.0,  # ramped by curriculum — see hop_stand_tax_weight below
        params={
            "target_height":  STAND_Z,
            "min_hop_height": MIN_HOP_HEIGHT,
            "gate_width":     HOP_GATE_WIDTH,
        },
    )

    # Always-on bootstrap layers — broad Gaussian + L1 on pose/height against
    # HOME. RUN 1 (200 iters, wandb wno6h9g0): at the original weights these
    # were reachable and nearly maxed within ~100 iterations — pose_stand
    # 0.06->0.98, height_stand 0.04->0.50, upright_always 0.08->0.99 — worth
    # roughly 2.5/step for the full 150-step episode just from holding HOME,
    # for zero risk. hop_airborne_progress/hop_launch_velocity meanwhile
    # PEAKED around iter 10 and declined toward 0 by iter 150: the policy
    # converged onto "stand perfectly still" and never attempted liftoff
    # again, exactly the local optimum this design flagged as a risk. Cut
    # ~60-70% here so standing well is still learned (these can't go to zero,
    # or there is no pose/balance shaping at all) but no longer dominates the
    # per-episode return the way a real hop (hop_landing_composite, once
    # earned, pays per-step for the rest of the episode) should.
    cfg.rewards["pose_stand"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=0.4,
        params={"joint_indices": _LEG_JOINTS, "target_overrides": None},
    )
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=0.3,
        params={"joint_indices": _LEG_JOINTS, "target_overrides": None},
    )
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=0.2,
        params={"target_height": STAND_Z, "std": 0.04},
    )
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=0.6,
        params={"target_height": STAND_Z},
    )

    # Always-on upright — UNLIKE roulade (which deletes this because the flip
    # needs to leave vertical), a hop should never flip: it's a vertical
    # leap, not a somersault. Cut alongside the other bootstrap terms above
    # for the same run-1 reason (it was one of the three terms the policy
    # maxed out while abandoning the hop attempt).
    cfg.rewards["upright_always"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=0.4,
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    # Motion-blockers stay near zero during discovery (liftoff needs a large,
    # fast, coordinated bilateral extension — taxing attempts before the
    # skill exists makes "do nothing" win, the standup timing lesson).
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # near zero: liftoff/landing need free ang vel
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # Arrival damper — trunk ω_xy² gated on near-standing height AND low
    # tilt, so the hop itself is never taxed; introduced at 0, ramped later.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":    STAND_Z - 0.02,
            "height_high":   STAND_Z,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # |a_z| impact shaping on landing — introduced late (see file docstring:
    # unlike roulade, discovery here is plausibly hard, so style pressure
    # waits for the skill to exist first). Self-negating -> POSITIVE weight.
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Self-collision — light: a tucked landing can involve some body-on-body
    # contact, don't fight it hard.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]  # replaced by upright_always above

    # ── Observations (identical layout to walking / standup / roulade) ────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots: zero padding for both head (4) and body (6) — hop
    # doesn't track either; 61D layout parity keeps the runtime stack unchanged.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # Falling on a bad landing is part of what the policy must learn to
    # avoid via reward shaping, not a hard termination (roulade's choice, for
    # the same reason: cutting the episode short on a fall removes the
    # recovery-learning signal a short episode like this can't afford to lose).
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Standing start + small joint noise; also zeros the hop peak-height
    # accumulator. Must run after reset_robot_joints (dict insertion order —
    # this event's joint noise is added on top of the HOME pose that wrote).
    cfg.events["set_hop_state"] = EventTermCfg(
        func=microduck_mdp.reset_hop_state,
        mode="reset",
        params={
            "standing_z_min":    0.11,
            "standing_z_max":    0.12,
            "standing_tilt_max": math.radians(5.0),
            "joint_noise_std":   0.02,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # Smoothness/impact/tax polish — introduced only after the jump skill
    # exists (standup timing lesson: any attempt-tax active during discovery
    # prevents the maneuver from being found at all). Stage boundaries are a
    # first guess (no prior run to time them against, unlike roulade/standup's
    # dated stage numbers) — watch wandb for a reward step-down right at a
    # boundary, the standup/roulade sign that pacing is off.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.05},
                {"step": 1500 * 24,  "weight": -0.15},
                {"step": 3000 * 24,  "weight": -0.3},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": -5e-4},
                {"step": 3000 * 24,  "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": -0.025},
                {"step": 3000 * 24,  "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the func is self-negating (returns -|a_z|).
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": 0.002},
                {"step": 3000 * 24,  "weight": 0.005},
            ],
        },
    )
    cfg.curriculum["hop_stand_tax_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weight: hop_stand_tax is self-negating.
            "reward_name":   "hop_stand_tax",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1500 * 24,  "weight": 1.5},
                {"step": 2500 * 24,  "weight": 3.0},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckHopRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_hop",
    run_name="microduck_hop",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)
