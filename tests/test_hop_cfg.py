import mujoco

from mjlab_microduck.robot.microduck_constants import get_standup_spec
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    _LEG_JOINTS,
    EPISODE_LENGTH_S,
    MAX_PAID_HOP_HEIGHT,
    MIN_HOP_HEIGHT,
    STAND_Z,
    make_microduck_hop_env_cfg,
)


def test_env_builds_train_and_play():
    assert make_microduck_hop_env_cfg() is not None
    assert make_microduck_hop_env_cfg(play=True) is not None


def test_episode_length():
    cfg = make_microduck_hop_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 3.0


def test_terrain_is_plain_plane():
    cfg = make_microduck_hop_env_cfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401  (import triggers registration)

    assert "Mjlab-Hop-Flat-MicroDuck" in list_tasks()


def test_joint_indices_match_actual_model():
    """Lock: MICRODUCK_STANDUP_ROBOT_CFG's joint order must match _LEG_JOINTS.

    Compiles the real MjSpec and checks names at the indices this cfg's
    reward functions use — pure CPU, no sim, same guard as the roller-standup
    test that caught wheel/joint interleaving bugs on a different model.
    """
    model = get_standup_spec().compile()
    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    assert [articulated[i] for i in _LEG_JOINTS] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]


def test_sensors_present():
    cfg = make_microduck_hop_env_cfg()
    names = [s.name for s in cfg.scene.sensors]
    # feet_ground_contact is load-bearing for inherited base obs terms
    # (foot_contact_forces/foot_air_time) and robot_state_is_nan's
    # sensor_names — must survive even though hop's own logic uses the
    # per-foot sensors instead.
    assert "feet_ground_contact" in names
    assert "left_foot_ground_contact" in names
    assert "right_foot_ground_contact" in names
    assert "self_collision" in names


def test_nan_guard_present_with_force_sensor():
    cfg = make_microduck_hop_env_cfg()
    assert "nan_state" in cfg.terminations
    assert cfg.terminations["nan_state"].params["sensor_names"] == ("feet_ground_contact",)


def test_no_fell_over_termination():
    # Falling on a bad landing is learning signal, not a hard cutoff —
    # same choice roulade makes and for the same reason (short episode,
    # can't afford to lose the recovery-learning signal).
    cfg = make_microduck_hop_env_cfg()
    assert "fell_over" not in cfg.terminations


def test_twist_command_is_neutralised():
    cfg = make_microduck_hop_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.rel_standing_envs == 0.0


def test_head_and_body_command_are_zero_padded():
    cfg = make_microduck_hop_env_cfg()
    from mjlab_microduck.tasks import mdp as microduck_mdp

    for grp in ("actor", "critic"):
        assert cfg.observations[grp].terms["head_command"].func is microduck_mdp.zero_command_padding
        assert cfg.observations[grp].terms["head_command"].params["dim"] == 4
        assert cfg.observations[grp].terms["body_command"].func is microduck_mdp.zero_command_padding
        assert cfg.observations[grp].terms["body_command"].params["dim"] == 6


def test_hop_task_rewards_present_with_expected_weights():
    cfg = make_microduck_hop_env_cfg()
    # Rebalanced after run 1 (200 iters, wandb wno6h9g0): the original weights
    # let the policy converge to standing perfectly still, never attempting
    # liftoff. See the reward-weight comments in microduck_hop_env_cfg.py.
    expected = {
        "hop_airborne_progress":   10.0,
        "hop_launch_velocity":     2.0,
        "hop_tuck":                1.5,
        "hop_crouch":              1.5,
        "hop_landing_composite":   4.0,
        "hop_stand_tax":           0.0,   # ramped in by curriculum, see below
        "pose_stand":              0.4,
        "pose_stand_l1":           0.3,
        "height_stand":            0.2,
        "height_stand_l1":         0.6,
        "upright_always":          0.4,
        "track_angular_velocity":  0.5,
        "gentle_landing":          0.0,   # ramped in by curriculum, see below
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, f"missing hop reward: {name}"
        assert cfg.rewards[name].weight == weight, f"unexpected weight on {name}"


def test_hop_tuck_uses_per_foot_sensors_and_leg_joints():
    from mjlab_microduck.tasks.microduck_hop_env_cfg import HOP_TUCK_OVERRIDES

    cfg = make_microduck_hop_env_cfg()
    params = cfg.rewards["hop_tuck"].params
    assert params["left_sensor_name"] == "left_foot_ground_contact"
    assert params["right_sensor_name"] == "right_foot_ground_contact"
    assert params["joint_indices"] == _LEG_JOINTS
    assert params["tuck_overrides"] == HOP_TUCK_OVERRIDES
    # Overrides only touch hip_pitch/knee/ankle (indices 2,3,4 left / 11,12,13
    # right) -- hip_yaw/hip_roll aren't part of the tuck spec.
    assert set(HOP_TUCK_OVERRIDES.keys()) == {2, 3, 4, 11, 12, 13}
    # Mirrored the same way every other pose target in this file is: right
    # side negates hip_pitch/knee/ankle relative to the left.
    assert HOP_TUCK_OVERRIDES[11] == -HOP_TUCK_OVERRIDES[2]
    assert HOP_TUCK_OVERRIDES[12] == -HOP_TUCK_OVERRIDES[3]
    assert HOP_TUCK_OVERRIDES[13] == -HOP_TUCK_OVERRIDES[4]


def test_hop_crouch_uses_per_foot_sensors_and_leg_joints():
    from mjlab_microduck.tasks.microduck_hop_env_cfg import (
        HOP_CROUCH_HEIGHT_MARGIN,
        HOP_CROUCH_OVERRIDES,
    )

    cfg = make_microduck_hop_env_cfg()
    params = cfg.rewards["hop_crouch"].params
    assert params["left_sensor_name"] == "left_foot_ground_contact"
    assert params["right_sensor_name"] == "right_foot_ground_contact"
    assert params["joint_indices"] == _LEG_JOINTS
    assert params["crouch_overrides"] == HOP_CROUCH_OVERRIDES
    assert params["standing_height"] == STAND_Z
    assert params["height_margin"] == HOP_CROUCH_HEIGHT_MARGIN
    assert set(HOP_CROUCH_OVERRIDES.keys()) == {2, 3, 4, 11, 12, 13}
    assert HOP_CROUCH_OVERRIDES[11] == -HOP_CROUCH_OVERRIDES[2]
    assert HOP_CROUCH_OVERRIDES[12] == -HOP_CROUCH_OVERRIDES[3]
    assert HOP_CROUCH_OVERRIDES[13] == -HOP_CROUCH_OVERRIDES[4]
    # The crouch and tuck targets must be genuinely different poses (crouch
    # is deep-knee-bend-while-grounded, tuck is knees-pulled-up-while-
    # airborne) -- catches an accidental copy-paste of one onto the other.
    assert HOP_CROUCH_OVERRIDES != cfg.rewards["hop_tuck"].params["tuck_overrides"]


def test_track_angular_velocity_uses_twist_command():
    cfg = make_microduck_hop_env_cfg()
    assert cfg.rewards["track_angular_velocity"].params["command_name"] == "twist"


def test_already_negative_penalties_use_positive_weights():
    """Lock on the sign-bug class documented across standup/roller_standup/
    roulade: some mdp functions already return a negative magnitude (self-
    negating), and those need a POSITIVE weight or a negative weight turns
    the penalty into a reward for the violation."""
    cfg = make_microduck_hop_env_cfg()
    for name in ("height_stand_l1", "pose_stand_l1", "gentle_landing", "hop_stand_tax"):
        assert cfg.rewards[name].weight >= 0, (
            f"{name} calls a function that already returns a negative value: "
            f"a negative weight here would reward the violation"
        )
    for name in ("action_rate_l2",):
        assert cfg.rewards[name].weight < 0, f"{name} expects a negative weight"


def test_no_ungated_impact_penalty_at_start():
    # gentle_landing (|a_z|) and hop_stand_tax both start at weight 0.0 and
    # are ramped in by curriculum, not active from step 0 — see the file
    # docstring's reasoning (liftoff is plausibly a hard skill; standup's
    # timing lesson is any attempt-tax during discovery makes "do nothing" win).
    cfg = make_microduck_hop_env_cfg()
    assert cfg.rewards["gentle_landing"].weight == 0.0
    assert cfg.rewards["hop_stand_tax"].weight == 0.0


def test_landing_reward_gated_on_min_hop_height():
    cfg = make_microduck_hop_env_cfg()
    for name in ("hop_landing_composite", "hop_stand_tax"):
        assert cfg.rewards[name].params["min_hop_height"] == MIN_HOP_HEIGHT


def test_pose_rewards_target_legs_only_at_home():
    cfg = make_microduck_hop_env_cfg()
    for name in ("pose_stand", "pose_stand_l1"):
        assert cfg.rewards[name].params["joint_indices"] == _LEG_JOINTS
        assert cfg.rewards[name].params["target_overrides"] is None
    assert cfg.rewards["hop_landing_composite"].params["joint_indices"] == _LEG_JOINTS
    assert cfg.rewards["hop_landing_composite"].params["target_overrides"] is None


def test_height_targets_use_stand_z():
    cfg = make_microduck_hop_env_cfg()
    for name in ("height_stand", "height_stand_l1"):
        assert cfg.rewards[name].params["target_height"] == STAND_Z
    assert cfg.rewards["hop_landing_composite"].params["target_height"] == STAND_Z
    assert cfg.rewards["hop_stand_tax"].params["target_height"] == STAND_Z
    # launch bootstrap cuts off just above standing height, not at/below it —
    # otherwise the policy could park at the cutoff without ever finishing
    # the extension (same reasoning as roller_standup's com_upward_velocity gate).
    assert cfg.rewards["hop_launch_velocity"].params["max_height"] == STAND_Z + 0.005


def test_airborne_progress_uses_per_foot_sensors():
    cfg = make_microduck_hop_env_cfg()
    params = cfg.rewards["hop_airborne_progress"].params
    assert params["left_sensor_name"] == "left_foot_ground_contact"
    assert params["right_sensor_name"] == "right_foot_ground_contact"
    assert params["standing_height"] == STAND_Z
    assert params["max_paid_height"] == MAX_PAID_HOP_HEIGHT


def test_body_ang_vel_kept_near_zero():
    # A dynamic liftoff/landing needs free angular velocity briefly — the
    # roulade lesson (locking this down at -0.15+ froze that env's rollover).
    cfg = make_microduck_hop_env_cfg()
    assert cfg.rewards["body_ang_vel"].weight == -0.002
    assert cfg.rewards["angular_momentum"].weight == -0.001


def test_upright_kept_unlike_roulade():
    # UNLIKE roulade (which deletes "upright" because the flip must leave
    # vertical on purpose), hop keeps an always-on upright term — a hop
    # should not turn into a somersault.
    cfg = make_microduck_hop_env_cfg()
    assert "upright_always" in cfg.rewards
    assert cfg.rewards["upright_always"].weight > 0


def test_curriculum_stages_are_monotonic():
    cfg = make_microduck_hop_env_cfg()
    for name in (
        "action_rate_weight",
        "torque_rate_weight",
        "arrival_damping_weight",
        "gentle_landing_weight",
        "hop_stand_tax_weight",
        "com_range",
        "head_com_range",
    ):
        stages = cfg.curriculum[name].params.get("weight_stages") or cfg.curriculum[name].params.get(
            "range_stages"
        )
        steps = [s["step"] for s in stages]
        assert steps == sorted(steps), f"{name} curriculum steps are not monotonic"
        assert len(set(steps)) == len(steps), f"{name} curriculum has duplicate step values"


def test_gentle_landing_and_stand_tax_ramp_up_not_down():
    cfg = make_microduck_hop_env_cfg()
    gl = [s["weight"] for s in cfg.curriculum["gentle_landing_weight"].params["weight_stages"]]
    st = [s["weight"] for s in cfg.curriculum["hop_stand_tax_weight"].params["weight_stages"]]
    assert gl == sorted(gl), "gentle_landing weight should ramp UP (both self-negating, positive weight)"
    assert st == sorted(st), "hop_stand_tax weight should ramp UP"


def test_trunk_asset_cfgs_are_distinct_objects():
    """mjlab resolves and MUTATES SceneEntityCfg in place — a shared object
    across multiple terms causes stale indices. Each term needs its own."""
    cfg = make_microduck_hop_env_cfg()
    names = ("arrival_damping", "gentle_landing", "upright_always")
    seen = [id(cfg.rewards[n].params.get("asset_cfg")) for n in names if "asset_cfg" in cfg.rewards[n].params]
    assert len(set(seen)) == len(seen), "asset_cfg shared between multiple terms"
