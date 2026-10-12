from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop5_env_cfg import make_microduck_hop5_env_cfg
from mjlab_microduck.tasks.microduck_hop7_env_cfg import (
    HOP7_GATE_WIDTH,
    HOP7_HEAD_POSE_WEIGHT,
    HOP7_MAX_PAID_HEIGHT,
    HOP7_MIN_HOP_HEIGHT,
    MicroduckHop7RlCfg,
    make_microduck_hop7_env_cfg,
)


def test_hop7_same_reward_terms_as_hop5():
    assert make_microduck_hop5_env_cfg().rewards.keys() == make_microduck_hop7_env_cfg().rewards.keys()


def test_hop7_gates_and_cap():
    r = make_microduck_hop7_env_cfg().rewards
    assert r["hop_airborne_progress"].params["max_paid_height"] == HOP7_MAX_PAID_HEIGHT
    for k in ("hop_crouch", "hop_landing_composite", "hop_stand_tax"):
        assert r[k].params["min_hop_height"] == HOP7_MIN_HOP_HEIGHT
        assert r[k].params["gate_width"] == HOP7_GATE_WIDTH


def test_hop7_tuck_is_home_pose_and_head_weight():
    r = make_microduck_hop7_env_cfg().rewards
    assert r["hop_tuck"].params["tuck_overrides"] == {}
    assert r["head_pose_l1"].weight == HOP7_HEAD_POSE_WEIGHT > 0  # self-negating penalty


def test_hop7_does_not_change_hop5():
    assert make_microduck_hop5_env_cfg().rewards["head_pose_l1"].weight == 1.0
    assert make_microduck_hop5_env_cfg().events["set_hop_state"].func is not microduck_mdp.reset_hop7_state


def test_hop7_reset_and_experiment():
    ev = make_microduck_hop7_env_cfg().events["set_hop_state"]
    assert ev.func is microduck_mdp.reset_hop7_state
    assert MicroduckHop7RlCfg.experiment_name == "microduck_hop7"


def test_hop7_registered():
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Hop7-Flat-MicroDuck" in list_tasks()
