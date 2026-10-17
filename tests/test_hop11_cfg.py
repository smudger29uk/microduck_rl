from mjlab_microduck.tasks.microduck_hop5_env_cfg import HOP5_SQUAT_OVERRIDES
from mjlab_microduck.tasks.microduck_hop10_env_cfg import make_microduck_hop10_env_cfg
from mjlab_microduck.tasks.microduck_hop11_env_cfg import (
    HOP11_CROUCH_DEPTH,
    HOP11_CROUCH_FULL_DEPTH,
    HOP11_SQUAT_OVERRIDES,
    MicroduckHop11RlCfg,
    make_microduck_hop11_env_cfg,
)


def test_hop11_squat_is_deeper_and_mirrored():
    s5, s11 = HOP5_SQUAT_OVERRIDES, HOP11_SQUAT_OVERRIDES
    assert s11[3] > s5[3] and s11[4] > s5[4] and s11[2] > s5[2]  # more knee/ankle/hip flexion
    for l, r in ((2, 11), (3, 12), (4, 13)):
        assert s11[l] == -s11[r]
    assert max(abs(v) for v in s11.values()) < 1.4  # inside the +-1.57 joint range with margin
    assert HOP11_CROUCH_DEPTH > 0.032


def test_hop11_changes_only_crouch_params():
    a, b = make_microduck_hop10_env_cfg().rewards, make_microduck_hop11_env_cfg().rewards
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].weight == b[k].weight
    c = b["hop_crouch"].params
    assert c["crouch_depth"] == HOP11_CROUCH_DEPTH
    assert c["squat_overrides"] == HOP11_SQUAT_OVERRIDES
    n = 0
    for k, t in b.items():
        if "crouch_full_depth" in t.params:
            n += 1
            assert t.params["crouch_full_depth"] == HOP11_CROUCH_FULL_DEPTH
    assert n >= 4
    # terminations unchanged from hop10
    assert make_microduck_hop10_env_cfg().terminations.keys() == make_microduck_hop11_env_cfg().terminations.keys()


def test_hop11_does_not_mutate_hop10():
    make_microduck_hop11_env_cfg()
    assert make_microduck_hop10_env_cfg().rewards["hop_crouch"].params["crouch_depth"] == 0.032


def test_hop11_experiment_and_registered():
    assert MicroduckHop11RlCfg.experiment_name == "microduck_hop11"
    import mjlab_microduck.tasks  # noqa
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Hop11-Flat-MicroDuck" in list_tasks()
