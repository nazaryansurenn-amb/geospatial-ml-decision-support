from core.live import ACTION_NOW, ACTION_SOON, ACTION_WAIT, to_actions


def test_to_actions_threshold_mapping():
    probs = [0.1, 0.4, 0.9]
    out = to_actions(probs, T_decision=0.3, T_now=0.8)
    assert out == [ACTION_WAIT, ACTION_SOON, ACTION_NOW]
