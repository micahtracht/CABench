from simulate import step_1d, simulate
from rules import Rule1D

def test_step_identity_rule0():
    rule_0 = Rule1D("000000") # this should make every cell go to 0
    state = [1, 0, 1, 1]
    assert step_1d(state, rule_0) == [0, 0, 0, 0]

def test_step_rule_identity():
    r_id = Rule1D("000111")
    state = [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    assert step_1d(state, r_id) == state

def test_multi_step_consistency():
    r = Rule1D("110001")
    state = [1, 0, 0, 1]
    one = step_1d(state, r)
    two  = step_1d(one, r)
    assert simulate(state, r, 2) == two