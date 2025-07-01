import pytest
from simulate import simulate_2d, step_1d, simulate, step_2d
from rules import Rule1D, Rule2D
from simulate import _neighbor_sum

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

def test_zero_length_state():
    """Empty lattice should stay empty for any rule / timestep."""
    rule = Rule1D("000000")
    empty = []
    assert step_1d(empty, rule) == []
    assert simulate(empty, rule, 5) == []


def test_all_dead_state_invariant():
    """With a rule that keeps dead cells dead, an all-zero state is fixed."""
    rule = Rule1D("000111")               # dead -> dead for all neighbor sums
    state = [0, 0, 0, 0, 0]
    assert step_1d(state, rule) == state
    assert simulate(state, rule, 4) == state


def test_all_alive_state_invariant():
    """Rule with alive→alive for all sums keeps an all-one state unchanged."""
    rule = Rule1D("111111")               # alive -> alive, dead bits irrelevant
    state = [1, 1, 1, 1, 1]
    assert step_1d(state, rule) == state
    assert simulate(state, rule, 3) == state


def test_neighbor_sum_corners_and_center():
    """
    Validate neighbor counts for a 3x3 grid with ones on the diagonal:
    1 0 0
    0 1 0
    0 0 1
    """
    grid = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    # corner (0,0) → only (1,1) is a live neighbor
    assert _neighbor_sum(grid, 0, 0) == 1
    # center (1,1) → two live neighbors (0,0) and (2,2)
    assert _neighbor_sum(grid, 1, 1) == 2

def test_step2d_all_zero_rule():
    """A rule of all zeros should wipe the grid in one step."""
    rule = Rule2D("0" * 18)
    grid = [
        [1, 0, 1],
        [0, 1, 0],
    ]
    expected = [[0] * len(grid[0]) for _ in grid]
    assert step_2d(grid, rule) == expected


def test_simulate2d_all_one_rule_multi_step():
    """
    All-one rule turns every cell live on the first step and keeps it live.
    Multi-step simulation should therefore yield an all-ones grid.
    """
    rule = Rule2D("1" * 18)
    grid = [
        [0, 0],
        [1, 0],
    ]
    h, w = len(grid), len(grid[0])
    expected = [[1] * w for _ in range(h)]
    assert simulate_2d(grid, rule, 3) == expected