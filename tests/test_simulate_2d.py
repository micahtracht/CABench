from simulate import step_2d, simulate_2d, _neighbor_sum
from rules import Rule2D


def test_step2d_zero_rule_blanket():
    rule = Rule2D("0" * 18)          # everything dies
    grid = [[1, 0], [0, 1]]
    expected = [[0, 0], [0, 0]]
    assert step_2d(grid, rule) == expected


def test_step2d_one_rule_blanket():
    rule = Rule2D("1" * 18)          # everything lives
    grid = [[0, 0], [1, 0]]
    expected = [[1, 1], [1, 1]]
    assert simulate_2d(grid, rule, 2) == expected   # stays all-ones


def test_neighbor_sum_edge_and_center():
    diag = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]

    assert _neighbor_sum(diag, 0, 0) == 1   # corner sees middle
    assert _neighbor_sum(diag, 1, 1) == 2   # centre sees two ones
