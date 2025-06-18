import pytest
from rules import Rule1D

def test_from_int_roundtrip():
    for code in (0, 1, 17, 42, 63):
        rule = Rule1D.from_int(code)
        assert int(rule.rule, 2) == code
        assert len(rule.rule) == 6

@pytest.mark.parametrize(
    "rule_bits,self_state,neigh,sum_expected",
    [
        ("000000", 0, 0, 0),
        ("000000", 1, 2, 0),
        ("111111", 0, 2, 1),
        ("110001", 1, 0, 0),
    ],
)
def test_rule_call(rule_bits, self_state, neigh, sum_expected):
    rule = Rule1D(rule_bits)
    assert rule(self_state, neigh) == sum_expected