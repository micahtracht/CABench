import pytest
from rules import Rule1D, Rule2D

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



def test_rule1d_invalid_rule_length():
    """Constructor should reject bit-strings that are not exactly 6 chars."""
    with pytest.raises(ValueError):
        Rule1D("00000")          # 5 bits
    with pytest.raises(ValueError):
        Rule1D("0" * 7)          # 7 bits


def test_rule1d_from_int_out_of_range():
    """Integer codes ≥ 2**6 must fail because they need > 6 bits."""
    with pytest.raises(ValueError):
        Rule1D.from_int(64)      # 0b1000000 → length 7

def test_rule2d_bits_length_validation():
    """Bit-strings shorter or longer than 18 should raise."""
    for bad_len in (17, 19):
        with pytest.raises(ValueError):
            Rule2D("0" * bad_len)


def test_rule2d_from_int_roundtrip():
    """`from_int` should faithfully encode and decode representative codes."""
    for code in (0, 1, 42, (1 << 18) - 1):
        rule = Rule2D.from_int(code)
        assert len(rule.rule_bits) == 18
        assert int(rule.rule_bits, 2) == code


def test_rule2d_call_all_zeros_ones():
    """All-zero and all-one rule strings act as constant functions."""
    zero_rule = Rule2D("0" * 18)
    one_rule  = Rule2D("1" * 18)

    for self_state in (0, 1):
        for neigh_sum in range(9):          # 0 … 8
            assert zero_rule(self_state, neigh_sum) == 0
            assert one_rule(self_state, neigh_sum)  == 1


def test_rule2d_call_invalid_args():
    """Out-of-range arguments must raise ValueError."""
    rule = Rule2D("0" * 18)

    with pytest.raises(ValueError):
        rule(2, 0)      # invalid self_state

    with pytest.raises(ValueError):
        rule(-1, 0)     # negative self_state

    with pytest.raises(ValueError):
        rule(0, 9)      # neighbor_sum too large

    with pytest.raises(ValueError):
        rule(0, -1)     # neighbor_sum negative