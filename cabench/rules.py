from __future__ import annotations
from dataclasses import dataclass


class _RuleBase:
    """
    Shared behavior for outer-totalistic binary rules. Subclasses are frozen
    dataclasses that expose their bit-string via the ``bits`` property and set
    ``num_states``/``neighbor_count``.

    Bit-string layout (length = num_states * (neighbor_count + 1)):
        state 0 outcomes: indices 0 .. neighbor_count
        state 1 outcomes: indices (neighbor_count + 1) .. 2*(neighbor_count + 1) - 1
    Simulation is binary (num_states == 2); the parameter generalizes the bit
    count only.
    """

    num_states: int
    neighbor_count: int

    @property
    def bits(self) -> str:
        raise NotImplementedError

    def _validate(self) -> None:
        expected = self.num_states * (self.neighbor_count + 1)
        if len(self.bits) != expected:
            raise ValueError(
                f"rule bit-string must be {expected} bits "
                f"({self.num_states} states x {self.neighbor_count + 1} sums), "
                f"got {len(self.bits)}"
            )

    def __call__(self, self_state: int, neighbor_sum: int) -> int:
        """Return the next-state bit (0/1) for a cell state and neighbor sum."""
        if not (0 <= self_state < self.num_states):
            raise ValueError("invalid self_state")
        if not (0 <= neighbor_sum <= self.neighbor_count):
            raise ValueError("invalid neighbor sum")
        return int(self.bits[self_state * (self.neighbor_count + 1) + neighbor_sum])


def _bits_from_int(code: int, num_states: int, neighbor_count: int) -> str:
    """Zero-padded binary encoding of a rule code."""
    return f"{code:0{num_states * (neighbor_count + 1)}b}"


@dataclass(frozen=True)
class Rule1D(_RuleBase):
    """Outer-totalistic 1D rule (ECA): 6 bits for 2 states, 2 neighbors."""

    rule: str
    num_states: int = 2
    neighbor_count: int = 2

    def __post_init__(self) -> None:
        self._validate()

    @property
    def bits(self) -> str:
        return self.rule

    @classmethod
    def from_int(cls, code: int, num_states: int = 2, neighbor_count: int = 2) -> "Rule1D":
        """Construct from an integer code, e.g. 0..63 for a 2-state ECA."""
        return cls(
            _bits_from_int(code, num_states, neighbor_count),
            num_states=num_states,
            neighbor_count=neighbor_count,
        )


@dataclass(frozen=True)
class Rule2D(_RuleBase):
    """Outer-totalistic binary rule for a 2D Moore neighborhood: 18 bits."""

    rule_bits: str
    num_states: int = 2
    neighbor_count: int = 8

    def __post_init__(self) -> None:
        self._validate()

    @property
    def bits(self) -> str:
        return self.rule_bits

    @classmethod
    def from_int(cls, code: int, *, num_states: int = 2, neighbor_count: int = 8) -> "Rule2D":
        """Build a Rule2D from an integer in 0 .. 2**(num_states*(neighbor_count+1)) - 1."""
        return cls(
            _bits_from_int(code, num_states, neighbor_count),
            num_states=num_states,
            neighbor_count=neighbor_count,
        )
