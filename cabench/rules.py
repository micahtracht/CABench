from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol


class Rule(Protocol):
    num_states: int
    rule: str # May change later - think about how to encode multi-state rules. For now, work with 1D ECAs, so use a 6 bit string.

    def __call__(self, self_state: int, timesteps: int) -> int:
        ...

@dataclass(frozen=True)
class Rule1D:
    """
    Outer-totalistic 1D rule encoded as a flat bit-string:
    state 0 transitions: bits[0..N]
    state 1 transitions: bits[N+1..2N+1]
    ...
    state (S-1) transitions: bits[(S-1)*(N+1) .. S*(N+1)-1]
    where N = max number of neighbors (2 for ECA).
    """
    rule: str
    num_states: int = 2
    neighbor_count: int = 2
    
    def __post_init__(self):
        expected_length = self.num_states * (self.neighbor_count + 1)
        if len(self.rule) != expected_length:
            raise ValueError(f"rule string must be {expected_length} bits long, got {len(self.rule)}")

    def __call__(self, self_state: int, neighbor_sum: int) -> int:
        return int(self.rule[self_state * (self.neighbor_count + 1) + neighbor_sum])
    
    @classmethod
    def from_int(cls, code: int, num_states: int = 2, neighbor_count: int = 2) -> Rule1D:
        """Construct from integer code, e.g. 0..63 for 2-state ECA."""
        bits = f"{code:0{num_states*(neighbor_count+1)}b}"
        return cls(bits, num_states=num_states, neighbor_count=neighbor_count)


@dataclass(frozen=True)
class Rule2D:
    """
    Outer-totalistic binary rule for 2-D Moore neighborhood.
    Bitstring layout (length = num_states x (neighbor_count + 1)):
        state 0 outcomes: indices 0 .. 8   (sum = 0-8)
        state 1 outcomes: indices 9 .. 17
    """
    rule_bits: str
    num_states: int = 2
    neighbor_count: int = 8 # Moore neighborhood

    def __post_init__(self) -> None:
        expected = self.num_states * (self.neighbor_count + 1)
        if len(self.rule_bits) != expected:
            raise ValueError(
                f"rule_bits length {len(self.rule_bits)} "
                f"does not match expected {expected} "
                f"({self.num_states} states × {self.neighbor_count + 1} sums)."
            )

    def __call__(self, self_state: int, neighbor_sum: int) -> int:
        """Return next-state bit (0/1) for given cell state & neighbor sum."""
        if not (0 <= self_state < self.num_states):
            raise ValueError("invalid self_state")
        if not (0 <= neighbor_sum <= self.neighbor_count):
            raise ValueError("invalid neighbor sum")

        idx = self_state * (self.neighbor_count + 1) + neighbor_sum
        return int(self.rule_bits[idx])

    @classmethod
    def from_int(cls,code: int, *, num_states: int = 2, neighbor_count: int = 8) -> "Rule2D":
        """
        Build a Rule2D from an integer in the range
            0 .. 2**(num_states*(neighbor_count+1)) - 1
        """
        bit_len = num_states * (neighbor_count + 1)
        rule_bits = f"{code:0{bit_len}b}"          # zero-padded binary
        return cls(rule_bits, num_states=num_states, neighbor_count=neighbor_count)
