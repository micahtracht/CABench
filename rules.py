from dataclasses import dataclass
from typing import List, Protocol
from __future__ import annotations

class Rule(Protocol):
    num_states: int
    rule: str # May change later - think about how to encode multi-state rules. For now, work with 1D ECAs, so use a 6 bit string.
    assert len(rule) == 6, "For 1 dimension, rules must have 6 bits."

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