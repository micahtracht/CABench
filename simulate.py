from rules import Rule1D
from typing import List

def step_1d(state: List[int], rule: Rule1D) -> List[int]:
    '''
    One ECA step w/ outside bounds being treated as dead (0)
    '''
    n = len(state)
    next_state = [0] * n
    for i, s in enumerate(state):
        left = state[i-1] if i > 0 else 0
        right = state[i+1] if i < n-1 else 0
        next_state[i] = rule(s, left + right)
    return next_state

def simulate(state: List[int], rule: Rule1D, t: int = 1) -> List[int]:
    curr = state
    for i in range(t):
        curr = step_1d(curr, rule)
    return curr