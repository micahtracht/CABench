from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from rules import Rule1D

@dataclass
class Problem1D:
    '''
    A single 1D, ECA problem w/ initial state, rule, and the number of timesteps.
    '''
    start_state: List[int]
    rule: Rule1D
    timesteps: int = 1


class ECAProblemGenerator:
    '''
    This is the generator for 1D ECA problems. (outer totalistic)
    '''
    def __init__(self, state_size: int, seed: int = 42, density: float = 0.5):
        self.state_size = state_size
        self.density = density
        self.rng = np.random.default_rng(seed)
    
    def num_rule_bits(num_states: int = 2, neighbor_count: int = 2) -> int:
        '''
        Number of bits needed to represent the ruleset, given the number of states and the number of neighbors:
        num_states * (neighbor_count + 1)
        '''
        return num_states * (neighbor_count + 1)
    
    def _make_rule(self) -> Rule1D:
        '''
        Generate a random rule by sampling an integer (between 0 and 2**bit_count - 1) and converting to binary.
        '''
        needed_bits = self.num_rule_bits()
        integer_code = int(self.rng.integers(0, 2**needed_bits)) # in [0, 2**needed_bits - 1]
        return Rule1D.from_int(integer_code)

    def _make_state(self) -> List[int]:
        '''
        Generate a random 1D starting state, with the density of ones given by self.density.
        '''
        return [int(self.rng.random() < self.density) for _ in range(self.state_size)]

    def generate(self, timesteps: int = 1) -> Problem1D:
        '''
        Uses functions to generate state and rule, then makes and returns a Problem1D object using those.
        '''
        state = self._make_state()
        rule = self._make_rule()
        return Problem1D(start_state=state, rule=rule, timesteps=timesteps)
    
    def generate_batch(self, num_problems: int, timesteps: List[int]) -> List[Problem1D]:
        '''
        Makes a list of random Problem1D instances.
        '''
        return [self.generate(timesteps[i]) for i in range(num_problems)]
        


def generateRawRulesECA(numStates: int, seed: int = 42) -> List[str]:
    assert 0 < numStates <= 64, "Only up to 64 distinct OT-ECA rules exist."
    rng = np.random.default_rng(seed = seed)
    
    rules = set() # use set to avoid duplicates
    
    while len(rules) < numStates:
        rule = rng.integers(0, 64) # doesn't include 64
        binaryRule = bin(rule)
        binaryRule = binaryRule[2:] # exclude 0b
        while len(binaryRule) < 6:
            binaryRule = "0" + binaryRule
        rules.add(binaryRule)
    
    return sorted(list(rules)) # sorted for reproducibility between runs

def generateStartStates(stateSize: int, numStates: int, seed: int = 42, density: float = 0.5) -> List[List[int]]:
    rng = np.random.default_rng(seed = seed)
    values = rng.random(stateSize * numStates)
    states = []
    for i in range(numStates):
        newState = []
        for j in range(stateSize):
            idx = i*stateSize + j
            if values[idx] > 1 - density:
                newState.append(1)
            else:
                newState.append(0)
        states.append(newState)
    return states

def generateProblems(stateSize: int, numStates: int, seed: int = 42, density: float = 0.5) -> List[Tuple[List[int], str]]:
    states = generateStartStates(stateSize, numStates, seed = seed, density = density)
    rules = generateRawOTRulesECA(numStates, seed = seed)
    
    return list(zip(states, rules))


def generatePrompt1D(state: List[int], rule: str, timesteps: int) -> str:
    deadToLiving = [i for i in range(len(rule)) if i < 3 and int(rule[i]) == 1]
    livingToDead = [i-3 for i in range(len(rule)) if i >= 3 and int(rule[i]) == 0]
    
    return f"You are given the following initial state of a 1-Dimensional cellular automaton: {state}. Each cell can either be alive (1) or dead (0). All cells outside the boundary are considered dead.\nA cell's neighborhood consists of its two immediate neighbors: one to the left, and one to the right.\nThe automaton evolves according to the following rules:\nIf a cell is dead and its number of neighbors is in {deadToLiving}, it becomes living. Otherwise, the cell remains dead.\nIf a cell is alive and its number of neighbors is in {livingToDead}, it becomes dead. Otherwise, the cell remains living.\nWhat is the final state after {timesteps} timestep(s)?\nReturn the result as a binary string of equal length to the initial state without spaces or extra text of any kind."


