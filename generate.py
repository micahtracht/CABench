import numpy as np
from typing import List, Tuple


def generateRawOTRulesECA(numStates: int, seed: int = 42) -> List[str]:
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


def simulate(state: List[int], rule: str, timesteps: int = 1) -> List[int]:
    currState = state[:]
    for _ in range(timesteps):
        nextState = [0] * len(currState)
        
        for i, cell in enumerate(currState):
            numNeighbors = 0
            if i > 0 and currState[i-1] == 1:
                numNeighbors += 1
            if i < len(currState) - 1 and currState[i+1] == 1:
                numNeighbors += 1
            
            if cell:
                nextState[i] = int(rule[3 + numNeighbors])
            else:
                nextState[i] = int(rule[numNeighbors])
        currState = nextState
    return currState

def generatePrompt1D(state: List[int], rule: str, timesteps: int) -> str:
    deadToLiving = [i for i in range(len(rule)) if i < 3 and int(rule[i]) == 1]
    livingToDead = [i-3 for i in range(len(rule)) if i >= 3 and int(rule[i]) == 0]
    
    return f"You are given the following initial state of a 1-Dimensional cellular automaton: {state}. Each cell can either be alive (1) or dead (0). All cells outside the boundary are considered dead.\nA cell's neighborhood consists of its two immediate neighbors: one to the left, and one to the right.\nThe automaton evolves according to the following rules:\nIf a cell is dead and its number of neighbors is in {deadToLiving}, it becomes living. Otherwise, the cell remains dead.\nIf a cell is alive and its number of neighbors is in {livingToDead}, it becomes dead. Otherwise, the cell remains living.\nWhat is the final state after {timesteps} timestep(s)?\nReturn the result as a binary string of equal length to the initial state without spaces or extra text of any kind."

rule = generateRawOTRulesECA(seed = 10)
print(rule)
state = [1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
print(state)
print(simulate(state, rule))

