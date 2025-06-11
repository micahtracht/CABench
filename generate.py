import numpy as np
from typing import List
'''
Let's start with 1D ECAs. We'll also only use outer-totalistic rules. This only has 64 potential rules.

For this, we have to generate mappings for each of 3 sums (0, 1, 2) based on state of interior.
6 mappings we need to generate.
'''
def generateOTRulesECA(seed: int = 42) -> str:
    rng = np.random.default_rng(seed = seed)
    
    rule = rng.integers(0, 64) # doesn't include 64
    binaryRule = bin(rule)
    binaryRule = binaryRule[2:] # exclude 0b
    while len(binaryRule) < 6:
        binaryRule = "0" + binaryRule
    return binaryRule

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

def scoreResponse(correct: str, response: str) -> float:
    '''
    
    '''
    if not correct:
        return 0.0
    
    hammingDistance = sum(c != r for c, r in zip(correct ,response))
    hammingDistance += abs(len(correct) - len(response)) # responses too long or too short have all excess/nonexistent characters counted as incorrect
    
    return 1 - (hammingDistance/max(len(correct), len(response))) # in [0, 1]

rule = generateOTRulesECA(seed = 10)
print(rule)
state = [1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
print(state)
print(simulate(state, rule))

