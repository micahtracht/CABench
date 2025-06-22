from dataclasses import dataclass
import numpy as np
from typing import List, Sequence, Tuple
from rules import Rule1D
from simulate import step_1d
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
    
    @staticmethod
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
    
    def is_trivial(self, problem: Problem1D) -> bool:
        start_state = problem.start_state
        if all(x == 0 for x in start_state) or all(x == 1 for x in start_state): # prune all dead/all alive
            return True
        if step_1d(start_state, problem.rule) == start_state: # state doesn't ever change
            return True
        return False
    
    def generate_batch(self, num_problems: int, timesteps: Sequence[int] | int, trim_trivial: bool = True, max_attempts_factor: int = 10) -> List[Problem1D]:
        '''
        Makes a list of random Problem1D instances.
        '''
        if isinstance(timesteps, int):
            timesteps = [timesteps] * num_problems
        else:
            if len(timesteps) < num_problems:
                raise ValueError("Length of timesteps must equal the number of problems.")
        
        problems = []
        attempts, max_attempts = 0, max_attempts_factor * num_problems
        while attempts < max_attempts and len(problems) < num_problems:
            idx = len(problems)
            trial_problem = self.generate(timesteps[idx])
            if not trim_trivial or not self.is_trivial(trial_problem):
                problems.append(trial_problem)
            attempts += 1
        if len(problems) < num_problems:
            raise RuntimeError(f'Could only find {len(problems)} nontrivial problems in {max_attempts} attempts.')
        return problems
    
    def generate_prompt_1D(self, problem: Problem1D, timesteps: int = 1) -> str:
        rule = problem.rule.rule # this naming is awful, fix it
        deadToLiving = [i for i in range(len(rule)) if i < 3 and int(rule[i]) == 1]
        livingToDead = [i-3 for i in range(len(rule)) if i >= 3 and int(rule[i]) == 0]
        
        return (
        f"You are given the following initial state of a 1-Dimensional cellular automaton:\n"
        f"{problem.start_state}\n\n"
        "Each cell can be alive (1) or dead (0). Cells outside the boundary are dead.\n"
        "A cell’s neighborhood is its immediate left and right neighbors.\n\n"
        "Transition rules:\n"
        f"-Dead to alive if neighbor count is in {deadToLiving}; else the cell stays dead.\n"
        f"-Alive to dead if neighbor count is in {livingToDead}; else the cell stays alive.\n\n"
        f"After {timesteps} timestep(s), what is the final state?\n\n"
        "Return your answer as valid JSON, for example:\n"
        "`{\"answer\": [0,1,0,1,…]}`\n"
        "Do **not** include any extra text or explanation."
    )
    def generate_prompt_1d_batch(self, problem_list: List[Problem1D], timesteps: int | Sequence[int]) -> List[str]:
        if isinstance(timesteps, int):
            timestep_list = [timesteps] * len(problem_list)
        else:
            if len(timesteps) < len(problem_list):
                raise ValueError("Length of timesteps must equal the number of problems.")
        return [self.generate_prompt_1D(problem_list[i], timestep_list[i]) for i in range(len(problem_list))]