from dataclasses import dataclass
import numpy as np
from typing import List, Sequence, Tuple
from rules import Rule1D, Rule2D
from simulate import step_1d, step_2d
@dataclass
class Problem1D:
    '''
    A single 1D, ECA problem w/ initial state, rule, and the number of timesteps.
    '''
    start_state: List[int]
    rule: Rule1D
    timesteps: int = 1

@dataclass
class Problem2D:
    start_grid: List[List[int]]
    rule: Rule2D
    timesteps: int = 1

class CAProblemGenerator2D:
    """
    Random 2-D outer-totalistic binary CA task generator.
    Grid cells outside the HxW rectangle are treated as 0 (dead).
    """
    def __init__(self, height: int, width: int, *, seed: int = 42, density: float = 0.5):
        self.h = height
        self.w = width
        self.density = density
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def num_rule_bits(num_states: int = 2, neighbor_count: int = 8) -> int:
        """
        Number of bits needed to represent the ruleset (outer-totalistic binary):
        num_states * (neighbor_count + 1).
        """
        return num_states * (neighbor_count + 1)

    def _make_rule(self) -> Rule2D:
        """
        Generate a random 2-D outer-totalistic binary rule by sampling an integer
        and converting to an 18-bit string.
        """
        bit_len = self.num_rule_bits()
        code = int(self.rng.integers(0, 2**bit_len))
        return Rule2D.from_int(code)

    def _make_grid(self) -> List[List[int]]:
        """
        Generate a random HxW binary grid with the given density.
        """
        return [
            [int(self.rng.random() < self.density) for _ in range(self.w)]
            for _ in range(self.h)
        ]

    def generate(self, timesteps: int = 1) -> Problem2D:
        """
        Create a single 2-D CA problem with a random start grid and rule.
        """
        grid = self._make_grid()
        rule = self._make_rule()
        return Problem2D(start_grid=grid, rule=rule, timesteps=timesteps)

    def is_trivial(self, problem: Problem2D) -> bool:
        """
        Return True if the problem is trivial: all cells dead, all alive, or
        if one step leaves the grid unchanged.
        """
        flat = [cell for row in problem.start_grid for cell in row]
        if all(x == 0 for x in flat) or all(x == 1 for x in flat):
            return True
        return step_2d(problem.start_grid, problem.rule) == problem.start_grid

    def generate_batch(
        self,
        num_problems: int,
        timesteps: Sequence[int] | int,
        trim_trivial: bool = True,
        max_attempts_factor: int = 10,
    ) -> List[Problem2D]:
        """
        Generate a batch of non-trivial 2-D CA problems.
        If trim_trivial is True, filters out trivial problems.
        """
        if isinstance(timesteps, int):
            timesteps_list = [timesteps] * num_problems
        else:
            if len(timesteps) < num_problems:
                raise ValueError("Length of timesteps must equal number of problems.")
            timesteps_list = list(timesteps)

        problems: List[Problem2D] = []
        attempts = 0
        max_attempts = max_attempts_factor * num_problems

        while attempts < max_attempts and len(problems) < num_problems:
            idx = len(problems)
            prob = self.generate(timesteps_list[idx])
            if not trim_trivial or not self.is_trivial(prob):
                problems.append(prob)
            attempts += 1

        if len(problems) < num_problems:
            raise RuntimeError(
                f"Could only create {len(problems)}/{num_problems} nontrivial problems "
                f"in {max_attempts} attempts."
            )

        return problems
    
    
    def generate_prompt_2D(self, problem: Problem2D, timesteps: int = 1) -> str:
        """
        Return a text prompt instructing an LLM to predict the final grid state.
        Format and wording deliberately parallel the 1-D prompt builder.
        """
        rule_bits = problem.rule.rule_bits  # length 18, indices 0-8 (dead), 9-17 (alive)
        dead_to_live   = [i for i in range(9) if rule_bits[i] == "1"]
        alive_to_dead  = [i for i in range(9) if rule_bits[9 + i] == "0"]

        return (
            "You are given the following initial state of a 2-Dimensional cellular automaton:\n"
            f"{problem.start_grid}\n\n"
            "Each cell can be alive (1) or dead (0). Cells outside the boundary are dead.\n"
            "A cell's neighborhood is its eight surrounding cells (Moore neighborhood).\n\n"
            "Transition rules:\n"
            f"- Dead -> alive if neighbor count is in {dead_to_live}; otherwise the cell stays dead.\n"
            f"- Alive -> dead if neighbor count is in {alive_to_dead}; otherwise the cell stays alive.\n\n"
            f"After {timesteps} timestep(s), what is the final state?\n\n"
            "You may write your reasoning first.\n"
            "Then, on the very last line of your reply, output only a valid JSON object such as:\n"
            '{"answer": [[0,1,0],[1,0,1]]}\n'
            "Do not include any extra text or explanation."
        )


    def generate_prompt_2d_batch(self, problems: Sequence[Problem2D], timesteps: int | Sequence[int] = 1) -> List[str]:
        """
        Vectorised helper analogous to generate_prompt_1d_batch.
        """
        if isinstance(timesteps, int):
            t_list = [timesteps] * len(problems)
        else:
            if len(timesteps) < len(problems):
                raise ValueError("Length of timesteps must equal the number of problems.")
            t_list = list(timesteps)

        return [self.generate_prompt_2D(prob, t_list[idx]) for idx, prob in enumerate(problems)]

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
        rule_bits = problem.rule.rule
        dead_to_living = [i for i in range(3) if rule_bits[i] == "1"]
        living_to_dead = [i - 3 for i in range(3, 6) if rule_bits[i] == "0"]

        return (
            f"You are given the following initial state of a 1-Dimensional cellular automaton:\n"
            f"{problem.start_state}\n\n"
            "Each cell can be alive (1) or dead (0). Cells outside the boundary are dead.\n"
            "A cell's neighborhood is its immediate left and right neighbors.\n\n"
            "Transition rules:\n"
            f"- Dead → alive if neighbor count ∈ {dead_to_living}; otherwise the cell stays dead.\n"
            f"- Alive → dead if neighbor count ∈ {living_to_dead}; otherwise the cell stays alive.\n\n"
            f"After {timesteps} timestep(s), what is the final state?\n\n"
            "You may write your reasoning first.\n"
            'Then, on the very last line of your reply, output only a valid JSON object such as:\n'
            '{"answer": [0,1,0,1]}\n'
            "Do not include any extra text or explanation."
        )
    
    def generate_prompt_1d_batch(self, problem_list: List[Problem1D], timesteps: int | Sequence[int]) -> List[str]:
        if isinstance(timesteps, int):
            timestep_list = [timesteps] * len(problem_list)
        else:
            timestep_list = list(timesteps)
            if len(timestep_list) != len(problem_list):
                raise ValueError("Length of timesteps must equal the number of problems.")

        return [self.generate_prompt_1D(problem_list[i], timestep_list[i]) for i in range(len(problem_list))]
