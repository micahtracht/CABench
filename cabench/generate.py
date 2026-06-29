from dataclasses import dataclass
import numpy as np
from typing import List, Sequence
from cabench.rules import Rule1D, Rule2D
from cabench.simulate import simulate, simulate_2d


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


def _transition_sets(rule) -> tuple[List[int], List[int]]:
    """
    Derive the (dead->alive, alive->dead) neighbor-sum sets from a rule, using its
    neighbor_count so the prose can never diverge from the actual rule bits.
    """
    nc = rule.neighbor_count
    bits = rule.bits
    dead_to_live = [s for s in range(nc + 1) if bits[s] == "1"]
    alive_to_dead = [s for s in range(nc + 1) if bits[(nc + 1) + s] == "0"]
    return dead_to_live, alive_to_dead


class _CABaseGenerator:
    """
    Shared logic for the 1D/2D outer-totalistic CA problem generators. Subclasses
    provide the dimension-specific hooks; everything else (rule sampling, batch
    generation, triviality, prompt batching) lives here.
    """

    NEIGHBOR_COUNT: int

    def num_rule_bits(self, num_states: int = 2, neighbor_count: int | None = None) -> int:
        '''
        Number of bits needed to represent the ruleset (outer-totalistic):
        num_states * (neighbor_count + 1).
        '''
        nc = self.NEIGHBOR_COUNT if neighbor_count is None else neighbor_count
        return num_states * (nc + 1)

    def _make_rule(self):
        '''Sample a random rule integer and convert it to the rule bit-string.'''
        bit_len = self.num_rule_bits()
        code = int(self.rng.integers(0, 2 ** bit_len))
        return self._rule_from_int(code)

    def generate(self, timesteps: int = 1):
        '''
        Create a single problem with a random initial state and rule. The initial
        state is sampled BEFORE the rule to keep the RNG sequence deterministic.
        '''
        initial = self._make_initial()
        rule = self._make_rule()
        return self._make_problem(initial, rule, timesteps)

    def is_trivial(self, problem) -> bool:
        '''
        Trivial if the start is all-dead/all-alive, or the state evolves back to
        its start over the full timestep horizon (the answer copies the input).
        '''
        cells = self._flat_cells(problem)
        if all(x == 0 for x in cells) or all(x == 1 for x in cells):
            return True
        initial = self._initial_of(problem)
        return self._simulate(initial, problem.rule, problem.timesteps) == initial

    def generate_batch(
        self,
        num_problems: int,
        timesteps: Sequence[int] | int,
        trim_trivial: bool = True,
        max_attempts_factor: int = 10,
    ) -> List:
        '''
        Generate a batch of (optionally non-trivial) problems.
        '''
        if isinstance(timesteps, int):
            timesteps_list = [timesteps] * num_problems
        else:
            if len(timesteps) < num_problems:
                raise ValueError("Length of timesteps must equal number of problems.")
            timesteps_list = list(timesteps)

        problems: List = []
        attempts = 0
        max_attempts = max_attempts_factor * num_problems
        while attempts < max_attempts and len(problems) < num_problems:
            prob = self.generate(timesteps_list[len(problems)])
            if not trim_trivial or not self.is_trivial(prob):
                problems.append(prob)
            attempts += 1

        if len(problems) < num_problems:
            raise RuntimeError(
                f"Could only create {len(problems)}/{num_problems} nontrivial problems "
                f"in {max_attempts} attempts."
            )
        return problems

    def _prompt_batch(self, problems: Sequence, timesteps: int | Sequence[int], builder) -> List[str]:
        '''Apply a per-problem prompt builder over a batch with shared timestep handling.'''
        if isinstance(timesteps, int):
            t_list = [timesteps] * len(problems)
        else:
            t_list = list(timesteps)
            if len(t_list) < len(problems):
                raise ValueError("Length of timesteps must equal the number of problems.")
        return [builder(problems[i], t_list[i]) for i in range(len(problems))]

    # --- dimension-specific hooks ---
    def _rule_from_int(self, code: int):
        raise NotImplementedError

    def _make_initial(self):
        raise NotImplementedError

    def _make_problem(self, initial, rule, timesteps):
        raise NotImplementedError

    def _initial_of(self, problem):
        raise NotImplementedError

    def _flat_cells(self, problem) -> List[int]:
        raise NotImplementedError

    def _simulate(self, initial, rule, timesteps):
        raise NotImplementedError


class ECAProblemGenerator(_CABaseGenerator):
    '''
    Generator for 1D ECA problems (outer-totalistic, left/right neighbors).
    '''

    NEIGHBOR_COUNT = 2

    def __init__(self, state_size: int, seed: int = 42, density: float = 0.5):
        self.state_size = state_size
        self.density = density
        self.rng = np.random.default_rng(seed)

    def _rule_from_int(self, code: int) -> Rule1D:
        return Rule1D.from_int(code)

    def _make_initial(self) -> List[int]:
        return [int(self.rng.random() < self.density) for _ in range(self.state_size)]

    def _make_problem(self, initial: List[int], rule: Rule1D, timesteps: int) -> Problem1D:
        return Problem1D(start_state=initial, rule=rule, timesteps=timesteps)

    def _initial_of(self, problem: Problem1D) -> List[int]:
        return problem.start_state

    def _flat_cells(self, problem: Problem1D) -> List[int]:
        return problem.start_state

    def _simulate(self, initial: List[int], rule: Rule1D, timesteps: int) -> List[int]:
        return simulate(initial, rule, timesteps)

    def generate_prompt_1D(self, problem: Problem1D, timesteps: int = 1) -> str:
        dead_to_living, living_to_dead = _transition_sets(problem.rule)

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
        return self._prompt_batch(problem_list, timesteps, self.generate_prompt_1D)


class CAProblemGenerator2D(_CABaseGenerator):
    '''
    Random 2-D outer-totalistic binary CA task generator.
    Grid cells outside the HxW rectangle are treated as 0 (dead).
    '''

    NEIGHBOR_COUNT = 8

    def __init__(self, height: int, width: int, *, seed: int = 42, density: float = 0.5):
        self.h = height
        self.w = width
        self.density = density
        self.rng = np.random.default_rng(seed)

    def _rule_from_int(self, code: int) -> Rule2D:
        return Rule2D.from_int(code)

    def _make_initial(self) -> List[List[int]]:
        return [
            [int(self.rng.random() < self.density) for _ in range(self.w)]
            for _ in range(self.h)
        ]

    def _make_problem(self, initial: List[List[int]], rule: Rule2D, timesteps: int) -> Problem2D:
        return Problem2D(start_grid=initial, rule=rule, timesteps=timesteps)

    def _initial_of(self, problem: Problem2D) -> List[List[int]]:
        return problem.start_grid

    def _flat_cells(self, problem: Problem2D) -> List[int]:
        return [cell for row in problem.start_grid for cell in row]

    def _simulate(self, initial: List[List[int]], rule: Rule2D, timesteps: int) -> List[List[int]]:
        return simulate_2d(initial, rule, timesteps)

    def generate_prompt_2D(self, problem: Problem2D, timesteps: int = 1) -> str:
        """
        Return a text prompt instructing an LLM to predict the final grid state.
        Format and wording deliberately parallel the 1-D prompt builder.
        """
        dead_to_live, alive_to_dead = _transition_sets(problem.rule)

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
        return self._prompt_batch(problems, timesteps, self.generate_prompt_2D)
