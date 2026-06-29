from cabench.rules import Rule1D, Rule2D


def step_1d(state: list[int], rule: Rule1D) -> list[int]:
    """
    One ECA step w/ outside bounds being treated as dead (0)
    """
    n = len(state)
    next_state = [0] * n
    for i, s in enumerate(state):
        left = state[i - 1] if i > 0 else 0
        right = state[i + 1] if i < n - 1 else 0
        next_state[i] = rule(s, left + right)
    return next_state


def simulate(state: list[int], rule: Rule1D, t: int = 1) -> list[int]:
    curr = state
    for _i in range(t):
        curr = step_1d(curr, rule)
    return curr


def _neighbor_sum(grid: list[list[int]], r: int, c: int) -> int:
    """
    Return number of live neighbors (Moore, eight cells) for cell (r,c).
    Out-of-bounds neighbors are treated as 0 (dead).
    """
    h, w = len(grid), len(grid[0])
    total = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                total += grid[rr][cc]
    return total


def step_2d(grid: list[list[int]], rule: Rule2D) -> list[list[int]]:
    """
    One synchronous update of a 2-D outer-totalistic binary CA
    (Moore neighborhood, zero boundary).
    """
    h, w = len(grid), len(grid[0])
    nxt = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            self_state = grid[r][c]
            neigh_sum = _neighbor_sum(grid, r, c)
            nxt[r][c] = rule(self_state, neigh_sum)
    return nxt


def simulate_2d(grid: list[list[int]], rule: Rule2D, timesteps: int = 1) -> list[list[int]]:
    curr = [row[:] for row in grid]
    for _ in range(timesteps):
        curr = step_2d(curr, rule)
    return curr
