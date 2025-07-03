from generate import CAProblemGenerator2D, Problem2D
from rules import Rule2D


def test_deterministic_seed_2d():
    g1 = CAProblemGenerator2D(4, 5, seed=123, density=0.4)
    g2 = CAProblemGenerator2D(4, 5, seed=123, density=0.4)

    p1 = g1.generate()
    p2 = g2.generate()

    assert p1.start_grid == p2.start_grid
    assert p1.rule.rule_bits == p2.rule.rule_bits


def test_density_bounds_2d():
    h, w = 64, 64
    density = 0.3
    gen = CAProblemGenerator2D(h, w, seed=0, density=density)
    grid = gen.generate().start_grid

    ones = sum(cell for row in grid for cell in row)
    actual = ones / (h * w)
    assert abs(actual - density) < 0.03      # within ±3 pp


def test_batch_nontrivial_2d():
    gen = CAProblemGenerator2D(6, 6, seed=1, density=0.5)
    batch = gen.generate_batch(10, timesteps=1)
    assert len(batch) == 10
    assert not any(gen.is_trivial(p) for p in batch)


def test_generate_prompt_contains_grid_and_json_2d():
    gen = CAProblemGenerator2D(3, 3, seed=2, density=0.5)
    prob = gen.generate()
    prompt = gen.generate_prompt_2D(prob, timesteps=1)

    assert str(prob.start_grid) in prompt
    assert '{"answer": [[' in prompt
    assert prompt.strip().endswith("explanation.")
