from generate import ECAProblemGenerator
from generate import Problem1D

def test_deterministic_seed():
    gen1 = ECAProblemGenerator(state_size=8, seed=0, density=0.5)
    gen2 = ECAProblemGenerator(state_size=8, seed=0, density=0.5)
    prob1 = gen1.generate()
    prob2 = gen2.generate()
    assert prob1.start_state == prob2.start_state
    assert prob1.rule.rule == prob2.rule.rule

def test_density_bounds():
    gen = ECAProblemGenerator(state_size=25000, seed=1, density=0.2)
    state = gen.generate().start_state
    density = sum(state) / len(state)
    assert 0.18 < density < 0.22 # the chance density is not in [0.18, 0.22] | code works correctly is ~10^-15.

def test_batch_nontrivial():
    gen = ECAProblemGenerator(state_size=12, seed=2, density=0.5)
    batch = gen.generate_batch(20, timesteps=3)
    assert len(batch) == 20
    assert not any (gen.is_trivial(p) for p in batch)