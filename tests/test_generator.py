from generate import ECAProblemGenerator
from generate import Problem1D
from rules import Rule1D

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


def test_num_rule_bits_default_and_custom():
    gen = ECAProblemGenerator(state_size=1)
    assert gen.num_rule_bits() == 6          # 2 states, 2 neighbours
    assert gen.num_rule_bits(num_states=3, neighbor_count=1) == 6  # 3*(1+1)


def test_is_trivial_cases():
    gen = ECAProblemGenerator(state_size=5, seed=0, density=0.0)

    # all-dead start
    dead_problem = Problem1D([0, 0, 0, 0, 0], Rule1D("000000"), timesteps=1)
    assert gen.is_trivial(dead_problem)

    # all-alive start
    alive_problem = Problem1D([1] * 5, Rule1D("111111"), timesteps=1)
    assert gen.is_trivial(alive_problem)

    # rule that leaves state unchanged
    state = [1, 0, 1, 0, 1]
    identity_rule = Rule1D("000111")
    static_problem = Problem1D(state, identity_rule, timesteps=1)
    assert gen.is_trivial(static_problem)

    # clearly non-trivial
    nontrivial_problem = Problem1D(state, Rule1D("110001"), timesteps=1)
    assert not gen.is_trivial(nontrivial_problem)


def test_generate_prompt_contains_start_and_json():
    gen = ECAProblemGenerator(state_size=4, seed=1, density=0.5)
    prob = gen.generate(timesteps=1)
    prompt = gen.generate_prompt_1D(prob, timesteps=1)

    # prompt must include the exact start_state string and the JSON sentinel
    assert str(prob.start_state) in prompt
    assert '{"answer": [' in prompt
    assert prompt.strip().endswith("explanation.")


def test_generate_prompt_batch_length():
    gen = ECAProblemGenerator(state_size=6, seed=2, density=0.4)
    probs = gen.generate_batch(3, timesteps=1)
    prompts = gen.generate_prompt_1d_batch(probs, 1)

    assert len(prompts) == len(probs)
    # ensure ordering is preserved
    for p_obj, p_txt in zip(probs, prompts):
        assert str(p_obj.start_state) in p_txt