import json, subprocess, tempfile, pathlib, os
from generate_dataset import main as gen_cli

def test_cli_generates_jsonl(tmp_path):
    outfile = tmp_path / "demo.jsonl"
    gen_cli(["--n","10","--size","16","--timesteps","2", "--density","0.4", "--seed","123", "--outfile",str(outfile)])
    lines = outfile.read_text().splitlines()
    
    assert len(lines) == 10
    first = json.loads(lines[0])
    for field in ("rule", "timesteps", "init" ,"target"):
        assert field in first

def _run_generator(tmp_path, *, n=6, size=10, timesteps=2, density=0.35, seed=77):
    """
    Helper that invokes the CLI and returns pathlib.Path to the file.
    """
    outfile = tmp_path / "nested" / "out.jsonl"   # nested dir exercises mkdir
    gen_cli(
        [
            "--n", str(n),
            "--size", str(size),
            "--timesteps", str(timesteps),
            "--density", str(density),
            "--seed", str(seed),
            "--outfile", str(outfile),
        ]
    )
    return outfile


def test_cli_creates_nested_directories(tmp_path):
    out_path = _run_generator(tmp_path)
    assert out_path.exists()
    # parent directory must have been auto-created
    assert out_path.parent.is_dir()


def test_cli_deterministic_with_seed(tmp_path):
    file1 = _run_generator(tmp_path, seed=123)
    file2 = _run_generator(tmp_path, seed=123)    # same args, new file

    assert file1.read_text() == file2.read_text()


def test_targets_same_length_as_state(tmp_path):
    size = 12
    n = 4
    outfile = _run_generator(tmp_path, n=n, size=size)
    lines = outfile.read_text().splitlines()

    for line in lines:
        obj = json.loads(line)
        init = obj["init"]
        target = obj["target"]

        # both are binary strings of length == size
        assert len(init) == size
        assert len(target) == size
        assert set(init).issubset({"0", "1"})
        assert set(target).issubset({"0", "1"})
