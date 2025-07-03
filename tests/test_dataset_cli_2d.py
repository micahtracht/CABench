import json
from generate_dataset import main_2d as gen2d_cli
import pathlib


def _run_2d(tmp_path, *, n=4, h=4, w=4, t=1, density=0.4, seed=7):
    out = tmp_path / "data2d.jsonl"
    gen2d_cli([
        "--n", str(n),
        "--height", str(h),
        "--width", str(w),
        "--timesteps", str(t),
        "--density", str(density),
        "--seed", str(seed),
        "--outfile", str(out),
    ])
    return out


def test_cli_2d_generates_jsonl(tmp_path):
    path = _run_2d(tmp_path)
    lines = path.read_text().splitlines()
    assert len(lines) == 4

    first = json.loads(lines[0])
    for field in ("rule", "timesteps", "init", "target"):
        assert field in first


def test_target_shape_matches_init(tmp_path):
    path = _run_2d(tmp_path, n=2, h=5, w=6)
    for line in path.read_text().splitlines():
        obj = json.loads(line)
        init = obj["init"]
        target = obj["target"]

        assert len(init) == len(target)          # same rows
        for r_init, r_tgt in zip(init, target):
            assert len(r_init) == len(r_tgt)     # same cols
