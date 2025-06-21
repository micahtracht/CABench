import json, subprocess, tempfile, pathlib, os
from CABenchRoot.generate_dataset import main as gen_cli

def test_cli_generates_jsonl(tmp_path):
    outfile = tmp_path / "demo.jsonl"
    gen_cli(["--n","10","--size","16","--timesteps","2", "--density","0.4", "--seed","123", "--outfile",str(outfile)])
    lines = outfile.read_text().splitlines()
    
    assert len(lines) == 10
    first = json.loads(lines[0])
    for field in ("rule", "timesteps", "init" ,"target"):
        assert field in first