from __future__ import annotations

import os
from pathlib import Path

from cabench import env


def test_load_project_env_reads_repo_local_dotenv(tmp_path: Path, monkeypatch):
    project_root = tmp_path / "project"
    nested = project_root / "nested" / "work"
    package_dir = project_root / "cabench"
    nested.mkdir(parents=True)
    package_dir.mkdir(parents=True)
    dotenv_path = project_root / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    monkeypatch.chdir(nested)
    monkeypatch.setattr(env, "__file__", str(package_dir / "env.py"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(env, "_loaded_env_path", env._CACHE_UNSET)

    loaded = env.load_project_env()

    assert loaded == dotenv_path
    assert env.load_project_env() == dotenv_path
    assert os.getenv("OPENAI_API_KEY") == "from-dotenv"
