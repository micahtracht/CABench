from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_CACHE_UNSET = object()
_loaded_env_path: object = _CACHE_UNSET


def _candidate_dirs() -> list[Path]:
    roots: list[Path] = []
    for anchor in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (anchor, *anchor.parents):
            if candidate not in roots:
                roots.append(candidate)
    return roots


def load_project_env() -> Path | None:
    """
    Load the nearest .env file without overriding existing shell variables.
    Searches upward from the current working directory first, then from here.
    """
    global _loaded_env_path
    if _loaded_env_path is not _CACHE_UNSET:
        return _loaded_env_path if isinstance(_loaded_env_path, Path) else None

    for directory in _candidate_dirs():
        dotenv_path = directory / ".env"
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, override=False)
            _loaded_env_path = dotenv_path
            return dotenv_path

    _loaded_env_path = None
    return None
