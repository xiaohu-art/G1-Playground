from pathlib import Path


def resolve_repo_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve().as_posix()
