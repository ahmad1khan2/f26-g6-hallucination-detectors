import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str) -> dict:
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hash_key(seed: int, key: str) -> str:
    """Stable ordering key: sorting by it is the project's only 'shuffle'."""
    return hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
