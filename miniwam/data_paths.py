"""Shared dataset/cache locations, without importing the model runtime.

Set MINIWAM_DATA_ROOT to an absolute directory on the mounted NAS. Existing
``datasets/...`` config paths are then interpreted relative to that directory.
Explicit absolute paths retain their meaning.
"""

import os
from pathlib import Path


def data_root() -> Path:
    value = os.environ.get("MINIWAM_DATA_ROOT")
    if value is None:
        return Path("datasets")
    root = Path(value).expanduser()
    if not value.strip() or not root.is_absolute():
        raise ValueError("MINIWAM_DATA_ROOT must be an absolute directory on the mounted NAS")
    if "," in str(root):
        raise ValueError("MINIWAM_DATA_ROOT cannot contain commas (feature roots use comma separators)")
    return root


def data_path(*parts: str) -> Path:
    return data_root().joinpath(*parts)


def resolve_data_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if "MINIWAM_DATA_ROOT" in os.environ and path.parts[:1] == ("datasets",):
        return data_path(*path.parts[1:])
    return path


def resolve_feature_roots(value: str) -> str:
    return ",".join(str(resolve_data_path(part.strip()))
                    for part in value.split(",") if part.strip())
