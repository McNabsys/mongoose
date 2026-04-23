"""Parser for Nabsys `_version.txt` files.

Tiny key=value file, at minimum:

    PROGRAM_VERSION=<semver-ish>
    PICKER=<picker-id>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VersionInfo:
    program_version: str | None
    picker: str | None
    extra: dict[str, str]


def load_version_file(path: Path | str) -> VersionInfo:
    path = Path(path)
    values: dict[str, str] = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    program_version = values.pop("PROGRAM_VERSION", None)
    picker = values.pop("PICKER", None)
    return VersionInfo(
        program_version=program_version,
        picker=picker,
        extra=values,
    )
