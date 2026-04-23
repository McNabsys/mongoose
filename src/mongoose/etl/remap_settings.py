"""Parser for Nabsys `_remapSettings.txt` / `_remapStickySettings.txt`.

File format: one ``key=value`` per line, ``//`` comments, blank lines ignored.
Values are untyped strings. Callers coerce via :func:`get_float`,
:func:`get_int`, :func:`get_bool` — each returns None (and does not crash)
when the key is missing, so a run with an older setting schema is still
ingestible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemapSettings:
    path: Path
    values: dict[str, str]

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.values.get(key, default)

    def get_float(self, key: str, default: float | None = None) -> float | None:
        raw = self.values.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def get_int(self, key: str, default: int | None = None) -> int | None:
        raw = self.values.get(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            # Some "int" values are written as floats (e.g., "2.0"); fall back.
            try:
                return int(float(raw))
            except ValueError:
                return default

    def get_bool(self, key: str, default: bool | None = None) -> bool | None:
        raw = self.values.get(key)
        if raw is None:
            return default
        return raw.strip().lower() in {"true", "1", "yes"}


def load_remap_settings(path: Path | str) -> RemapSettings:
    path = Path(path)
    values: dict[str, str] = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return RemapSettings(path=path, values=values)
