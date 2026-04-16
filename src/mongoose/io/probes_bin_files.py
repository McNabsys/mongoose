"""Parser for `_probes.bin.files` companion text files.

Each line maps a `file_name_index` (as used in the corresponding probes.bin
Molecule record) to the TDB file that produced it. Lines are CRLF-terminated
with the format:

    <6-digit zero-padded index><absolute source-machine path>\\r\\n

We return only the TDB basenames -- the source-machine absolute paths
(typically D:\\SharedData\\...) are not valid on any processing machine.
"""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath


_LINE_RE = re.compile(r"^(\d{6})(.+)$")


def parse_probes_bin_files(path: str | Path) -> list[str]:
    """Parse a probes.bin.files sidecar into an ordered list of TDB basenames.

    Args:
        path: Path to the `_probes.bin.files` file.

    Returns:
        List of TDB basenames indexed by `file_name_index`. Returned list has
        length `max(index) + 1`.

    Raises:
        ValueError: If indices are duplicated or if there is a gap in the index
            sequence starting from 0.
    """
    path = Path(path)
    # Read raw bytes then decode so that Python's universal-newlines mode
    # doesn't silently fold \r\n to \n before our CRLF split.
    text = path.read_bytes().decode("latin-1")
    lines = [ln for ln in text.split("\r\n") if ln.strip()]

    by_index: dict[int, str] = {}
    for ln in lines:
        m = _LINE_RE.match(ln)
        if not m:
            continue
        idx = int(m.group(1))
        tdb_path = m.group(2)
        basename = PureWindowsPath(tdb_path).name
        if idx in by_index:
            raise ValueError(
                f"{path.name}: duplicate index {idx}: "
                f"{by_index[idx]!r} vs {basename!r}"
            )
        by_index[idx] = basename

    if not by_index:
        return []

    # Require dense indexing starting from 0.
    max_idx = max(by_index)
    for i in range(max_idx + 1):
        if i not in by_index:
            raise ValueError(f"{path.name}: missing index {i} (max index is {max_idx})")

    return [by_index[i] for i in range(max_idx + 1)]
