"""Parse HDM instrument Run Logs to extract per-molecule telemetry.

Input: the RunLog.txt emitted by HD-Mapping alongside TDB files.
Format: ``//Key: Value`` header lines, blank line, then a tab-separated
event table. Event rows have the shape::

    <MM/DD/YYYY HH:MM:SS AM/PM>  <Source>  <Message>  <DataKey>  <DataValue>

Continuation rows drop the first three cells and only carry
``<DataKey> <DataValue>`` — they belong to the most recently seen parent
event's timestamp.

V3 pilot needs three telemetry channels extracted from here:
  * Bias voltage (V)         — from ``Bias`` / ``FinalBias`` / ``Original Bias``
  * NanoPress (psi)          — from ``NanoPress``
  * Per-channel baseline (mV) — from ``Baselines`` (256-element vector)

All three are event-triggered (piecewise-constant over run time), so the
API returns per-timestamp lookups via nearest-preceding search, which
collapses cleanly to a single scalar (or 256-vector) per molecule given
the molecule's start timestamp.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

_TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"
_TS_RE = re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} (?:AM|PM)$")
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class RunLog:
    """Parsed Run Log with telemetry timelines.

    Attributes:
        header: ``//Key: Value`` metadata from the file head.
        start_time: Parsed ``Run Start Time`` from the header.
        bias_timeline: List of (timestamp, bias_V) in chronological order.
        pressure_timeline: List of (timestamp, NanoPress_psi) in chronological order.
        baseline_timeline: List of (timestamp, per-channel mV array [256])
            in chronological order.
    """

    header: dict[str, str] = field(default_factory=dict)
    start_time: datetime | None = None
    bias_timeline: list[tuple[datetime, float]] = field(default_factory=list)
    pressure_timeline: list[tuple[datetime, float]] = field(default_factory=list)
    baseline_timeline: list[tuple[datetime, np.ndarray]] = field(default_factory=list)

    def bias_at(self, ts: datetime) -> float | None:
        """Nearest-preceding Bias voltage at ``ts``. None if before first event."""
        return _nearest_preceding_scalar(self.bias_timeline, ts)

    def nano_press_at(self, ts: datetime) -> float | None:
        """Nearest-preceding NanoPress at ``ts``. None if before first event."""
        return _nearest_preceding_scalar(self.pressure_timeline, ts)

    def baseline_at(self, ts: datetime) -> np.ndarray | None:
        """Nearest-preceding baseline mV vector at ``ts``. None if before first event."""
        for t, v in reversed(self.baseline_timeline):
            if t <= ts:
                return v
        return None


def _nearest_preceding_scalar(
    timeline: list[tuple[datetime, float]], ts: datetime
) -> float | None:
    for t, v in reversed(timeline):
        if t <= ts:
            return v
    return None


def _parse_bias_value(raw: str) -> float | None:
    """Extract numeric bias from strings like '3.050', '3.132 V', '3.05V'."""
    m = _FLOAT_RE.search(raw)
    return float(m.group()) if m else None


def _parse_pressure_value(raw: str) -> float | None:
    """Extract numeric pressure from strings like '1.00 psi', '0.50psi', '0.50'."""
    m = _FLOAT_RE.search(raw)
    return float(m.group()) if m else None


def _parse_baseline_vector(raw: str) -> np.ndarray | None:
    """Parse a '[a, b, c, ...]' array, optionally with ' mV' unit suffixes.

    The baseline data comes as either ``[1008.1,1173.8,...]`` (comma-separated
    bare floats) or ``[1008.1 mV,1173.8 mV,...]`` (with unit suffixes).
    Returns ``None`` if the string doesn't look like a list of numbers.
    """
    raw = raw.strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        return None
    inside = raw[1:-1]
    values: list[float] = []
    for item in inside.split(","):
        m = _FLOAT_RE.search(item)
        if m is None:
            return None
        values.append(float(m.group()))
    if not values:
        return None
    return np.asarray(values, dtype=np.float32)


def parse_run_log(path: Path) -> RunLog:
    """Parse an HDM RunLog.txt into a ``RunLog`` with telemetry timelines.

    Args:
        path: Path to the ``*_RunLog.txt`` file.

    Returns:
        Populated ``RunLog``. Missing telemetry channels are empty lists.
    """
    rl = RunLog()
    current_ts: datetime | None = None

    with open(path, encoding="utf-8") as f:
        in_header = True
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if in_header:
                if line.startswith("//"):
                    body = line[2:]
                    if ":" in body:
                        key, val = body.split(":", 1)
                        rl.header[key.strip()] = val.strip()
                    continue
                # Blank line or first non-// line terminates header.
                if line.strip() == "":
                    continue
                in_header = False
                # The first non-// line is the column header row —
                # "Date/Time\tSource\tMessage\tData". Skip it.
                if line.startswith("Date/Time"):
                    continue

            # Body rows. Split on tab. Empty leading cells mean continuation.
            cells = line.split("\t")
            if not cells or all(c == "" for c in cells):
                continue

            first_cell = cells[0].strip()
            if _TS_RE.match(first_cell):
                try:
                    current_ts = datetime.strptime(first_cell, _TS_FORMAT)
                except ValueError:
                    current_ts = None
                # Primary row — cells: [ts, source, message, key, value]
                key = cells[3].strip() if len(cells) > 3 else ""
                value = cells[4].strip() if len(cells) > 4 else ""
            else:
                # Continuation row — cells: ["", "", key, value]
                key = cells[2].strip() if len(cells) > 2 else ""
                value = cells[3].strip() if len(cells) > 3 else ""

            if current_ts is None or not key:
                continue

            _ingest(rl, current_ts, key, value)

    # Parse Run Start Time from header (format: "2:56:14.975 PM 2/24/2025")
    rt = rl.header.get("Run Start Time")
    if rt:
        rl.start_time = _parse_run_start_time(rt)

    return rl


def _parse_run_start_time(raw: str) -> datetime | None:
    """Parse 'H:MM:SS.fff AM/PM M/D/YYYY' from the header."""
    try:
        # Handle milliseconds by stripping them if they confuse strptime.
        cleaned = re.sub(r"\.\d+", "", raw)
        return datetime.strptime(cleaned, "%I:%M:%S %p %m/%d/%Y")
    except ValueError:
        return None


def _ingest(rl: RunLog, ts: datetime, key: str, value: str) -> None:
    """Route a parsed (key, value) pair into the appropriate telemetry timeline."""
    if key in ("Bias", "FinalBias", "Original Bias"):
        v = _parse_bias_value(value)
        if v is not None:
            rl.bias_timeline.append((ts, v))
    elif key == "NanoPress":
        v = _parse_pressure_value(value)
        if v is not None:
            rl.pressure_timeline.append((ts, v))
    elif key == "Baselines":
        arr = _parse_baseline_vector(value)
        if arr is not None:
            rl.baseline_timeline.append((ts, arr))
