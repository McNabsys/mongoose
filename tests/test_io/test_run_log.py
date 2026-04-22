"""Tests for ``mongoose.io.run_log`` using the real Black-STB03-060A sample."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mongoose.io.run_log import parse_run_log

# Real example supplied for the V3 reconnaissance spike. If the repo is
# checked out without the E. coli directory (e.g. CI), skip.
EXAMPLE_RUN_LOG = Path(
    "C:/git/mongoose/E. coli/Black/STB03-060A-02L58270w05-433B23e/"
    "2025-02-24/"
    "STB03-060A-02L58270w05-433B23e-OhmX433-433_20250224145614_RunLog.txt"
)


@pytest.mark.skipif(not EXAMPLE_RUN_LOG.exists(), reason="example RunLog not present")
def test_parses_header_and_start_time() -> None:
    rl = parse_run_log(EXAMPLE_RUN_LOG)
    assert rl.header["System ID"] == "OhmX433"
    assert rl.header["Sample ID"] == "STB03-060A-02L58270w05-433B23e"
    assert rl.start_time == datetime(2025, 2, 24, 14, 56, 14)


@pytest.mark.skipif(not EXAMPLE_RUN_LOG.exists(), reason="example RunLog not present")
def test_extracts_bias_timeline() -> None:
    rl = parse_run_log(EXAMPLE_RUN_LOG)
    assert len(rl.bias_timeline) >= 3, f"expected multiple bias events, got {len(rl.bias_timeline)}"
    # All biases in the 2.5–3.5 V range (normal operating setpoints).
    for ts, v in rl.bias_timeline:
        assert 2.0 < v < 4.0, f"bias {v} V at {ts} outside plausible range"


@pytest.mark.skipif(not EXAMPLE_RUN_LOG.exists(), reason="example RunLog not present")
def test_extracts_nano_press_timeline() -> None:
    rl = parse_run_log(EXAMPLE_RUN_LOG)
    assert len(rl.pressure_timeline) >= 2
    for ts, p in rl.pressure_timeline:
        assert 0.0 <= p <= 5.0, f"pressure {p} psi outside plausible range"


@pytest.mark.skipif(not EXAMPLE_RUN_LOG.exists(), reason="example RunLog not present")
def test_extracts_256_channel_baselines() -> None:
    rl = parse_run_log(EXAMPLE_RUN_LOG)
    assert len(rl.baseline_timeline) >= 2
    for ts, arr in rl.baseline_timeline:
        assert arr.shape == (256,), f"baseline not 256-ch at {ts}: shape {arr.shape}"
        # Typical mV range for open-pore current is 500-1500.
        assert 0.0 < arr.mean() < 2000.0


@pytest.mark.skipif(not EXAMPLE_RUN_LOG.exists(), reason="example RunLog not present")
def test_nearest_preceding_lookup() -> None:
    rl = parse_run_log(EXAMPLE_RUN_LOG)
    # TDB file timestamp: 15:27:19 (from its filename).
    tdb_ts = datetime(2025, 2, 24, 15, 27, 19)

    bias = rl.bias_at(tdb_ts)
    nano = rl.nano_press_at(tdb_ts)
    baseline = rl.baseline_at(tdb_ts)

    assert bias is not None and 2.0 < bias < 4.0
    assert nano is not None and 0.0 <= nano <= 5.0
    assert baseline is not None and baseline.shape == (256,)


@pytest.mark.skipif(not EXAMPLE_RUN_LOG.exists(), reason="example RunLog not present")
def test_lookup_before_first_event_returns_none() -> None:
    rl = parse_run_log(EXAMPLE_RUN_LOG)
    # Far in the past — should precede every event.
    t0 = datetime(2020, 1, 1, 0, 0, 0)
    assert rl.bias_at(t0) is None
    assert rl.nano_press_at(t0) is None
    assert rl.baseline_at(t0) is None
