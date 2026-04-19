# Probe Visualization Tool — Design

**Date:** 2026-04-18
**Purpose:** Let a user point at a sample-run directory, cycle through molecules in its TDB files, and visually verify that probe centers from `_probes.bin` align with peaks on the raw molecule waveform.

## Goal

Given a path like `E. coli/Black/STB03-060A-02L58270w05-202G16j/2025-02-19/`, open a desktop plot window that:

1. Discovers the sample run's `_probes.bin` and its referenced TDB files.
2. Iterates molecules defined in `_probes.bin`, filtered by `do_not_use=False` by default.
3. For each molecule, plots the TDB waveform in µV vs ms with probe centers overlaid as vertical markers.
4. Lets the user flip through molecules with the keyboard.

## Non-goals (explicitly out of MVP)

- Multi-sample-directory browsing from within the UI.
- PNG export beyond matplotlib's built-in save button.
- Showing raw int16 samples instead of µV.
- Per-channel filters (the `g` goto key covers targeted access).
- Zoom-to-probe affordances.
- A web UI (option B/C from brainstorming were declined).

## Parser review: findings and fixes

A byte-level investigation of real TDB data (`..._20250219163115.tdb`, 35,319 molecules) confirmed the existing parsers in `src/mongoose/io/tdb.py` and `src/mongoose/io/probes_bin.py` produce correctly aligned output. Key findings:

- **The "undocumented 4-byte field" in TDB molecule blocks is real and benign.** Block boundaries match the index file under the current parsing for every molecule tested. `morph_count == total_data_count` in all cases, which confirms the current interpretation of the waveform sample count.
- **Alignment math is sound.** `sample_idx = (start_within_tdb_ms + probe.center_ms) * sample_rate / 1000` places probe centers on samples with elevated amplitudes relative to baseline.
- **Sample rate is 32000 Hz** for this data (the probes.bin spec's "25 µs" comment refers to processed resolution).
- The `Remapped/AllCh/_probes.bin` references only the TDB files that were remapped; not every TDB in the sample directory is covered. The tool iterates molecules defined in probes.bin, so unreferenced TDBs are implicitly ignored.

Two small parser fixes ship with this tool:

1. `src/mongoose/io/probes_bin.py`: `MOLECULE_FIXED_SIZE = 89` → `93`. Stale constant; unused today but misleading. The actual fixed-field size is 93 bytes per record (verified by summing the `struct.unpack` reads).
2. `src/mongoose/io/tdb.py`: Update the comment on the 4-byte skip to reflect empirical verification — "4 bytes of apparently random content always present between the waveform and the MorphOpen length prefix; `morph_count` consistently equals `total_data_count`; block boundaries align under this interpretation."

No functional parser changes.

## Architecture

Three modules under a new package `src/mongoose/tools/probe_viz/`:

```
src/mongoose/tools/__init__.py           (new, empty)
src/mongoose/tools/probe_viz/__init__.py (new, empty)
src/mongoose/tools/probe_viz/loader.py   (discovery + iteration state; pure, no matplotlib)
src/mongoose/tools/probe_viz/viewer.py   (matplotlib figure, key bindings, rendering)
src/mongoose/tools/probe_viz/__main__.py (argparse + wiring)
```

Keeping `loader.py` free of matplotlib means it is unit-testable without a display and without the optional `viz` dependency.

### loader.py

Responsibilities:

- **Discovery.** Given a sample directory, locate the probes.bin. Search order:
  1. `Remapped/AllCh/*_probes.bin`
  2. `Remapping/AllCh/*_probes.bin` (fallback — see commit `d45e9fb`)
  3. If still not found, fail with a clear error listing the attempted paths.
  If more than one `.bin` is found in the preferred location, pick the first alphabetically and log the others.
- **File resolution.** Parse the `_probes.bin.files` sidecar. For each referenced TDB basename, look for the file as a direct child of the sample directory. If any are missing, raise with the list.
- **TDB cache.** Lazy per-TDB cache mapping basename to `(TdbHeader, tdb_index_dict)`.
- **Iteration state.**
  - `include_do_not_use: bool` (default `False`).
  - An ordered list of indices into `probes.molecules` matching the current filter.
  - `current_position: int` within that list.
  - `advance(delta)` clamps to `[0, len-1]`.
  - `goto_uid(uid)` and `goto_channel_mid(channel, mid)` snap to the matching molecule if it is in the current list, otherwise the nearest one.
  - `toggle_do_not_use()` rebuilds the list and snaps the current position to the nearest remaining molecule by UID distance.
- **Molecule fetch.** `current_view() -> ViewData`, where `ViewData` bundles the `Molecule` record (from probes.bin), the `TdbMolecule` (with waveform), the sample rate, and the amplitude scale factor for this channel. The loader handles the channel-position lookup via `header.channel_ids.index(channel)`.

### viewer.py

Responsibilities:

- Create one matplotlib figure with one axes.
- Render a `ViewData` onto the axes:
  - Waveform: `x = arange(N) / sample_rate * 1000` ms; `y = waveform.astype(float) * scale` µV.
  - Probe centers: `axvline` per probe at `start_within_tdb_ms + probe.center_ms`.
    - Accepted (`attribute & 0x80`): green, solid.
    - Excluded: gray, dashed. Hidden when the `a` toggle hides them.
  - Molecule start line (`start_within_tdb_ms`): amber dashed.
  - Level 1 horizontal line (`mean_lvl1`): faded red dashed.
  - Structured regions (`molecule.structures`): light red `axvspan`. Hidden when `s` toggle is off.
  - TDB edge markers (`rise_conv_max_index`, `fall_conv_min_index` in ms): dotted gray. Hidden by default, `e` toggles.
- Title: `[i/N] UID=u ch=c mid=m   probes=P (A accepted)   structured | do_not_use` with flags shown only when set.
- Corner annotation: TDB basename, translocation time, mean level 1 in µV.
- Key handler bound to the figure's `key_press_event`:
  - `right` / `left`: ±1
  - `shift+right` / `shift+left`: ±10
  - `pageup` / `pagedown`: ±100
  - `home` / `end`: first / last
  - `g`: prompt on stdin for `UID=<n>` or `<ch>:<mid>` and jump
  - `a`: toggle show-excluded-probes
  - `s`: toggle structured-region shading
  - `e`: toggle TDB edge markers
  - `u`: toggle `do_not_use` inclusion (rebuilds iteration list via loader)
  - `r`: reset axis limits to auto
  - `q` / `escape`: close window
- After any state change, a `redraw()` method clears the axes and re-plots; it does not create a new figure.

Matplotlib's default toolbar remains enabled so the user gets pan / zoom / save.

### __main__.py

```
usage: python -m mongoose.tools.probe_viz <sample_dir>
                                          [--include-do-not-use]
                                          [--start-at <uid>]
```

Wires discovery → loader → viewer → `plt.show()`. If `--start-at` is given, calls `loader.goto_uid` before showing.

## Error handling

All failure paths print a clear message and exit non-zero:

- Sample dir does not exist or is not a directory.
- No `_probes.bin` found in either `Remapped/AllCh/` or `Remapping/AllCh/`.
- `_probes.bin.files` references a TDB basename that is not present at the sample dir root.
- `_probes.bin.files` references a TDB basename whose `_index` sidecar is missing.
- Matplotlib not installed (print install hint for `pip install -e .[viz]`).

No retries, no fallbacks beyond `Remapped` / `Remapping`.

## Dependencies

Matplotlib is added to `pyproject.toml` under a new optional group:

```toml
[project.optional-dependencies]
viz = ["matplotlib>=3.7"]
```

Training / CI installs stay unchanged. Users run `pip install -e ".[viz]"` to use the tool.

## Testing

Minimal — this is interactive dev tooling.

1. **Loader smoke test** (`tests/tools/test_probe_viz_loader.py`):
   - Use `pytest.importorskip` / path-exists `skipif` guarding `E. coli/Black/STB03-060A-02L58270w05-202G16j/2025-02-19`.
   - Assert discovery returns exactly one probes.bin.
   - Assert the filtered iteration list is non-empty.
   - Assert `current_view()` returns a `ViewData` with a non-empty waveform and a non-zero sample rate.
   - Assert `advance(1)` moves to a different molecule.
2. **Parser-fix test**: one assertion that `MOLECULE_FIXED_SIZE == 93`.
3. **No viewer unit tests.** Headless matplotlib testing is fragile and the viewer code is thin glue.

Manual acceptance test: launch on the sample dir, cycle 50 molecules, confirm probe centers visibly land on waveform peaks, confirm each documented key binding.

## Implementation order (for writing-plans)

1. Parser-fix commit: update `MOLECULE_FIXED_SIZE` and the TDB comment; ship the parser-fix test.
2. Add `tools/__init__.py` + `probe_viz/__init__.py` skeleton.
3. Implement `loader.py` with `ViewData`, discovery, iteration state, molecule fetch. Ship loader smoke test.
4. Implement `viewer.py`: figure setup, single-molecule render, key bindings.
5. Implement `__main__.py` and wire end to end.
6. Add matplotlib optional dep to `pyproject.toml`.
7. Manual acceptance pass on the sample dir.
