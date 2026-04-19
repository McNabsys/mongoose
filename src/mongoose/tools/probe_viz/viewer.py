"""Matplotlib viewer for the probe-viz tool."""

from __future__ import annotations

import numpy as np

from .loader import ProbeVizLoader, ViewData


# Probe attribute bit 7: Probe Accepted
_ATTR_ACCEPTED = 0x80


class ProbeVizViewer:
    """Matplotlib window showing one molecule's waveform with probe overlays."""

    def __init__(self, loader: ProbeVizLoader) -> None:
        # Import lazily so that loader tests don't require matplotlib.
        import matplotlib.pyplot as plt

        self._plt = plt
        self.loader = loader
        self.show_excluded_probes = True
        self.show_structures = True
        self.show_tdb_edges = False

        self.fig, self.ax = plt.subplots(figsize=(13, 6))
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.1)
        self._info_text = None
        self.redraw()

    def show(self) -> None:
        self._plt.show()

    def redraw(self) -> None:
        self.ax.clear()
        view = self.loader.current_view()
        self._render(view)
        self.fig.canvas.draw_idle()

    def _render(self, view: ViewData) -> None:
        pm = view.probe_molecule
        tm = view.tdb_molecule

        # Waveform in µV vs ms
        n = tm.waveform.size
        t_ms = np.arange(n) / view.sample_rate * 1000.0
        y_uv = tm.waveform.astype(np.float64) * view.scale_uv_per_lsb

        self.ax.plot(t_ms, y_uv, color="steelblue", linewidth=0.6, label="waveform")

        # Level 1 horizontal
        self.ax.axhline(
            pm.mean_lvl1, color="firebrick", linestyle="--",
            linewidth=0.7, alpha=0.5, label=f"level1={pm.mean_lvl1:.0f}µV",
        )

        # Molecule start (where probes are measured relative to)
        self.ax.axvline(
            pm.start_within_tdb_ms, color="goldenrod", linestyle="--",
            linewidth=0.8, alpha=0.8, label=f"mol start={pm.start_within_tdb_ms:.2f}ms",
        )

        # Structured regions (behind waveform)
        if self.show_structures:
            for s in pm.structures:
                self.ax.axvspan(
                    pm.start_within_tdb_ms + s.start_time,
                    pm.start_within_tdb_ms + s.end_time,
                    color="salmon", alpha=0.15, zorder=0,
                )

        # TDB convolution edges (optional)
        if self.show_tdb_edges:
            rise_ms = tm.rise_conv_max_index / view.sample_rate * 1000.0
            fall_ms = tm.fall_conv_min_index / view.sample_rate * 1000.0
            self.ax.axvline(rise_ms, color="gray", linestyle=":", linewidth=0.7)
            self.ax.axvline(fall_ms, color="gray", linestyle=":", linewidth=0.7)

        # Probe centers
        accepted = 0
        for probe in pm.probes:
            is_accepted = bool(probe.attribute & _ATTR_ACCEPTED)
            if not is_accepted and not self.show_excluded_probes:
                continue
            t = pm.start_within_tdb_ms + probe.center_ms
            color = "seagreen" if is_accepted else "dimgray"
            ls = "-" if is_accepted else "--"
            self.ax.axvline(t, color=color, linestyle=ls, linewidth=0.8, alpha=0.85)
            if is_accepted:
                accepted += 1

        self.ax.set_xlabel("time (ms, relative to TDB block start)")
        self.ax.set_ylabel("amplitude (µV)")
        self.ax.set_xlim(0, t_ms[-1] if n else 1.0)
        self.ax.grid(True, alpha=0.2)

        flags = []
        if pm.structured:
            flags.append("structured")
        if pm.do_not_use:
            flags.append("do_not_use")
        flag_str = f"   {' | '.join(flags)}" if flags else ""
        self.ax.set_title(
            f"[{view.iter_index + 1}/{view.iter_total}]  "
            f"UID={pm.uid}  ch={pm.channel}  mid={pm.molecule_id}   "
            f"probes={len(pm.probes)} ({accepted} accepted){flag_str}",
            loc="left", fontsize=10,
        )

        info = (
            f"{view.tdb_basename}\n"
            f"transloc = {pm.transloc_time_ms:.2f} ms\n"
            f"mean_lvl1 = {pm.mean_lvl1:.1f} µV"
        )
        self.ax.text(
            0.99, 0.98, info, transform=self.ax.transAxes,
            ha="right", va="top", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85),
        )
