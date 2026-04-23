"""Plot training convergence curves from train.py log files.

Produces a grid of subplots — one per loss component — with all runs
overlaid for comparison. Covers both scaled (what the loss function
weights) and raw (physically interpretable) views:

  * Total loss (train + val)
  * Probe head loss
  * L_511 per-probe RMSE (sqrt of bp_raw, interpretable bp)
  * Smoothness (velocity TV)
  * L_length span error (sqrt of count_raw, interpretable bp)

Usage:
    python scripts/plot_training_curves.py \\
        --log l511_spike_train.log --label "L_511 spike" \\
        --log l511_spike_ext_train.log --label "L_511 extension" \\
        --log option_a_smoke_train.log --label "Option A" \\
        --output training_curves.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Matches the per-epoch summary lines produced by trainer.Trainer._log:
#   Epoch 1/3 | loss=0.6573 | probe=0.2876 | bp=0.0925 | vel=6.0140 | count=0.5423 |
#     val_loss=0.8465 | blend=1.000 | lr=0.000229 |
#     raw[p=0.29 bp=24153 vel=6 count=542305852.87]
_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*"
    r"loss=(?P<loss>[\d.eE+-]+)\s*\|\s*"
    r"probe=(?P<probe>[\d.eE+-]+)\s*\|\s*"
    r"bp=(?P<bp>[\d.eE+-]+)\s*\|\s*"
    r"vel=(?P<vel>[\d.eE+-]+)\s*\|\s*"
    r"count=(?P<count>[\d.eE+-]+)\s*\|\s*"
    r"val_loss=(?P<val_loss>[\d.eE+-]+).*?"
    r"raw\[p=(?P<probe_raw>[\d.eE+-]+)\s+"
    r"bp=(?P<bp_raw>[\d.eE+-]+)\s+"
    r"vel=(?P<vel_raw>[\d.eE+-]+)\s+"
    r"count=(?P<count_raw>[\d.eE+-]+)"
)


def parse_log(path: Path) -> dict[str, list[float]]:
    """Return dict of metric name -> list of floats, one per epoch."""
    out: dict[str, list[float]] = {
        "epoch": [],
        "loss": [],
        "probe": [],
        "bp": [],
        "vel": [],
        "count": [],
        "val_loss": [],
        "probe_raw": [],
        "bp_raw": [],
        "vel_raw": [],
        "count_raw": [],
    }
    with open(path) as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if m is None:
                continue
            out["epoch"].append(int(m.group(1)))
            for k in ("loss", "probe", "bp", "vel", "count", "val_loss",
                     "probe_raw", "bp_raw", "vel_raw", "count_raw"):
                out[k].append(float(m.group(k)))
    return out


def _plot_series(
    ax: plt.Axes,
    runs: list[tuple[str, dict]],
    field: str,
    title: str,
    ylabel: str,
    *,
    transform=None,
    log_y: bool = False,
    also_val: bool = False,
    hline: float | None = None,
    hline_label: str | None = None,
) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (lbl, d) in enumerate(runs):
        c = colors[i % len(colors)]
        y = np.asarray(d[field], dtype=float)
        if transform is not None:
            y = transform(y)
        ax.plot(d["epoch"], y, marker="o", linestyle="-", color=c, label=lbl)
        if also_val and "val_loss" in d:
            vy = np.asarray(d["val_loss"], dtype=float)
            ax.plot(d["epoch"], vy, marker="s", linestyle="--",
                    color=c, alpha=0.5, label=f"{lbl} (val)")
    if hline is not None:
        ax.axhline(y=hline, color="gray", linestyle=":", linewidth=1, alpha=0.6,
                   label=hline_label or "")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True, type=Path,
                        help="Training log file. Repeat for multiple runs.")
    parser.add_argument("--label", action="append", required=True, type=str,
                        help="Label for each --log (same order).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Training convergence")
    args = parser.parse_args()

    if len(args.log) != len(args.label):
        raise SystemExit("--log and --label must have matching counts")

    runs: list[tuple[str, dict]] = []
    for log, lbl in zip(args.log, args.label):
        data = parse_log(log)
        if not data["epoch"]:
            print(f"WARNING: no epochs parsed from {log}")
            continue
        print(f"Parsed {len(data['epoch'])} epochs from {log} ({lbl})")
        runs.append((lbl, data))

    # 3x2 grid.
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(args.title, fontsize=13, y=0.995)

    # Row 0: total loss (log y) + val loss (separate)
    _plot_series(axes[0, 0], runs, "loss",
                 title="Train loss (total weighted sum)",
                 ylabel="loss", log_y=True)
    _plot_series(axes[0, 1], runs, "val_loss",
                 title="Val loss",
                 ylabel="val_loss", log_y=True)

    # Row 1: probe head + L_511 per-probe RMSE
    _plot_series(axes[1, 0], runs, "probe_raw",
                 title="Probe head loss (raw CenterNet focal)",
                 ylabel="probe_raw")
    _plot_series(axes[1, 1], runs, "bp_raw",
                 title="L_511 per-probe RMSE — sqrt(bp_raw) (the physics metric)",
                 ylabel="per-probe RMSE (bp)",
                 transform=np.sqrt,
                 hline=50, hline_label="~10% of 511 bp (target)")

    # Row 2: smoothness + length span
    _plot_series(axes[2, 0], runs, "vel_raw",
                 title="L_smooth velocity TV (raw)",
                 ylabel="vel_raw")
    _plot_series(axes[2, 1], runs, "count_raw",
                 title="L_length span error — sqrt(count_raw) (first-to-last probe bp span)",
                 ylabel="span error (bp)",
                 transform=np.sqrt, log_y=True)

    # Shared x-label on last row only.
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
