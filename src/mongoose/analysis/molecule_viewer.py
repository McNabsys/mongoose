"""Per-molecule deep-view figure generator.

One function, :func:`render_molecule_figure`, which takes a molecule
uid and produces a 5-row PNG comparing:

  1. Raw waveform  (current vs time)
  2. Signal-processing peak finding  (from probes.bin: green=accepted,
     gray=rejected)
  3. V3 model peak finding            (model heatmap + local-maxima ticks)
  4. Legacy T2D on genome             (ticks mapped to genome bp)
  5. V3 model T2D on genome           (ticks mapped to genome bp)

Rows 4 and 5 overlay all reference sites from referenceMap.txt that
fall within the molecule's claimed genomic window -- those dots are
physical ground truth (Nb.BssSI recognition sites on E. coli MG1655),
not the aligner's per-probe assignment. The aligner's per-probe
assignment is used ONLY to establish the molecule-local <-> genome bp
affine mapping (via OLS on matched probes). Individual red ticks then
show WHICH reference sites each method thinks each detected probe
belongs to -- giving the reader visual grounds to accept or reject the
aligner's per-probe assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.inference.legacy_t2d import TDB_SAMPLE_RATE_HZ, legacy_t2d_bp_positions
from mongoose.io.assigns import MoleculeAssignment
from mongoose.io.probes_bin import Molecule as PbinMolecule
from mongoose.io.reference_map import ReferenceMap
from mongoose.io.transform import ChannelTransform


COLOR_ACCEPT = "#2ca02c"  # green
COLOR_REJECT = "#888888"  # gray
COLOR_SP_T2D = "#f4c20d"  # yellow
COLOR_V3 = "#4285f4"      # blue
COLOR_REF = "#db4437"     # red
COLOR_REF_SITE = "#888888"  # light gray for background ref sites


@dataclass
class MoleculePredictions:
    """Per-molecule model outputs + metadata needed by the viewer."""

    uid: int
    channel: int
    run_id: str
    n_tags_matched: int
    alignment_score: int
    secondbest_score: int
    translocation_time_ms: float
    waveform: np.ndarray                 # int16 raw current samples (cached-waveform coords)
    mask: np.ndarray                     # bool, length == waveform
    v3_heatmap: np.ndarray               # float, probe heatmap over samples
    v3_cum_bp: np.ndarray                # float, cumulative bp over samples
    sp_probe_center_ms: np.ndarray       # all detected probes, molecule-local ms
    sp_probe_duration_ms: np.ndarray     # parallel to sp_probe_center_ms
    sp_probe_accepted: np.ndarray        # bool parallel
    warmstart_centers_samples: np.ndarray  # matched probe centers in cached-sample coords
    warmstart_center_ms: np.ndarray        # parallel, molecule-local ms
    ref_genomic_bp: np.ndarray             # matched probe ref positions (aligner assignment)
    t2d_bp_local: np.ndarray               # legacy T2D prediction, molecule-local bp
    v3_bp_local: np.ndarray                # V3 cum_bp at matched centers, molecule-local bp


def compute_predictions(
    *,
    batch: dict,
    manifest_entry: dict,
    pbin_mol: PbinMolecule,
    assignment: MoleculeAssignment,
    transform: ChannelTransform,
    run_id: str,
    model: torch.nn.Module,
    device: torch.device,
    use_hybrid: bool,
) -> MoleculePredictions:
    """Run the model + T2D on a single-molecule batch and assemble the
    ``MoleculePredictions`` record the renderer consumes."""
    waveform = batch["waveform"].to(device)
    conditioning = batch["conditioning"].to(device)
    mask = batch["mask"].to(device)
    t2d_params = batch["t2d_params"].to(device) if use_hybrid else None
    with torch.no_grad():
        probe, cum_bp, _vel, _logits = model(
            waveform, conditioning, mask, t2d_params=t2d_params,
        )
    probe = probe.float().cpu().numpy()[0]
    cum_bp = cum_bp.float().cpu().numpy()[0]
    # The collated waveform has a channel dim (B, C, T); take the first
    # channel since the cache is single-channel normalized current.
    wf = batch["waveform"][0].detach().cpu().numpy().squeeze()
    mk = batch["mask"][0].detach().cpu().numpy().astype(bool).squeeze()

    centers = batch["warmstart_probe_centers_samples"][0].detach().cpu().numpy().astype(np.int64)
    ref_bp = batch["reference_bp_positions"][0].detach().cpu().numpy().astype(np.int64)

    t2d_local = legacy_t2d_bp_positions(
        centers, mol=pbin_mol,
        mult_const=transform.mult_const, addit_const=transform.addit_const, alpha=transform.alpha,
    )
    v3_local = cum_bp[centers].astype(np.float64)

    # Detected probes (SP output) -- all of them, in probes.bin detection order.
    sp_center_ms = np.array([p.center_ms for p in pbin_mol.probes], dtype=np.float64)
    sp_duration_ms = np.array([p.duration_ms for p in pbin_mol.probes], dtype=np.float64)
    sp_accepted = np.array([bool((p.attribute >> 7) & 1) for p in pbin_mol.probes], dtype=bool)

    # Warmstart centers to ms (same convention as legacy_t2d):
    sample_period_ms = 1000.0 / TDB_SAMPLE_RATE_HZ
    warmstart_ms = centers.astype(np.float64) * sample_period_ms - float(pbin_mol.start_within_tdb_ms)

    return MoleculePredictions(
        uid=int(pbin_mol.uid),
        channel=int(pbin_mol.channel),
        run_id=run_id,
        n_tags_matched=int(sum(1 for v in assignment.probe_indices if v > 0)),
        alignment_score=int(assignment.alignment_score),
        secondbest_score=int(assignment.second_best_score),
        translocation_time_ms=float(pbin_mol.transloc_time_ms),
        waveform=wf,
        mask=mk,
        v3_heatmap=probe,
        v3_cum_bp=cum_bp,
        sp_probe_center_ms=sp_center_ms,
        sp_probe_duration_ms=sp_duration_ms,
        sp_probe_accepted=sp_accepted,
        warmstart_centers_samples=centers,
        warmstart_center_ms=warmstart_ms,
        ref_genomic_bp=ref_bp,
        t2d_bp_local=t2d_local,
        v3_bp_local=v3_local,
    )


def _fit_local_to_genome(
    local_bp: np.ndarray, genome_bp: np.ndarray,
) -> tuple[float, float]:
    """OLS fit of genome_bp = slope * local_bp + intercept."""
    x = local_bp.astype(np.float64)
    y = genome_bp.astype(np.float64)
    n = x.size
    if n < 2:
        return 1.0, 0.0
    sx, sy, sxx, sxy = x.sum(), y.sum(), (x * x).sum(), (x * y).sum()
    denom = n * sxx - sx * sx
    if denom <= 0:
        return 1.0, float(y[0] - x[0])
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return float(slope), float(intercept)


def render_molecule_figure(
    pred: MoleculePredictions,
    *,
    refmap: ReferenceMap,
    output_path: Path,
    peak_threshold: float = 0.3,
    peak_distance_samples: int = 20,
) -> Path:
    """Render the 5-row deep-view figure for one molecule."""
    # --- Affine map molecule-local bp -> genome bp, using matched probes. ---
    # Fit on (warmstart local bp, ref genomic bp). For molecule-local, use V3's
    # predictions at those centers as a neutral anchor (either T2D or V3 local
    # would work; V3 was trained to this signal).
    slope, intercept = _fit_local_to_genome(pred.v3_bp_local, pred.ref_genomic_bp)

    # Genome window covered by the molecule (with a modest buffer).
    g_lo = float(pred.ref_genomic_bp.min())
    g_hi = float(pred.ref_genomic_bp.max())
    g_span = g_hi - g_lo if g_hi > g_lo else 1000.0
    buffer = 0.1 * g_span
    window_lo = g_lo - buffer
    window_hi = g_hi + buffer

    # Reference sites inside the window -- ground truth.
    mask_sites = (refmap.probe_positions >= window_lo) & (
        refmap.probe_positions <= window_hi
    )
    ref_sites_genome = refmap.probe_positions[mask_sites].astype(np.float64)

    # T2D / V3 predictions in genome coordinates.
    t2d_genome = slope * pred.t2d_bp_local + intercept
    v3_genome = slope * pred.v3_bp_local + intercept

    # ----- Figure layout: 5 rows. -----
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        5, 1, height_ratios=[1.6, 0.7, 1.3, 0.7, 0.7],
        hspace=0.45,
    )
    ax_wave = fig.add_subplot(gs[0])
    ax_sp = fig.add_subplot(gs[1], sharex=ax_wave)
    ax_v3 = fig.add_subplot(gs[2], sharex=ax_wave)
    ax_t2d_g = fig.add_subplot(gs[3])
    ax_v3_g = fig.add_subplot(gs[4], sharex=ax_t2d_g)

    # Time axis for rows 1-3 (ms, molecule-local frame).
    sample_period_ms = 1000.0 / TDB_SAMPLE_RATE_HZ
    n_samples = pred.waveform.size
    t_ms = np.arange(n_samples, dtype=np.float64) * sample_period_ms

    # ---- Row 1: waveform ----
    wf = pred.waveform.astype(np.float64)
    # z-score for display
    if wf.std() > 0:
        wf_z = (wf - wf.mean()) / wf.std()
    else:
        wf_z = wf - wf.mean()
    ax_wave.plot(t_ms, wf_z, color="#555", linewidth=0.6)
    ax_wave.set_ylabel("current\n(z-score)", fontsize=9)
    ax_wave.set_title(
        f"uid {pred.uid} / ch {pred.channel} / {pred.run_id}  |  "
        f"{pred.n_tags_matched} matched tags  |  align_score {pred.alignment_score:,}  |  "
        f"2nd-best {pred.secondbest_score:,}  |  transloc {pred.translocation_time_ms:.1f} ms",
        fontsize=11,
    )
    ax_wave.grid(axis="x", alpha=0.2)

    # Translocation-region shading on waveform:
    # start_within_tdb_ms = (sample 0 of waveform is before molecule start)
    # For cached molecules, the waveform already starts at molecule start.
    # So the "active" translocation region is t_ms in [0, transloc_time].
    ax_wave.axvspan(0, pred.translocation_time_ms, color="#eef", alpha=0.35)

    # ---- Row 2: SP peaks on time axis ----
    ax_sp.set_ylim(-0.1, 1.2)
    ax_sp.set_yticks([])
    ax_sp.set_ylabel("SP peaks\n(probes.bin)", fontsize=9)
    # All detected probes (convert to cache-coord ms).
    sp_ms_cache = pred.sp_probe_center_ms + 0.0  # molecule-local ms (as in probes.bin)
    # sp_probe_center_ms IS molecule-local (measured from molecule start).
    # Ticks:
    for c, a in zip(sp_ms_cache, pred.sp_probe_accepted):
        color = COLOR_ACCEPT if a else COLOR_REJECT
        ax_sp.vlines(c, 0, 1.0, color=color, linewidth=1.8)
    ax_sp.hlines(0, 0, pred.translocation_time_ms, color="#bbb", linewidth=0.8)
    n_acc = int(pred.sp_probe_accepted.sum())
    n_rej = int((~pred.sp_probe_accepted).sum())
    ax_sp.text(
        0.99, 0.92, f"accepted: {n_acc}  rejected: {n_rej}",
        transform=ax_sp.transAxes, ha="right", va="top", fontsize=8,
    )

    # ---- Row 3: V3 heatmap + extracted peaks ----
    ax_v3.plot(t_ms, pred.v3_heatmap, color=COLOR_V3, linewidth=0.8, alpha=0.9)
    ax_v3.axhline(peak_threshold, color="#aaa", linewidth=0.6, linestyle=":")
    ax_v3.set_ylim(-0.02, max(1.05, float(pred.v3_heatmap.max()) * 1.05))
    ax_v3.set_ylabel("V3 probe\nheatmap", fontsize=9)
    peaks, _ = find_peaks(
        pred.v3_heatmap, height=peak_threshold, distance=peak_distance_samples,
    )
    for p_idx in peaks:
        t_peak = p_idx * sample_period_ms
        ax_v3.vlines(t_peak, 0, pred.v3_heatmap[p_idx], color=COLOR_V3, linewidth=1.4)
    ax_v3.text(
        0.99, 0.92, f"V3 NMS peaks: {len(peaks)}   "
                   f"(threshold {peak_threshold}, dist {peak_distance_samples} samples)",
        transform=ax_v3.transAxes, ha="right", va="top", fontsize=8,
    )
    ax_v3.set_xlabel("time (ms) -- molecule-local; shaded region is the translocation", fontsize=9)

    # Drop shared-axis xlabel clutter on top two rows.
    ax_wave.tick_params(labelbottom=False)
    ax_sp.tick_params(labelbottom=False)

    # ---- Row 4 & 5: genome-bp axis. ----
    for ax in (ax_t2d_g, ax_v3_g):
        ax.set_xlim(window_lo, window_hi)
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([])
        # Faint reference sites.
        for g in ref_sites_genome:
            ax.axvline(g, color=COLOR_REF_SITE, linewidth=0.6, alpha=0.6)
        # Actual aligner-assigned reference positions (red, emphasized).
        for g in pred.ref_genomic_bp:
            ax.vlines(g, 0, 1.0, color=COLOR_REF, linewidth=2.0)
        ax.hlines(
            0, window_lo, window_hi, color="#bbb", linewidth=0.8,
        )

    # T2D ticks (yellow) on genome.
    for g in t2d_genome:
        ax_t2d_g.vlines(g, 0, 1.0, color=COLOR_SP_T2D, linewidth=1.6)
    ax_t2d_g.set_ylabel("T2D on\ngenome bp", fontsize=9)
    ax_t2d_g.text(
        0.99, 0.92,
        f"yellow = Legacy T2D predictions (mapped via per-molecule affine).  "
        f"red = aligner's matched reference sites.  "
        f"gray = all reference sites in window ({len(ref_sites_genome)} sites).",
        transform=ax_t2d_g.transAxes, ha="right", va="top", fontsize=7, color="#444",
    )

    # V3 ticks (blue) on genome.
    for g in v3_genome:
        ax_v3_g.vlines(g, 0, 1.0, color=COLOR_V3, linewidth=1.6)
    ax_v3_g.set_ylabel("V3 on\ngenome bp", fontsize=9)
    ax_v3_g.set_xlabel(
        "genomic basepair (E. coli MG1655; red = aligner's matched sites; gray = all BssSI sites in window)",
        fontsize=9,
    )

    # Format genome-bp x-axis labels compactly.
    for ax in (ax_t2d_g, ax_v3_g):
        ax.ticklabel_format(style="plain", axis="x")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path
