"""Build a browsable molecule library: sorted index + per-molecule deep views.

For each holdout, rank aligned molecules by a composite confidence score
(primarily alignment score, with matched-ratio and second-best separation
as refinements), render the top-N deep-view PNGs, and emit an HTML
landing page that lets the user click through.

Usage (from worktree root):

    PYTHONPATH=src python scripts/build_molecule_library.py \\
        --checkpoint option_a_long_checkpoints/best_model.pt \\
        --cache-dir "...STB03-063B..." --transform-file "...STB03-063B_transForm.txt" \\
        --cache-dir "...STB03-064D..." --transform-file "...STB03-064D_transForm.txt" \\
        --cache-dir "...STB03-065H..." --transform-file "...STB03-065H_transForm.txt" \\
        --output-dir phase6_holdout_eval/molecule_library \\
        --top-n 25

Output layout:
    <output-dir>/
        index.html        -- landing page; 3 tabs, one per holdout
        <run_id>/<uid>.png -- per-molecule deep view
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from mongoose.analysis.molecule_viewer import (
    MoleculePredictions,
    compute_predictions,
    render_molecule_figure,
)
from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.io.assigns import MoleculeAssignment, load_assigns
from mongoose.io.probes_bin import Molecule as PbinMolecule
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.reference_map import load_reference_map
from mongoose.io.transform import load_transforms
from mongoose.model.unet import T2DUNet


@dataclass
class RankedMolecule:
    run_id: str
    uid: int
    manifest_idx: int
    alignment_score: int
    secondbest_score: int
    n_matched: int
    n_detected: int
    matched_ratio: float
    secondbest_ratio: float  # alignment_score / max(secondbest_score, 1)
    confidence: float


def _rank_molecules(
    *,
    cache_dir: Path,
    assigns_path: Path,
    min_tags: int,
    max_tags: int,
    run_id: str,
) -> list[RankedMolecule]:
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    mol_entries = manifest["molecules"]
    assigns_by_uid = {int(a.fragment_uid): a for a in load_assigns(assigns_path)}
    ranked: list[RankedMolecule] = []
    for idx, me in enumerate(mol_entries):
        uid = int(me["uid"])
        a = assigns_by_uid.get(uid)
        if a is None or a.ref_index < 0:
            continue
        n_matched = sum(1 for v in a.probe_indices if v > 0)
        n_detected = len(a.probe_indices)
        if n_matched < min_tags or n_matched > max_tags:
            continue
        matched_ratio = n_matched / max(n_detected, 1)
        sb_ratio = a.alignment_score / max(a.second_best_score, 1)
        # Composite: log(align_score) dominates; matched_ratio pushes "complete"
        # alignments up; sb_ratio rewards unambiguous region choice.
        import math
        confidence = (
            math.log(max(a.alignment_score, 1))
            + 0.5 * matched_ratio
            + 0.1 * min(sb_ratio, 10.0)
        )
        ranked.append(RankedMolecule(
            run_id=run_id,
            uid=uid,
            manifest_idx=idx,
            alignment_score=a.alignment_score,
            secondbest_score=a.second_best_score,
            n_matched=n_matched,
            n_detected=n_detected,
            matched_ratio=matched_ratio,
            secondbest_ratio=sb_ratio,
            confidence=confidence,
        ))
    ranked.sort(key=lambda r: r.confidence, reverse=True)
    return ranked


def _load_model(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = T2DUNet(
        config.in_channels, config.conditioning_dim,
        probe_aware_velocity=bool(getattr(config, "probe_aware_velocity", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, bool(getattr(config, "use_t2d_hybrid", False))


def _render_for_holdout(
    *,
    ranked: list[RankedMolecule],
    cache_dir: Path,
    transform_file: Path,
    model, device, use_hybrid: bool,
    output_dir: Path,
    run_id: str,
    top_n: int,
) -> list[tuple[RankedMolecule, Path]]:
    transforms = load_transforms(transform_file)
    pbin = load_probes_bin(next(transform_file.parent.glob("*_probes.bin")))
    mol_by_uid = {int(m.uid): m for m in pbin.molecules}
    asgn_path = None
    for p in transform_file.parent.glob("*probeassignment.assigns"):
        if ".subset." not in p.name and ".tvcsubset." not in p.name:
            asgn_path = p
            break
    asgns_by_uid = {int(a.fragment_uid): a for a in load_assigns(asgn_path)}
    refmap = load_reference_map(
        next(transform_file.parent.glob("*_referenceMap.txt"))
    )

    target = ranked[:top_n]
    target_by_idx = {r.manifest_idx: r for r in target}
    dataset = CachedMoleculeDataset([cache_dir], augment=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_molecules,
    )
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    mol_entries = manifest["molecules"]

    results: list[tuple[RankedMolecule, Path]] = []
    holdout_dir = output_dir / run_id
    holdout_dir.mkdir(parents=True, exist_ok=True)
    remaining = set(target_by_idx.keys())
    max_target_idx = max(target_by_idx.keys()) if target_by_idx else -1
    for idx, batch in enumerate(loader):
        if idx > max_target_idx:
            break
        if idx not in target_by_idx:
            continue
        me = mol_entries[idx]
        uid = int(me["uid"])
        pbin_mol = mol_by_uid[uid]
        assignment = asgns_by_uid[uid]
        transform = transforms[f"Ch{pbin_mol.channel:03d}"]
        pred = compute_predictions(
            batch=batch, manifest_entry=me, pbin_mol=pbin_mol,
            assignment=assignment, transform=transform,
            run_id=run_id, model=model, device=device, use_hybrid=use_hybrid,
        )
        png_path = holdout_dir / f"uid_{uid}.png"
        render_molecule_figure(pred, refmap=refmap, output_path=png_path)
        results.append((target_by_idx[idx], png_path))
        remaining.discard(idx)
        print(f"    [{run_id}] uid {uid} -> {png_path.name}  ({len(results)}/{len(target_by_idx)})")
        if not remaining:
            break
    # Preserve ranked order.
    results.sort(key=lambda r: target[target_by_idx and 0 or 0] is None)  # no-op placeholder
    ranked_order = {r.uid: i for i, r in enumerate(target)}
    results.sort(key=lambda rp: ranked_order.get(rp[0].uid, 10**9))
    return results


def _write_index_html(
    *,
    holdouts: list[tuple[str, list[tuple[RankedMolecule, Path]]]],
    output_dir: Path,
) -> Path:
    parts: list[str] = []
    parts.append("""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>Mongoose molecule library</title>
<style>
body { font-family: system-ui, -apple-system, sans-serif; margin: 20px; max-width: 1400px; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 36px; padding-bottom: 6px; border-bottom: 2px solid #ddd; }
.legend { background: #f6f7f9; border: 1px solid #ddd; padding: 10px 14px; margin: 12px 0 18px 0; border-radius: 6px; font-size: 13px; }
.legend ul { margin: 6px 0 0 20px; }
table { border-collapse: collapse; margin: 8px 0 18px 0; font-size: 13px; }
th, td { padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right; }
th { background: #f1f3f5; font-weight: 600; border-bottom: 2px solid #ccc; }
td.runid, th.runid { text-align: left; font-family: monospace; font-size: 11px; }
td.uid { text-align: right; font-family: monospace; }
tr:hover { background: #fafbfc; }
a { color: #1976d2; text-decoration: none; }
a:hover { text-decoration: underline; }
img { max-width: 100%; border: 1px solid #ccc; margin: 6px 0 18px 0; display: block; }
.anchor { scroll-margin-top: 12px; }
.toc { margin: 8px 0; }
</style></head><body>
""")
    parts.append("<h1>Mongoose molecule library</h1>")
    parts.append(
        "<p>Aligned molecules from the 3 holdouts, ranked by a composite "
        "confidence score (log alignment score × matched ratio × second-best "
        "separation). Click a row to jump to the molecule's 5-row deep view.</p>"
    )
    parts.append("""
<div class='legend'>
  <strong>Deep-view legend</strong>
  <ul>
    <li><b>Row 1</b>: raw waveform (z-scored current vs time). Shaded band is the translocation window.</li>
    <li><b>Row 2</b>: signal-processing peaks from probes.bin &mdash; <span style='color:#2ca02c'>green = accepted</span>, <span style='color:#888'>gray = rejected</span>.</li>
    <li><b>Row 3</b>: V3 model probe heatmap (<span style='color:#4285f4'>blue line</span>) with NMS-extracted peaks as ticks.</li>
    <li><b>Row 4</b>: Legacy T2D prediction (<span style='color:#f4c20d'>yellow ticks</span>) on the genome bp axis.</li>
    <li><b>Row 5</b>: V3 T2D prediction (<span style='color:#4285f4'>blue ticks</span>) on the same genome bp axis.</li>
    <li>On rows 4/5: <span style='color:#db4437'>red ticks</span> = aligner's per-probe assignment (what V3/T2D are compared against); <span style='color:#888'>gray ticks</span> = all BssSI reference sites in the window (physical ground truth from E. coli MG1655).</li>
  </ul>
  <em>Note:</em> the per-molecule local-bp &rarr; genome-bp mapping is an OLS fit on the matched probes. That fit uses the aligner's per-probe assignment, so the red ticks inherit the aligner's 250&ndash;500 bp wobble. The gray reference sites do <em>not</em> &mdash; they are positions of actual BssSI recognition sequences on the E. coli reference genome.
</div>
""")

    # Table of contents.
    parts.append("<h2>Contents</h2><div class='toc'><ul>")
    for run_id, _rows in holdouts:
        parts.append(f"<li><a href='#{html.escape(run_id)}'>{html.escape(run_id)}</a></li>")
    parts.append("</ul></div>")

    # One section per holdout.
    for run_id, rows in holdouts:
        parts.append(f"<h2 id='{html.escape(run_id)}'>{html.escape(run_id)} &mdash; top {len(rows)} molecules</h2>")
        # Ranking table.
        parts.append(
            "<table><thead><tr>"
            "<th>rank</th><th class='runid'>uid</th><th>matched</th><th>detected</th>"
            "<th>match ratio</th><th>align_score</th><th>2nd-best</th>"
            "<th>align/2nd</th><th>confidence</th>"
            "</tr></thead><tbody>"
        )
        for rank, (rm, png_path) in enumerate(rows, 1):
            anchor = f"{run_id}-uid-{rm.uid}"
            rel_png = png_path.relative_to(output_dir).as_posix()
            parts.append(
                f"<tr>"
                f"<td>{rank}</td>"
                f"<td class='uid'><a href='#{html.escape(anchor)}'>{rm.uid}</a></td>"
                f"<td>{rm.n_matched}</td>"
                f"<td>{rm.n_detected}</td>"
                f"<td>{rm.matched_ratio:.3f}</td>"
                f"<td>{rm.alignment_score:,}</td>"
                f"<td>{rm.secondbest_score:,}</td>"
                f"<td>{rm.secondbest_ratio:.2f}</td>"
                f"<td>{rm.confidence:.2f}</td>"
                f"</tr>"
            )
        parts.append("</tbody></table>")
        # Inline deep-view images.
        for rm, png_path in rows:
            anchor = f"{run_id}-uid-{rm.uid}"
            rel_png = png_path.relative_to(output_dir).as_posix()
            parts.append(
                f"<h3 id='{html.escape(anchor)}' class='anchor'>uid {rm.uid} "
                f"&nbsp;&middot;&nbsp; matched {rm.n_matched} / {rm.n_detected} "
                f"&nbsp;&middot;&nbsp; align_score {rm.alignment_score:,} "
                f"&nbsp;&middot;&nbsp; 2nd-best {rm.secondbest_score:,}</h3>"
            )
            parts.append(f"<img src='{html.escape(rel_png)}' alt='uid {rm.uid}'>")

    parts.append("</body></html>")
    index_path = output_dir / "index.html"
    index_path.write_text("\n".join(parts), encoding="utf-8")
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, action="append", required=True,
        help="Repeat per holdout, in order. Each must be paired with a --transform-file.",
    )
    parser.add_argument(
        "--transform-file", type=Path, action="append", required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--min-tags", type=int, default=5)
    parser.add_argument("--max-tags", type=int, default=20)
    args = parser.parse_args()
    if len(args.cache_dir) != len(args.transform_file):
        raise SystemExit("--cache-dir and --transform-file must have the same count")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, use_hybrid = _load_model(args.checkpoint, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdouts_out: list[tuple[str, list[tuple[RankedMolecule, Path]]]] = []
    for cache_dir, transform_file in zip(args.cache_dir, args.transform_file):
        run_id = cache_dir.name
        print(f"... {run_id}: ranking molecules")
        assigns_path = None
        for p in transform_file.parent.glob("*probeassignment.assigns"):
            if ".subset." not in p.name and ".tvcsubset." not in p.name:
                assigns_path = p
                break
        ranked = _rank_molecules(
            cache_dir=cache_dir, assigns_path=assigns_path,
            min_tags=args.min_tags, max_tags=args.max_tags, run_id=run_id,
        )
        print(f"    {len(ranked):,} eligible molecules; rendering top {args.top_n}")
        rows = _render_for_holdout(
            ranked=ranked, cache_dir=cache_dir, transform_file=transform_file,
            model=model, device=device, use_hybrid=use_hybrid,
            output_dir=args.output_dir, run_id=run_id, top_n=args.top_n,
        )
        holdouts_out.append((run_id, rows))
    index_path = _write_index_html(
        holdouts=holdouts_out, output_dir=args.output_dir,
    )
    print(f"\nwrote library index at: {index_path}")
    print(f"open in a browser: file:///{index_path.resolve().as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
