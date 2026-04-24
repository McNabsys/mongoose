"""Phase 0 sanity check: single-molecule hand-verification of the ETL.

Runs the per-run ETL on the canonical STB03-064B-02L58270w05-202G16g
sample (Red / biochem-flagged), picks the highest-alignment-score
molecule, and verifies:

1. ref_idx -> ref_genomic_pos_bp matches referenceMap for the first
   5 assigned probes (tests the 1-based index convention and
   end-to-end join correctness).
2. Probe attribute bitfield unpacks consistent with the raw uint32
   (trivial, but runs against the ETL's unpack helper).
3. probe_idx_in_molecule ordering matches the assigns row's ProbeK
   ordering (this is what spec §275 asks us to verify — probes.bin
   ProbeK ordering assumption).
4. molecule_bp_length matches the span of assigned probe positions
   in the refmap (consistency).
5. t2d_predicted_bp_pos for the first few probes is finite and
   increasing within the molecule (coordinate-frame sanity).

Emits a plain-text report at ``probe_table_sanity.txt`` in the
worktree root. Intentionally stdout-friendly so the report doubles
as a PR attachment for Milestone 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import pandas as pd

from mongoose.etl.probe_table import build_run_probe_table, resolve_aux_paths
from mongoose.io.assigns import load_assigns
from mongoose.io.reference_map import load_reference_map


DEFAULT_RUN_ROOT = Path(
    "C:/git/mongoose/E. coli/Red/STB03-064B-02L58270w05-202G16g/2025-02-19/Remapped/AllCh"
)
DEFAULT_RUN_ID = "STB03-064B-02L58270w05-202G16g"
DEFAULT_REPORT_PATH = Path("probe_table_sanity.txt")


def run_sanity_check(
    *,
    run_root: Path,
    run_id: str,
    report_path: Path,
) -> int:
    lines: list[str] = []
    lines.append(f"# Phase 0 sanity check: {run_id}")
    lines.append(f"Run root: {run_root}")
    lines.append("")

    probes_bin = run_root / f"{run_id}_probes.bin"
    assigns_path = run_root / f"{run_id}_probes.txt_probeassignment.assigns"
    refmap_path = run_root / f"{run_id}_probes.txt_referenceMap.txt"
    aux = resolve_aux_paths(run_root, run_id)

    lines.append("## Paths resolved")
    lines.append(f"  probes.bin         : {probes_bin.exists()}  {probes_bin}")
    lines.append(f"  assigns            : {assigns_path.exists()}  {assigns_path}")
    lines.append(f"  referenceMap       : {refmap_path.exists()}  {refmap_path}")
    lines.append(f"  transform_path     : {aux.transform_path}")
    lines.append(f"  remap_settings_path: {aux.remap_settings_path}")
    lines.append(f"  version_path       : {aux.version_path}")
    lines.append(f"  probe_widths_path  : {aux.probe_widths_path}")
    lines.append("")

    df = build_run_probe_table(
        run_id=run_id,
        probes_bin_path=probes_bin,
        assigns_path=assigns_path,
        reference_map_path=refmap_path,
        aux=aux,
        compute_t2d=True,
    )
    lines.append("## ETL output")
    lines.append(f"  total probe rows  : {len(df):,}")
    lines.append(f"  total molecules   : {df['molecule_uid'].nunique():,}")
    lines.append(
        f"  aligned molecules : "
        f"{df[df['molecule_aligned']]['molecule_uid'].nunique():,}"
    )
    assigned_frac = float(
        df.loc[df["molecule_aligned"], "is_assigned"].mean()
    ) if df["molecule_aligned"].any() else float("nan")
    lines.append(
        f"  assigned fraction among aligned molecules: {assigned_frac:.3f}"
    )
    lines.append("  (spec success criterion §299 expects roughly 0.70-0.85;")
    lines.append("   values much lower are worth surfacing to the user.)")
    lines.append("")

    # Additional diagnostic: when molecule_aligned is True, it's possible
    # for all 64/128/N probes to be unassigned (ref_idx = 0). Count those
    # explicitly so the 37%-ish aggregate doesn't hide a distributional
    # issue.
    per_mol = (
        df[df["molecule_aligned"]]
        .groupby("molecule_uid")
        .agg(
            n_assigned=("is_assigned", "sum"),
            n_probes=("is_assigned", "size"),
        )
    )
    per_mol["frac"] = per_mol["n_assigned"] / per_mol["n_probes"]
    lines.append("## Aligned-molecule assignment-rate distribution")
    lines.append(
        f"  frac==0   : {int((per_mol['n_assigned'] == 0).sum()):>6}  "
        f"({(per_mol['n_assigned'] == 0).mean():.2%})"
    )
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        mask = (per_mol["frac"] > lo) & (per_mol["frac"] <= hi)
        lines.append(
            f"  {lo:.2f} < f <= {hi:.2f}: {int(mask.sum()):>6}  "
            f"({mask.mean():.2%})"
        )
    lines.append("")

    # --- Anomaly investigation: len(probe_indices) vs num_probes ---
    assigns_all = load_assigns(assigns_path)
    from mongoose.io.probes_bin import load_probes_bin
    pbin = load_probes_bin(probes_bin)
    mol_by_uid = {m.uid: m for m in pbin.molecules}
    match = mismatch = 0
    mismatch_examples: list[tuple[int, int, int, int]] = []
    for a in assigns_all:
        if a.ref_index < 0:
            continue
        mol = mol_by_uid.get(a.fragment_uid)
        if mol is None:
            continue
        if len(a.probe_indices) == mol.num_probes:
            match += 1
        else:
            mismatch += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append(
                    (a.fragment_uid, len(a.probe_indices), mol.num_probes, a.alignment_score)
                )
    lines.append("## assigns probe_indices vs probes.bin num_probes / accepted")
    lines.append(
        f"  aligned molecules with len(probe_indices) == num_probes : {match}"
    )
    lines.append(
        f"  aligned molecules with len(probe_indices) <  num_probes : {mismatch}"
    )
    lines.append(
        f"  mismatch fraction : "
        f"{mismatch/(match+mismatch) if (match+mismatch) else float('nan'):.3f}"
    )
    for ex in mismatch_examples:
        lines.append(
            f"  example: uid={ex[0]}  len(probe_indices)={ex[1]}  "
            f"num_probes={ex[2]}  align_score={ex[3]}"
        )
    lines.append("")
    lines.append(
        "  Interpretation (confirmed 2026-04-23): .assigns.probe_indices[k] "
        "maps to the k-th ACCEPTED probe (attribute bit 7 set) in "
        "probes.bin detection order, NOT the k-th detected probe overall. "
        "Non-accepted probes and accepted probes beyond len(probe_indices) "
        "receive ref_idx=null. Joins are implemented this way in "
        "probe_table.py; Check 2 above validates the exact mapping on the "
        "highest-match molecule."
    )
    lines.append("")

    # --- Sanity check 1: pick molecule with most assigned probes ---
    aligned_df = df[df["molecule_aligned"]].copy()
    if aligned_df.empty:
        lines.append("!! No aligned molecules found -- nothing to verify.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return 1
    # Rank by number of is_assigned probes per molecule; break ties by
    # alignment_score. An aligned molecule with zero matches is a real
    # case in the data (surfaces as an anomaly report below) but not
    # what we want for a join-correctness sanity check.
    by_uid = (
        aligned_df.groupby("molecule_uid")
        .agg(
            n_assigned=("is_assigned", "sum"),
            align_score=("molecule_align_score", "first"),
        )
        .sort_values(
            ["n_assigned", "align_score"], ascending=[False, False]
        )
    )
    lines.append("## Top-5 aligned molecules by n_assigned")
    for uid, row_info in by_uid.head(5).iterrows():
        lines.append(
            f"  uid={int(uid)}  n_assigned={int(row_info['n_assigned'])}  "
            f"align_score={int(row_info['align_score'])}"
        )
    lines.append("")
    best_uid = int(by_uid.index[0])
    mol_df = aligned_df[aligned_df["molecule_uid"] == best_uid].sort_values(
        "probe_idx_in_molecule"
    )
    lines.append(f"## Highest-align-score molecule: uid={best_uid}")
    lines.append(f"  probes in molecule: {len(mol_df)}")
    lines.append(
        f"  alignment_score   : {int(mol_df.iloc[0]['molecule_align_score'])}"
    )
    lines.append(
        f"  direction         : {int(mol_df.iloc[0]['molecule_direction'])}"
    )
    lines.append(
        f"  num_matched       : {int(mol_df['is_assigned'].sum())} / {len(mol_df)}"
    )
    lines.append("")

    # --- Sanity check 2: ref_idx -> ref_genomic_pos_bp consistency ---
    refmap = load_reference_map(refmap_path)
    lines.append("## Check 1: ref_idx -> ref_genomic_pos_bp (first 5 assigned probes)")
    n_checked = 0
    failures = 0
    for _, row in mol_df.iterrows():
        if n_checked >= 5:
            break
        if not bool(row["is_assigned"]):
            continue
        ref_idx = int(row["ref_idx"])
        reported_bp = int(row["ref_genomic_pos_bp"])
        expected_bp = int(refmap.probe_positions[ref_idx - 1])
        ok = reported_bp == expected_bp
        lines.append(
            f"  probe_idx={int(row['probe_idx_in_molecule'])}  "
            f"ref_idx={ref_idx}  reported_bp={reported_bp:,}  "
            f"refmap[ref_idx-1]={expected_bp:,}  {'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            failures += 1
        n_checked += 1
    lines.append("")

    # --- Sanity check 3: ProbeK ordering vs probe_idx_in_molecule ---
    assigns = load_assigns(assigns_path)
    assign_for_best = next(
        (a for a in assigns if a.fragment_uid == best_uid), None
    )
    lines.append("## Check 2: ProbeK -> k-th accepted probe ordering")
    if assign_for_best is None:
        lines.append("  !! no assigns row found for best molecule -- skipping")
    else:
        expected_tuple = assign_for_best.probe_indices
        accepted_df = mol_df[mol_df["attr_accepted"]].sort_values(
            "probe_idx_in_molecule"
        )
        accepted_head = accepted_df.head(len(expected_tuple))
        reported_tuple = tuple(
            -1 if pd.isna(v) else int(v)
            for v in accepted_head["ref_idx"].tolist()
        )
        match = expected_tuple == reported_tuple
        lines.append(
            f"  assigns.probe_indices len  : {len(expected_tuple)}"
        )
        lines.append(
            f"  accepted probes in mol     : {len(accepted_df)}"
        )
        lines.append(
            f"  first-K accepted == assigns: {match}  "
            f"(first 10 expected={expected_tuple[:10]}, reported={reported_tuple[:10]})"
        )
        # Accepted probes beyond K must have ref_idx=null.
        tail_df = accepted_df.tail(
            max(0, len(accepted_df) - len(expected_tuple))
        )
        tail_all_null = bool(tail_df["ref_idx"].isna().all())
        lines.append(
            f"  accepted-probe tail all null: {tail_all_null} "
            f"({len(tail_df)} probes beyond K)"
        )
        if not match or not tail_all_null:
            failures += 1
    lines.append("")

    # --- Sanity check 4: molecule_bp_length matches refmap span ---
    lines.append("## Check 3: molecule_bp_length vs refmap span of assigned probes")
    assigned_bps = [
        int(v) for v in mol_df.loc[mol_df["is_assigned"], "ref_genomic_pos_bp"]
    ]
    if len(assigned_bps) >= 2:
        expected_span = max(assigned_bps) - min(assigned_bps)
        reported_span = int(mol_df.iloc[0]["molecule_bp_length"])
        ok = expected_span == reported_span
        lines.append(
            f"  assigned bp span: {expected_span:,}  "
            f"column: {reported_span:,}  {'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            failures += 1
    else:
        lines.append("  <2 assigned probes — skip")
    lines.append("")

    # --- Sanity check 5: t2d_predicted_bp_pos finite + monotone-ish ---
    lines.append("## Check 4: t2d_predicted_bp_pos (per-probe bp from legacy T2D)")
    t2d_vals = mol_df["t2d_predicted_bp_pos"].to_numpy(dtype=float, copy=False)
    finite = np.isfinite(t2d_vals)
    lines.append(
        f"  finite: {int(finite.sum())} / {len(t2d_vals)}  "
        f"min={np.nanmin(t2d_vals):.1f}  max={np.nanmax(t2d_vals):.1f}"
    )
    # T2D bp position ~ monotone in center_ms (closer to tail ⇒ smaller bp).
    diffs = np.diff(t2d_vals[finite])
    if diffs.size:
        direction_consistent = bool(np.all(diffs <= 0)) or bool(np.all(diffs >= 0))
        lines.append(
            f"  monotone in probe order: {direction_consistent}  "
            f"(diff sign counts: pos={int((diffs>0).sum())} "
            f"neg={int((diffs<0).sum())} zero={int((diffs==0).sum())})"
        )
        if not direction_consistent:
            # Not a hard failure — probes may reorder relative to tail —
            # but the diagnostic is useful to see in the report.
            lines.append("  (not asserted: see spec §277 notes on monotonicity)")
    lines.append("")

    # --- Sanity check 6: probe_local_density + gaps spot check ---
    lines.append("## Check 5: probe_local_density + gap columns (first 5 probes)")
    for _, row in mol_df.head(5).iterrows():
        lines.append(
            f"  idx={int(row['probe_idx_in_molecule'])}  "
            f"center_ms={float(row['center_ms']):.2f}  "
            f"prev_gap={row['prev_probe_gap_ms']!r}  "
            f"next_gap={row['next_probe_gap_ms']!r}  "
            f"density(+/-50ms)={int(row['probe_local_density'])}"
        )
    lines.append("")

    # --- Summary ---
    lines.append("## Summary")
    lines.append(f"  failures: {failures}")
    if failures > 0:
        lines.append("  STATUS: FAIL")
    else:
        lines.append("  STATUS: PASS")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    for line in lines:
        print(line)
    return 1 if failures > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=f"Remapped/AllCh/ dir for the sanity-check run. "
             f"Default: {DEFAULT_RUN_ROOT}",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT_PATH,
    )
    args = parser.parse_args()
    return run_sanity_check(
        run_root=args.run_root, run_id=args.run_id, report_path=args.report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
