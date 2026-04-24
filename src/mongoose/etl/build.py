"""Phase 0 ETL orchestrator: Excel manifest -> 30 probe-table shards.

Discovers runs from the Excel manifest, resolves each run's raw file
paths on disk (using the concentration_group -> color-folder mapping),
runs :func:`build_run_probe_table` per run in a process pool, and
writes:

  - ``<output_dir>/probe_table/<run_id>.parquet`` — one shard per run
  - ``<output_dir>/probe_table_manifest.json`` — provenance + counts
  - ``<output_dir>/probe_table.parquet`` (optional) — merged single file

The merged file is optional because it roughly doubles disk usage and
is trivial to rebuild from the shard directory via
``pyarrow.dataset.dataset(<shards>)`` or ``pd.read_parquet(<dir>)``.
"""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mongoose.etl.excel_manifest import RunMetadata, load_excel_manifest
from mongoose.etl.probe_table import build_run_probe_table, resolve_aux_paths
from mongoose.etl.schema import SCHEMA_VERSION


# concentration_group -> on-disk color folder. Verified 2026-04-23 against
# the Run IDs present in each E. coli/{Black,Red,Blue} directory.
GROUP_TO_COLOR_FOLDER: dict[str, str] = {
    "std": "Black",
    "low": "Red",
    "low_dil": "Blue",
}


@dataclass
class RunIngestResult:
    run_id: str
    status: str          # "ok" / "skipped" / "failed"
    probe_count: int = 0
    molecule_count: int = 0
    aligned_molecule_count: int = 0
    is_assigned_rate: float = 0.0  # fraction of probes in aligned mols with ref_idx > 0
    shard_path: str | None = None
    resolved_root: str | None = None
    elapsed_sec: float = 0.0
    error: str | None = None


def resolve_run_date_dir(data_root: Path, run: RunMetadata) -> Path | None:
    """Locate the <date> directory on disk for a given run.

    Layout: ``<data_root>/<color>/<run_id>/<YYYY-MM-DD>/``. We know color
    from the concentration_group; we discover the date by globbing since
    the Excel's ``Date`` column is a datetime but disk uses YYYY-MM-DD.
    Returns None if the run folder doesn't exist on disk (e.g., the raw
    files weren't copied locally yet).
    """
    color = GROUP_TO_COLOR_FOLDER.get(run.concentration_group)
    if color is None:
        return None
    run_dir = Path(data_root) / color / run.run_id
    if not run_dir.is_dir():
        return None
    # Prefer the Excel-declared date when present, fall back to any single
    # date subdir. If multiple date subdirs exist, that's a data layout
    # anomaly worth surfacing.
    expected_date: str | None = None
    date_value = run.fields.get("Date")
    if hasattr(date_value, "strftime"):
        expected_date = date_value.strftime("%Y-%m-%d")  # type: ignore[attr-defined]
    if expected_date:
        candidate = run_dir / expected_date
        if candidate.is_dir():
            return candidate
    subdirs = [p for p in run_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return None


def _ingest_one_run(
    run: RunMetadata,
    data_root: str,
    output_dir: str,
    compute_t2d: bool,
) -> RunIngestResult:
    """Worker: build one run's probe-table shard. Picklable for ProcessPoolExecutor."""
    t0 = time.monotonic()
    try:
        date_dir = resolve_run_date_dir(Path(data_root), run)
        if date_dir is None:
            return RunIngestResult(
                run_id=run.run_id,
                status="skipped",
                error=(
                    f"data folder not found under "
                    f"{data_root}/{GROUP_TO_COLOR_FOLDER.get(run.concentration_group)}/"
                    f"{run.run_id}/<YYYY-MM-DD>/"
                ),
                elapsed_sec=time.monotonic() - t0,
            )
        remap_allch = date_dir / "Remapped" / "AllCh"
        if not remap_allch.is_dir():
            remap_allch = date_dir / "Remapping" / "AllCh"
        probes_bin = remap_allch / f"{run.run_id}_probes.bin"
        assigns = remap_allch / f"{run.run_id}_probes.txt_probeassignment.assigns"
        refmap = remap_allch / f"{run.run_id}_probes.txt_referenceMap.txt"
        for required in (probes_bin, assigns, refmap):
            if not required.exists():
                return RunIngestResult(
                    run_id=run.run_id,
                    status="skipped",
                    resolved_root=str(remap_allch),
                    error=f"missing required input: {required.name}",
                    elapsed_sec=time.monotonic() - t0,
                )
        aux = resolve_aux_paths(remap_allch, run.run_id)

        df = build_run_probe_table(
            run_id=run.run_id,
            probes_bin_path=probes_bin,
            assigns_path=assigns,
            reference_map_path=refmap,
            aux=aux,
            compute_t2d=compute_t2d,
        )
        # Attach Excel metadata as broadcast columns. Column names come
        # straight from the Excel header; downstream diagnostic code
        # references them as ``df["% filtered remapped"]`` etc.
        df["concentration_group"] = run.concentration_group
        df["conc_raw"] = run.conc_raw
        df["biochem_flagged_good"] = run.biochem_flagged_good
        for header, value in run.fields.items():
            # Skip Run ID — already present as df["run_id"].
            if header == "Run ID":
                continue
            df[f"excel_{_sanitize_col(header)}"] = value

        shard_path = Path(output_dir) / "probe_table" / f"{run.run_id}.parquet"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(shard_path, index=False)

        aligned = df[df["molecule_aligned"]]
        return RunIngestResult(
            run_id=run.run_id,
            status="ok",
            probe_count=int(len(df)),
            molecule_count=int(df["molecule_uid"].nunique()),
            aligned_molecule_count=int(aligned["molecule_uid"].nunique()),
            is_assigned_rate=float(aligned["is_assigned"].mean()) if len(aligned) else 0.0,
            shard_path=str(shard_path),
            resolved_root=str(remap_allch),
            elapsed_sec=time.monotonic() - t0,
        )
    except Exception as exc:
        return RunIngestResult(
            run_id=run.run_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            elapsed_sec=time.monotonic() - t0,
        )


def _sanitize_col(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("%", "pct")
        .replace(">", "gt")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_per_")
        .replace("-", "_")
    )


def _current_git_hash(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def build_probe_table(
    *,
    excel_path: Path,
    data_root: Path,
    output_dir: Path,
    compute_t2d: bool = True,
    workers: int = 4,
    run_subset: list[str] | None = None,
    write_merged: bool = True,
) -> dict[str, Any]:
    """Run the full Phase 0 ETL end-to-end. Returns the manifest dict."""
    t0 = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_excel_manifest(excel_path)
    if run_subset is not None:
        wanted = set(run_subset)
        runs = [r for r in runs if r.run_id in wanted]
        missing = wanted - {r.run_id for r in runs}
        if missing:
            raise ValueError(f"unknown run IDs in --runs: {sorted(missing)}")

    print(f"Ingesting {len(runs)} run(s) with {workers} worker(s)...")

    results: list[RunIngestResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _ingest_one_run, run, str(data_root), str(output_dir), compute_t2d,
            ): run.run_id
            for run in runs
        }
        for fut in as_completed(futures):
            result = fut.result()
            status_tag = (
                "OK"
                if result.status == "ok"
                else "SKIP" if result.status == "skipped" else "FAIL"
            )
            print(
                f"  [{status_tag}] {result.run_id:>35}  "
                f"probes={result.probe_count:>8,}  "
                f"mols={result.molecule_count:>6,}  "
                f"aligned={result.aligned_molecule_count:>6,}  "
                f"assigned_rate={result.is_assigned_rate:.3f}  "
                f"{result.elapsed_sec:.1f}s"
                + (f"  [{result.error.splitlines()[0][:80]}]" if result.error else "")
            )
            results.append(result)

    results.sort(key=lambda r: r.run_id)
    ok_results = [r for r in results if r.status == "ok"]

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _current_git_hash(Path(__file__).resolve().parent),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "excel_path": str(excel_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "compute_t2d": compute_t2d,
        "n_runs_requested": len(runs),
        "n_runs_ingested": len(ok_results),
        "n_runs_skipped": sum(1 for r in results if r.status == "skipped"),
        "n_runs_failed": sum(1 for r in results if r.status == "failed"),
        "total_probe_count": sum(r.probe_count for r in ok_results),
        "total_molecule_count": sum(r.molecule_count for r in ok_results),
        "biochem_flagged_count": sum(1 for r in runs if r.biochem_flagged_good),
        "biochem_flagged_run_ids": sorted(
            r.run_id for r in runs if r.biochem_flagged_good
        ),
        "runs": [asdict(r) for r in results],
        "wall_time_sec": time.monotonic() - t0,
    }

    manifest_path = output_dir / "probe_table_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    if write_merged and ok_results:
        # Stream shard->merged via pyarrow to avoid holding all 30 runs
        # in memory at once (each shard can be ~1 GB; 30x that OOMs).
        # Pandas-inferred Excel column dtypes can drift across shards
        # (e.g., a numeric column that's "-" in one run); unify by
        # promoting any column whose type disagrees to large_string
        # before writing the merged file.
        import pyarrow as pa
        import pyarrow.parquet as pq

        merged_path = output_dir / "probe_table.parquet"
        shard_dir = output_dir / "probe_table"
        shards = sorted(shard_dir.glob("*.parquet"))
        print(f"\nMerging {len(shards)} shards to {merged_path} ...")

        # Pass 1: collect per-column dtype across shards, pick a unified type.
        types_by_col: dict[str, set[str]] = {}
        field_order: list[str] = []
        for s in shards:
            sch = pq.read_schema(s)
            for field in sch:
                if field.name not in types_by_col:
                    types_by_col[field.name] = set()
                    field_order.append(field.name)
                types_by_col[field.name].add(str(field.type))

        unified_fields: list[pa.Field] = []
        coerced_cols: list[str] = []
        for name in field_order:
            types = types_by_col[name]
            if len(types) == 1:
                # Single type across all shards -- use it as-is. Re-read
                # from the first shard that has it to get the real pa.DataType.
                first_shard = next(
                    s for s in shards if name in [f.name for f in pq.read_schema(s)]
                )
                unified_fields.append(pq.read_schema(first_shard).field(name))
            else:
                unified_fields.append(pa.field(name, pa.large_string()))
                coerced_cols.append(name)
        unified_schema = pa.schema(unified_fields)
        if coerced_cols:
            print(
                f"  coerced to large_string due to cross-shard type drift: "
                f"{coerced_cols}"
            )

        total_rows = 0
        writer = pq.ParquetWriter(merged_path, unified_schema)
        try:
            for s in shards:
                table = pq.read_table(s)
                # Cast each coerced column to string explicitly (null-safe).
                for col in coerced_cols:
                    if col in table.column_names:
                        table = table.set_column(
                            table.column_names.index(col),
                            col,
                            pa.compute.cast(
                                table.column(col), pa.large_string(), safe=False
                            ),
                        )
                # Re-order / backfill columns to match unified schema.
                aligned: list[pa.ChunkedArray] = []
                for field in unified_schema:
                    if field.name in table.column_names:
                        arr = table.column(field.name)
                        if arr.type != field.type:
                            arr = pa.compute.cast(arr, field.type, safe=False)
                        aligned.append(arr)
                    else:
                        aligned.append(
                            pa.nulls(table.num_rows, type=field.type)
                        )
                shard_table = pa.table(aligned, schema=unified_schema)
                writer.write_table(shard_table)
                total_rows += shard_table.num_rows
        finally:
            writer.close()
        print(
            f"  merged rows: {total_rows:,}  "
            f"file size: {merged_path.stat().st_size / 1e6:.1f} MB"
        )

    return manifest
