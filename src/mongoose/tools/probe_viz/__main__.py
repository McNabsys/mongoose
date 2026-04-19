"""CLI entry point: python -m mongoose.tools.probe_viz <sample_dir>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


KEY_HELP = """\
Keys:
  left / right           prev / next molecule
  shift+left / right     jump by 10
  pageup / pagedown      jump by 100
  home / end             first / last
  g                      goto prompt (UID=n or ch:mid)
  a                      toggle excluded probes
  s                      toggle structured regions
  e                      toggle TDB edge markers
  u                      toggle do_not_use molecules
  r                      reset axis zoom
  q / esc                quit
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mongoose.tools.probe_viz",
        description="Visualize probe centers overlaid on TDB waveforms.",
        epilog=KEY_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path, help="Sample-run directory.")
    parser.add_argument(
        "--include-do-not-use", action="store_true",
        help="Include molecules with do_not_use=True in the iteration list.",
    )
    parser.add_argument(
        "--start-at", type=int, default=None,
        help="Jump to this molecule UID on startup.",
    )
    args = parser.parse_args(argv)

    try:
        from mongoose.tools.probe_viz.loader import ProbeVizLoader
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        from mongoose.tools.probe_viz.viewer import ProbeVizViewer
    except ImportError as exc:
        print(
            f"error: matplotlib required. Install with: pip install -e '.[dev]'\n"
            f"  underlying import error: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        loader = ProbeVizLoader(
            args.sample_dir, include_do_not_use=args.include_do_not_use
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.start_at is not None:
        if not loader.goto_uid(args.start_at):
            print(
                f"warning: UID {args.start_at} not in iteration list; "
                f"starting at index 0",
                file=sys.stderr,
            )

    print(KEY_HELP)
    viewer = ProbeVizViewer(loader)
    viewer.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
