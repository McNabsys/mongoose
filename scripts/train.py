"""Train the T2D U-Net model.

Thin wrapper around ``mongoose.training.cli.main``. All argument parsing
and config-building logic lives in the package so it can be unit-tested.
"""

from __future__ import annotations

from mongoose.training.cli import main

if __name__ == "__main__":
    main()
