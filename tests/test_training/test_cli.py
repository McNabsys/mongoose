"""Tests for the training CLI argument parser and config builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from mongoose.training.cli import build_arg_parser, config_from_args


def test_parser_requires_synthetic_or_cache_dir():
    """Omitting both --synthetic and --cache-dir is a usage error."""
    parser = build_arg_parser()
    args = parser.parse_args([])
    with pytest.raises(SystemExit):
        config_from_args(args)


def test_parser_synthetic_path_builds_config():
    parser = build_arg_parser()
    args = parser.parse_args(["--synthetic", "--epochs", "2", "--batch-size", "4"])
    config = config_from_args(args)
    assert config.use_synthetic is True
    assert config.cache_dirs is None
    assert config.epochs == 2
    assert config.batch_size == 4


def test_parser_cache_dir_single_builds_config(tmp_path):
    cache_dir = tmp_path / "run_a"
    cache_dir.mkdir()
    parser = build_arg_parser()
    args = parser.parse_args(["--cache-dir", str(cache_dir), "--epochs", "3"])
    config = config_from_args(args)
    assert config.use_synthetic is False
    assert config.cache_dirs == [cache_dir]
    assert config.epochs == 3


def test_parser_cache_dir_repeatable(tmp_path):
    """--cache-dir should accept multiple occurrences for multi-run training."""
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--cache-dir", str(run_a), "--cache-dir", str(run_b)]
    )
    config = config_from_args(args)
    assert config.cache_dirs == [run_a, run_b]


def test_parser_rejects_synthetic_with_cache_dir(tmp_path):
    cache_dir = tmp_path / "run_a"
    cache_dir.mkdir()
    parser = build_arg_parser()
    args = parser.parse_args(["--synthetic", "--cache-dir", str(cache_dir)])
    with pytest.raises(SystemExit):
        config_from_args(args)


def test_parser_max_molecules_flag(tmp_path):
    cache_dir = tmp_path / "run_a"
    cache_dir.mkdir()
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--cache-dir", str(cache_dir), "--max-molecules", "200"]
    )
    config = config_from_args(args)
    assert config.max_molecules == 200


def test_parser_passes_warmstart_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--synthetic",
            "--warmstart-epochs",
            "3",
            "--warmstart-fade-epochs",
            "2",
        ]
    )
    config = config_from_args(args)
    assert config.warmstart_epochs == 3
    assert config.warmstart_fade_epochs == 2


def test_parser_no_amp_flag():
    parser = build_arg_parser()
    args = parser.parse_args(["--synthetic", "--no-amp"])
    config = config_from_args(args)
    assert config.use_amp is False


def test_parser_rejects_nonexistent_cache_dir(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args(["--cache-dir", str(tmp_path / "does_not_exist")])
    with pytest.raises(SystemExit):
        config_from_args(args)


def test_parser_checkpoint_dir_as_path(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--synthetic", "--checkpoint-dir", str(tmp_path / "ckpts")]
    )
    config = config_from_args(args)
    assert isinstance(config.checkpoint_dir, Path)
    assert config.checkpoint_dir == tmp_path / "ckpts"
