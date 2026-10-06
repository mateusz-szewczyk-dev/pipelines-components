"""Temporary storage isolation, cleanup, and diagnostics for training tasks."""

import logging
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from kfp_components.components.training.automl.shared.training_scratch import (
    log_scratch_usage,
    training_scratch,
)

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize("fail", [False, True])
def test_temporary_files_are_isolated_and_settings_restored(tmp_path, monkeypatch, caplog, fail):
    """Both success and failure remove temporary data and restore cached settings."""
    original_tmp = str(tmp_path / "original")
    monkeypatch.setenv("TMPDIR", original_tmp)
    monkeypatch.delenv("RAY_TMPDIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", original_tmp)
    root = tmp_path / "scratch"
    scratch_path = None

    def train():
        nonlocal scratch_path
        with training_scratch(LOGGER, root=root) as scratch_path:
            assert os.environ["TMPDIR"] == str(scratch_path)
            assert os.environ["RAY_TMPDIR"] == str(scratch_path)
            # gettempdir() must stop returning the previously cached directory.
            with tempfile.NamedTemporaryFile() as temporary_file:
                assert Path(temporary_file.name).parent == scratch_path
                temporary_file.write(b"temporary training data")
            ray_dir = scratch_path / "ray" / "session" / "logs"
            ray_dir.mkdir(parents=True)
            (ray_dir / "worker.log").write_text("training log")
            if fail:
                raise RuntimeError("training failed")

    with caplog.at_level(logging.INFO):
        if fail:
            with pytest.raises(RuntimeError, match="training failed"):
                train()
        else:
            train()

    assert scratch_path is not None and not scratch_path.exists()
    assert list(root.iterdir()) == []
    assert os.environ["TMPDIR"] == original_tmp
    assert "RAY_TMPDIR" not in os.environ
    assert tempfile.tempdir == original_tmp
    assert "Scratch storage (start)" in caplog.text
    assert "Scratch storage (before_cleanup)" in caplog.text


def test_existing_ray_temporary_setting_is_restored(tmp_path, monkeypatch):
    """A direct Python call preserves the caller's existing Ray configuration."""
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path / "original-ray"))
    monkeypatch.delenv("TMPDIR", raising=False)
    with training_scratch(LOGGER, root=tmp_path / "scratch"):
        pass
    assert os.environ["RAY_TMPDIR"] == str(tmp_path / "original-ray")
    assert "TMPDIR" not in os.environ


def test_cleanup_failure_does_not_mask_training_failure(tmp_path, caplog):
    """Cleanup errors are reported without replacing the original exception."""
    with (
        mock.patch("shutil.rmtree", side_effect=PermissionError("cannot remove")),
        pytest.raises(RuntimeError, match="training failed"),
        training_scratch(LOGGER, root=tmp_path / "scratch"),
    ):
        raise RuntimeError("training failed")
    assert "Could not remove training scratch" in caplog.text


def test_missing_scratch_diagnostic_is_best_effort(tmp_path, caplog):
    """A failed filesystem measurement does not abort training or cleanup."""
    log_scratch_usage(tmp_path / "missing", LOGGER, "after_selection")
    assert "Could not measure scratch storage" in caplog.text
