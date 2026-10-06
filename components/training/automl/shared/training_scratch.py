"""Task-local temporary storage and disk diagnostics for AutoGluon training."""

import logging
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def log_scratch_usage(path: Path, logger: logging.Logger, stage: str) -> None:
    """Report filesystem capacity without masking a training error."""
    try:
        usage = shutil.disk_usage(path)
        logger.info(
            "Scratch storage (%s): path=%s total_bytes=%d used_bytes=%d free_bytes=%d",
            stage,
            path,
            usage.total,
            usage.used,
            usage.free,
        )
    except OSError:
        logger.warning("Could not measure scratch storage at %s", path, exc_info=True)


@contextmanager
def training_scratch(logger: logging.Logger, root: Path = Path("/tmp/autogluon-scratch")) -> Iterator[Path]:
    """Route Python and Ray temporary files into scratch and clean up on exit.

    Each training task runs in its own process. Restore process-wide temporary
    settings for direct Python calls, including tempfile's cached directory.
    """
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if stat.S_ISLNK(root.lstat().st_mode):
        raise PermissionError(f"Unsafe scratch directory: {root}")
    # Use a short prefix for Ray's Unix-domain socket paths.
    path = Path(tempfile.mkdtemp(prefix="ag-", dir=root))
    previous_env = {name: os.environ.get(name) for name in ("TMPDIR", "RAY_TMPDIR")}
    previous_tempdir = tempfile.tempdir
    try:
        os.environ["TMPDIR"] = str(path)
        os.environ["RAY_TMPDIR"] = str(path)
        tempfile.tempdir = str(path)
        log_scratch_usage(path, logger, "start")
        yield path
    finally:
        log_scratch_usage(path, logger, "before_cleanup")
        tempfile.tempdir = previous_tempdir
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        try:
            shutil.rmtree(path)
        except OSError:
            logger.warning("Could not remove training scratch at %s", path, exc_info=True)
