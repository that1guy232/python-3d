"""Atomic file primitives and shared in-process locks for persistence.

Replacement is atomic for one file; these helpers do not provide a transaction
across files or cross-process writer exclusion. File fsync does not promise
portable directory-entry durability through every power failure.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from threading import Lock, RLock


_locks: dict[Path, RLock] = {}
_locks_guard = Lock()


def normalize_path(path: str | Path) -> Path:
    """Return the same expanded, resolved, case-normalized path for IO and locks."""

    resolved = Path(path).expanduser().resolve()
    return Path(os.path.normcase(str(resolved)))


def path_lock(path: str | Path) -> RLock:
    """Get the shared reentrant lock for a canonical primary path."""

    normalized = normalize_path(path)
    with _locks_guard:
        lock = _locks.get(normalized)
        if lock is None:
            lock = RLock()
            _locks[normalized] = lock
        return lock


def cleanup_temporary(path: Path | None) -> OSError | None:
    """Best-effort removal; return cleanup trouble without masking a commit."""

    if path is None:
        return None
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        return error
    return None


def _close_after_error(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass


def write_temporary(path: Path, data: bytes) -> Path:
    """Write and fsync a unique sibling; the destination parent must exist.

    On failure, remove the partial file when possible and preserve the original
    exception. The caller owns the returned file until replacement or cleanup.
    """

    if type(data) is not bytes:
        raise TypeError("data must be bytes")
    target = normalize_path(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name[:48]}-", suffix=".tmp", dir=target.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        _close_after_error(descriptor)
        cleanup_temporary(temporary)
        raise
    return temporary


def archive_file(path: Path) -> Path:
    """Stream an entire existing file to a unique, fsynced sibling archive.

    Archival is independent of JSON byte limits. A missing source raises
    FileNotFoundError without creating an archive. The caller serializes this
    operation with any managed replacements of the source.
    """

    source = normalize_path(path)
    descriptor = None
    archive = None
    try:
        with source.open("rb") as original:
            descriptor, name = tempfile.mkstemp(
                prefix=f"{source.stem[:48]}.archive-",
                suffix=source.suffix or ".save",
                dir=source.parent,
            )
            archive = Path(name)
            with os.fdopen(descriptor, "wb") as preserved:
                descriptor = None
                shutil.copyfileobj(original, preserved, length=1024 * 1024)
                preserved.flush()
                os.fsync(preserved.fileno())
    except BaseException:
        _close_after_error(descriptor)
        cleanup_temporary(archive)
        raise
    assert archive is not None
    return archive


def atomic_write_bytes(path: str | Path, data: bytes) -> OSError | None:
    """Replace one file, returning only a possible post-commit cleanup warning.

    Pre-commit failures raise and leave the previous destination intact.
    Successful os.replace is the commit point, even if subsequent temporary
    cleanup fails. This helper does not rotate backups or validate documents.
    """

    if type(data) is not bytes:
        raise TypeError("data must be bytes")
    target = normalize_path(path)
    with path_lock(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = write_temporary(target, data)
        try:
            os.replace(temporary, target)
        except BaseException:
            cleanup_temporary(temporary)
            raise
        return cleanup_temporary(temporary)


__all__ = [
    "normalize_path", "path_lock", "write_temporary", "archive_file",
    "cleanup_temporary", "atomic_write_bytes",
]
