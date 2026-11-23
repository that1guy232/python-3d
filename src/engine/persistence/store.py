"""Codec-driven JSON snapshots, previous-good backups, and explicit resets.

The store is synchronous and has no opinion about save slots or scheduling.
Callers own snapshot capture and use one logical writer per path. Locks cover
cooperating stores in this process, not competing processes or stale snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Generic, Literal, Protocol, TypeVar

from . import atomic
from .json_io import (
    DocumentError,
    JsonLimits,
    JsonOptions,
    JsonValue,
    UnsupportedDocument,
    WrongFormatDocument,
    encode_json,
    read_json,
)


T = TypeVar("T")


class FileState(str, Enum):
    MISSING = "missing"
    VALID = "valid"
    INVALID = "invalid"
    WRONG_FORMAT = "wrong_format"
    UNSUPPORTED = "unsupported"
    IO_ERROR = "io_error"


@dataclass(frozen=True)
class PersistenceIssue:
    """Stable machine-readable code and developer detail; games supply UI text."""

    code: str
    path: Path | None
    detail: str


@dataclass(frozen=True)
class Decoded(Generic[T]):
    value: T
    needs_save: bool = False
    issues: tuple[PersistenceIssue, ...] = ()


class JsonCodec(Protocol[T]):
    """Validate complete documents and encode detached, validated snapshots.

    Raise a DocumentError for expected invalid data. Unexpected exceptions are
    programming errors and propagate. No codec may mutate a live scene.
    """

    def decode(self, document: JsonValue) -> Decoded[T]: ...

    def encode(self, value: T) -> JsonValue: ...


@dataclass(frozen=True)
class StorePolicy:
    backup: bool = True
    unsupported_primary: Literal["stop", "readonly_backup"] = "stop"

    def __post_init__(self) -> None:
        if type(self.backup) is not bool:
            raise TypeError("backup must be a boolean")
        if self.unsupported_primary not in ("stop", "readonly_backup"):
            raise ValueError("unsupported_primary must be stop or readonly_backup")


@dataclass(frozen=True)
class LoadResult(Generic[T]):
    value: T | None
    source: Literal["primary", "backup", "none"]
    primary_state: FileState
    backup_state: FileState | None
    writable: bool
    needs_save: bool = False
    issues: tuple[PersistenceIssue, ...] = ()


@dataclass(frozen=True)
class WriteResult:
    """A successful primary replace remains committed despite cleanup warnings."""

    committed: bool
    issue: PersistenceIssue | None = None
    warnings: tuple[PersistenceIssue, ...] = ()
    archive_paths: tuple[Path, ...] = ()

    def __bool__(self) -> bool:
        return self.committed


@dataclass(frozen=True)
class _Observation(Generic[T]):
    state: FileState
    decoded: Decoded[T] | None = None
    raw: bytes | None = None
    issue: PersistenceIssue | None = None


class JsonStore(Generic[T]):
    """Store one game-defined JSON document at an explicit local path.

    Loading does not repair or create files. Normal writes independently
    revalidate existing files. Only explicit replacement can displace a damaged
    or unsupported primary, and it archives all managed files first.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        codec: JsonCodec[T],
        limits: JsonLimits,
        policy: StorePolicy | None = None,
        options: JsonOptions | None = None,
    ) -> None:
        self.path = atomic.normalize_path(path)
        if not callable(getattr(codec, "decode", None)) or not callable(
            getattr(codec, "encode", None)
        ):
            raise TypeError("codec must provide decode() and encode()")
        if not isinstance(limits, JsonLimits):
            raise TypeError("limits must be JsonLimits")
        if policy is not None and not isinstance(policy, StorePolicy):
            raise TypeError("policy must be StorePolicy")
        if options is not None and not isinstance(options, JsonOptions):
            raise TypeError("options must be JsonOptions")
        self.codec = codec
        self.limits = limits
        self.policy = StorePolicy() if policy is None else policy
        self.options = JsonOptions() if options is None else options

    @property
    def backup_path(self) -> Path:
        return self.path.with_name(self.path.name + ".bak")

    @staticmethod
    def _document_state(error: DocumentError) -> FileState:
        if isinstance(error, UnsupportedDocument):
            return FileState.UNSUPPORTED
        if isinstance(error, WrongFormatDocument):
            return FileState.WRONG_FORMAT
        return FileState.INVALID

    def _inspect(self, path: Path, role: str) -> _Observation[T]:
        # Catch filesystem errors only around reading, so a broken codec cannot
        # accidentally turn a programming error into a missing save.
        try:
            document, raw = read_json(path, limits=self.limits)
        except FileNotFoundError:
            return _Observation(FileState.MISSING)
        except OSError as error:
            return _Observation(
                FileState.IO_ERROR,
                issue=PersistenceIssue(f"{role}_io_error", path, str(error)),
            )
        except DocumentError as error:
            state = self._document_state(error)
            return _Observation(
                state, issue=PersistenceIssue(f"{role}_{state.value}", path, str(error))
            )
        try:
            decoded = self.codec.decode(document)
        except DocumentError as error:
            state = self._document_state(error)
            return _Observation(
                state, issue=PersistenceIssue(f"{role}_{state.value}", path, str(error))
            )
        if not isinstance(decoded, Decoded):
            raise TypeError("codec.decode() must return Decoded")
        return _Observation(FileState.VALID, decoded=decoded, raw=raw)

    def load(self) -> LoadResult[T]:
        """Read validated primary/backup data without changing any files."""
        with atomic.path_lock(self.path):
            primary = self._inspect(self.path, "primary")
            backup = None
            source: Literal["primary", "backup", "none"] = "none"
            chosen = None
            if primary.state == FileState.VALID:
                chosen, source = primary.decoded, "primary"
            elif self.policy.backup and not (
                primary.state == FileState.UNSUPPORTED
                and self.policy.unsupported_primary == "stop"
            ):
                backup = self._inspect(self.backup_path, "backup")
                if backup.state == FileState.VALID:
                    chosen, source = backup.decoded, "backup"

            writable = primary.state == FileState.VALID or (
                primary.state == FileState.MISSING
                and (backup is None or backup.state in (FileState.MISSING, FileState.VALID))
            )
            issues = tuple(
                observation.issue
                for observation in (primary, backup)
                if observation is not None and observation.issue is not None
            )
            if chosen is not None:
                issues += tuple(chosen.issues)
            return LoadResult(
                value=chosen.value if chosen is not None else None,
                source=source,
                primary_state=primary.state,
                backup_state=backup.state if backup is not None else None,
                writable=writable,
                needs_save=chosen.needs_save if chosen is not None else False,
                issues=issues,
            )

    def write(self, value: T) -> WriteResult:
        """Commit a detached snapshot without displacing protected documents."""
        return self._write(value, replacement=False, required_state=None)

    def replace_with_archive(
        self,
        value: T,
        *,
        required_primary_state: FileState | str | None = None,
    ) -> WriteResult:
        """Explicit reset; archive existing managed files and start a new primary.

        A required state is checked under the same lock as archiving and commit.
        Use INVALID for settings repair so an intervening newer document blocks
        the edit. Successful reset clears the rotating backup; the next ordinary
        save creates a backup belonging to the new world.
        """
        required = (
            None if required_primary_state is None else FileState(required_primary_state)
        )
        return self._write(value, replacement=True, required_state=required)

    def _write(
        self,
        value: T,
        *,
        replacement: bool,
        required_state: FileState | None,
    ) -> WriteResult:
        # All validation/serialization precedes filesystem mutation. Codecs own
        # domain validation; strict encoding also checks the outgoing JSON tree.
        try:
            data = encode_json(
                self.codec.encode(value), limits=self.limits, options=self.options
            )
        except DocumentError as error:
            return WriteResult(
                False, PersistenceIssue("invalid_value", self.path, str(error))
            )

        temporary = None
        backup_temporary = None
        archives: list[Path] = []
        warnings: list[PersistenceIssue] = []
        committed = False
        issue = None
        try:
            with atomic.path_lock(self.path):
                primary = self._inspect(self.path, "primary")
                if replacement:
                    if required_state is not None and primary.state != required_state:
                        return WriteResult(
                            False,
                            PersistenceIssue(
                                "state_changed",
                                self.path,
                                f"required {required_state.value}; found {primary.state.value}",
                            ),
                        )
                    if primary.state == FileState.IO_ERROR:
                        return WriteResult(False, primary.issue)
                    backup = (
                        self._inspect(self.backup_path, "backup")
                        if self.policy.backup
                        else None
                    )
                    if backup is not None and backup.state == FileState.IO_ERROR:
                        return WriteResult(False, backup.issue)
                else:
                    if primary.state not in (FileState.VALID, FileState.MISSING):
                        return WriteResult(False, primary.issue)
                    backup = None
                    if primary.state == FileState.MISSING and self.policy.backup:
                        backup = self._inspect(self.backup_path, "backup")
                        if backup.state not in (FileState.VALID, FileState.MISSING):
                            return WriteResult(False, backup.issue)

                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = atomic.write_temporary(self.path, data)
                if replacement:
                    if primary.state != FileState.MISSING:
                        archives.append(atomic.archive_file(self.path))
                    if backup is not None and backup.state != FileState.MISSING:
                        archives.append(atomic.archive_file(self.backup_path))
                        self.backup_path.unlink()
                elif self.policy.backup and primary.state == FileState.VALID:
                    # Preserve exactly the accepted older representation, not a
                    # re-encoding of a potentially migrated in-memory value.
                    assert primary.raw is not None
                    backup_temporary = atomic.write_temporary(self.backup_path, primary.raw)
                    os.replace(backup_temporary, self.backup_path)
                    backup_temporary = None

                os.replace(temporary, self.path)
                committed = True
                temporary = None
        except OSError as error:
            issue = PersistenceIssue("write_failed", self.path, str(error))
        finally:
            for leftover in (temporary, backup_temporary):
                error = atomic.cleanup_temporary(leftover)
                if error is not None:
                    warnings.append(
                        PersistenceIssue("cleanup_failed", leftover, str(error))
                    )
        return WriteResult(
            committed=committed,
            issue=None if committed else issue,
            warnings=tuple(warnings),
            archive_paths=tuple(archives),
        )


__all__ = [
    "Decoded",
    "FileState",
    "JsonCodec",
    "JsonStore",
    "LoadResult",
    "PersistenceIssue",
    "StorePolicy",
    "WriteResult",
]
