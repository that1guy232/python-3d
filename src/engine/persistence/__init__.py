"""Data-only persistence services with game-owned codecs, paths, and schemas.

Importing this package needs only the standard library. It starts no worker,
reads no preferences, and does not create any files or directories.
"""

from .atomic import atomic_write_bytes
from .json_io import (
    DocumentError,
    InvalidDocument,
    JsonLimits,
    JsonOptions,
    JsonValue,
    UnsupportedDocument,
    WrongFormatDocument,
    decode_json,
    encode_json,
    read_json,
    validate_json,
)
from .store import (
    Decoded,
    FileState,
    JsonCodec,
    JsonStore,
    LoadResult,
    PersistenceIssue,
    StorePolicy,
    WriteResult,
)
from .preferences import PreferenceSchema, PreferenceSpec, PreferenceStore


__all__ = [
    "Decoded",
    "DocumentError",
    "FileState",
    "InvalidDocument",
    "JsonCodec",
    "JsonLimits",
    "JsonOptions",
    "JsonStore",
    "JsonValue",
    "LoadResult",
    "PersistenceIssue",
    "PreferenceSchema",
    "PreferenceSpec",
    "PreferenceStore",
    "StorePolicy",
    "UnsupportedDocument",
    "WriteResult",
    "WrongFormatDocument",
    "atomic_write_bytes",
    "decode_json",
    "encode_json",
    "read_json",
    "validate_json",
]
