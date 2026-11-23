# Engine persistence

The [persistence package](../src/engine/persistence/__init__.py) provides standard-library-only JSON storage and validated preferences. It creates no files or workers on import and has no graphics dependencies. Games choose paths, document formats, versions, validation, migrations, messages, and save timing. Existing games continue using their own persistence until explicitly integrated.

## Storing a snapshot

A codec validates the complete document and encodes detached snapshots. Expected data problems raise `InvalidDocument`, `WrongFormatDocument`, or `UnsupportedDocument`; programming errors propagate.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.persistence import (
    Decoded, InvalidDocument, JsonLimits, JsonStore,
    UnsupportedDocument, WrongFormatDocument,
)


class CounterCodec:
    @staticmethod
    def count(value):
        if type(value) is not dict or type(value.get("count")) is not int:
            raise InvalidDocument("count must be an integer")
        if value["count"] < 0:
            raise InvalidDocument("count must be nonnegative")
        return value["count"]

    def decode(self, document):
        if type(document) is not dict or document.get("format") != "example.counter":
            raise WrongFormatDocument("not a counter document")
        if type(document.get("version")) is not int:
            raise InvalidDocument("version must be an integer")
        if document["version"] != 1:
            raise UnsupportedDocument("unsupported counter version")
        return Decoded({"count": self.count(document)})

    def encode(self, snapshot):
        return {"format": "example.counter", "version": 1,
                "count": self.count(snapshot)}


with TemporaryDirectory() as directory:
    store = JsonStore(Path(directory) / "counter.json",
                      codec=CounterCodec(), limits=JsonLimits(max_bytes=4096))
    assert store.write({"count": 1}).committed
    assert store.write({"count": 2}).committed
    loaded = store.load()
    assert loaded.value == {"count": 2}
    assert loaded.writable
    # counter.json.bak contains the exact first document.
```

The [store API](../src/engine/persistence/store.py) exposes `LoadResult` and `WriteResult`. Load results identify the source, observed primary/backup states, write eligibility, migration needs, and issues. `Decoded(needs_save=True)` records in-memory migration without making loading write anything. Write results expose `committed`, `issue`, `warnings`, and `archive_paths`; their boolean value equals `committed`.

Issue codes include `primary_invalid`, `primary_wrong_format`, `primary_unsupported`, and `primary_io_error`, with equivalent `backup_` codes. `invalid_value` means outgoing validation failed; `write_failed` reports filesystem failure. Games map codes to their own messages.

## Recovery and replacement

`StorePolicy` defaults to previous-good backups and `unsupported_primary="stop"`. Choose `backup=False` for files without recovery generations, or `unsupported_primary="readonly_backup"` to permit read-only recovery from a valid backup behind an unsupported primary.

| Observed files | Load behavior | Ordinary writes |
| --- | --- | --- |
| Valid primary | Use primary; skip inspecting backup | Allowed |
| Both missing | No snapshot | Allowed |
| Missing primary and valid backup | Use backup without repairing disk | Allowed; preserve backup |
| Invalid or wrong-format primary, valid backup | Use backup read-only | Blocked |
| Unsupported primary | Stop, or recover read-only under the chosen policy | Blocked |
| Missing primary and unusable backup | No snapshot | Blocked |
| I/O failure | Use a valid backup read-only when available | Blocked |

Every normal write rechecks current files under the path lock. Earlier load eligibility does not authorize overwriting a subsequently damaged file. A missing file is distinguished from permission errors and directories at the target path.

`store.replace_with_archive(snapshot)` is an explicit reset. It streams complete existing managed files to unique flushed archives, clears the old rotating backup, then replaces the primary. Its next ordinary save creates a backup belonging to the new state. Returned archive paths also accompany failures after archival. A reset failure may leave the old backup only in its archive, while the original primary remains intact.

For conditional repair, use `replace_with_archive(snapshot, required_primary_state=FileState.INVALID)`. The state condition is checked inside the same lock as archival and commit. A mismatch returns `state_changed` without changing files. This prevents a settings repair from displacing a file that became an unsupported document before the operation acquired its lock.

Each replacement commits one file. New data is validated, written to a unique sibling, flushed and synced; an ordinary save installs a backup of the validated previous primary before replacing that primary. Backup failure aborts the ordinary save. A primary replacement that succeeded remains `committed=True` even if cleanup reports a warning. This is not a transaction across snapshot, backup, archives, or metadata, and does not promise portable power-loss durability for directory entries.

## Strict JSON transport

The [JSON helpers](../src/engine/persistence/json_io.py) expose `read_json`, `decode_json`, `encode_json`, and `validate_json`. Reads consume at most `max_bytes + 1` bytes. Strict parsing rejects duplicate keys, malformed UTF-8, nonfinite constants, overflow to infinity, and excess byte/depth/node budgets.

Outgoing data must be an acyclic tree of exact built-in JSON types. Convert tuples, custom containers, NumPy values, and domain objects in the codec. Dictionary keys must be strings; silent key/type coercion is rejected. Repeated references to the same acyclic container are allowed. The root is depth zero; node counts include containers and scalar values, excluding dictionary keys. Encoded output has one trailing newline, included in the byte limit.

`JsonOptions(indent=2, sort_keys=True, ensure_ascii=False)` selects formatting. Budgets are configured with `JsonLimits(max_bytes=..., max_depth=..., max_nodes=...)`. Domain-specific bounds remain codec responsibilities.

For schema-free metadata bytes, `atomic_write_bytes(path, data)` provides [atomic replacement](../src/engine/persistence/atomic.py) without validation or backups. It raises on pre-commit failure and returns an optional `OSError` for post-commit cleanup trouble. It is a low-level primitive; use `JsonStore` for protected snapshot writes.

## Validated preferences

`PreferenceStore` wraps a `JsonStore` whose codec decodes a settings mapping. The codec owns the envelope and version checks; let the schema normalize individual fields instead of rejecting the whole mapping for a bad setting.

```python
from engine.persistence import PreferenceSchema, PreferenceSpec, PreferenceStore

schema = PreferenceSchema({
    "fov": PreferenceSpec(default=60.0, value_type=float,
                          minimum=45.0, maximum=90.0,
                          step=5.0, round_digits=6),
    "frame_limit": PreferenceSpec(default=60, value_type=int,
                                  choices=(30, 60, 120, 0)),
    "hud_visible": PreferenceSpec(default=True, value_type=bool,
                                  choices=(False, True)),
})

# Supply a JsonStore with the application's own settings codec and path.
preferences = PreferenceStore(settings_store, schema=schema,
                              repair_invalid_on_edit=True)
loaded = preferences.load()  # Defaults invalid fields; never writes.
result = preferences.update({"fov": 75.0})
if result.committed:
    apply_settings(preferences.values)  # Application-owned callback.
```

Specs support immutable scalar types, defaults, optional numeric bounds, choices, rounding, and steps. Numeric fields reject booleans and nonfinite values; integer choices reject floating-point equivalents. `range_policy="clamp"` clamps finite numeric input; the default is `"reject"`. A step controls adjustment, not which values may be stored. Defaults and schema definitions are validated at construction.

Missing loaded fields use defaults, unknown loaded keys are ignored, and invalid fields produce diagnostics. A wholly invalid JSON document yields all defaults. Edits reject unknown keys and validate the entire candidate. `.values` is an immutable snapshot; previously returned snapshots remain fixed after updates. The mutable copy in a load result cannot change the committed values.

`set(key, value)`, `update(changes)`, `adjust(key, direction)`, and `save(values=None)` persist before publishing. Invalid edits raise `ValueError`; expected storage failures return a noncommitted `WriteResult`. Every requested mutation attempts a write, including a same-value set or zero adjustment. The application applies live effects only after successful commit. No scene callbacks run inside persistence.

With `repair_invalid_on_edit=True`, only a `primary_invalid` failure triggers guarded archived repair. Wrong-format/unsupported files and I/O failures stay protected. `reset()` is the explicit unrestricted archive-and-replace operation using validated defaults. `last_load` supplies diagnostic history, not authorization to write.

## Ownership and integration

Use one logical snapshot/preference owner per path. In-process stores share locks for expanded, resolved, case-normalized paths; unrelated paths use separate locks. The canonical path is also used for I/O. Locks do not merge stale snapshots or exclude writers in another process.

The API is synchronous and starts no executor. Applications can call it from an existing worker using detached snapshots captured on the main thread. Keep save triggers, coalescing, autosave preferences, and shutdown/discard behavior in the application.

The [technical design](designs/persistence.md) records the recovery matrix, compatibility considerations, and later game adoption plan. The current change implements only engine services.
