"""Strict, bounded JSON transport independent of game document schemas."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterator, TypeAlias


JsonValue: TypeAlias = (
    bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class DocumentError(ValueError):
    """An expected failure to interpret or produce a stored document."""


class InvalidDocument(DocumentError):
    """A document is malformed or fails its caller's schema validation."""


class WrongFormatDocument(DocumentError):
    """A document belongs to another format."""


class UnsupportedDocument(DocumentError):
    """A recognized document has an unsupported version."""


@dataclass(frozen=True, slots=True)
class JsonLimits:
    """Transport bounds; the root has depth zero and counts as one node.

    Every container and scalar value counts as a node. Object keys do not.
    The byte limit includes the newline added by :func:`encode_json`.
    """

    max_bytes: int
    max_depth: int | None = None
    max_nodes: int | None = None

    def __post_init__(self) -> None:
        if type(self.max_bytes) is not int:
            raise TypeError("max_bytes must be an integer")
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        for name, minimum in (("max_depth", 0), ("max_nodes", 1)):
            value = getattr(self, name)
            if value is None:
                continue
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer or None")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class JsonOptions:
    """Serialization style without changing the allowed document types."""

    indent: int | None = None
    sort_keys: bool = False
    ensure_ascii: bool = True

    def __post_init__(self) -> None:
        if self.indent is not None:
            if type(self.indent) is not int:
                raise TypeError("indent must be an integer or None")
            if self.indent < 0:
                raise ValueError("indent must be nonnegative")
        if type(self.sort_keys) is not bool:
            raise TypeError("sort_keys must be a boolean")
        if type(self.ensure_ascii) is not bool:
            raise TypeError("ensure_ascii must be a boolean")


def validate_json(document: JsonValue, *, limits: JsonLimits) -> None:
    """Validate exact JSON builtins iteratively, rejecting cycles and coercion.

    Iterators keep traversal storage proportional to nesting depth. Repeated
    references to an acyclic container are allowed and serialize as copies.
    Byte size is checked by the transport functions, after serialization.
    """

    stack: list[tuple[Iterator[object], int, int | None]] = [
        (iter((document,)), 0, None)
    ]
    active: set[int] = set()
    nodes = 0
    while stack:
        children, depth, container_id = stack[-1]
        try:
            value = next(children)
        except StopIteration:
            stack.pop()
            if container_id is not None:
                active.remove(container_id)
            continue

        nodes += 1
        if limits.max_nodes is not None and nodes > limits.max_nodes:
            raise InvalidDocument("JSON document exceeds its node limit")
        if limits.max_depth is not None and depth > limits.max_depth:
            raise InvalidDocument("JSON document exceeds its depth limit")

        kind = type(value)
        if value is None or kind in (bool, int, str):
            continue
        if kind is float:
            if not math.isfinite(value):
                raise InvalidDocument("JSON numbers must be finite")
            continue
        if kind not in (dict, list):
            raise InvalidDocument("JSON values must use exact built-in JSON types")
        identity = id(value)
        if identity in active:
            raise InvalidDocument("JSON document contains a cycle")
        if kind is dict:
            if any(type(key) is not str for key in value):
                raise InvalidDocument("JSON object keys must be strings")
            values = iter(value.values())
        else:
            values = iter(value)
        active.add(identity)
        stack.append((values, depth + 1, identity))


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidDocument("JSON object contains duplicate keys")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise InvalidDocument(f"JSON contains a nonfinite constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise InvalidDocument("JSON number overflows to a nonfinite value")
    return number


def decode_json(data: bytes, *, limits: JsonLimits) -> JsonValue:
    """Decode bounded UTF-8 JSON and validate its complete value tree."""

    if type(data) is not bytes:
        raise TypeError("data must be bytes")
    if len(data) > limits.max_bytes:
        raise InvalidDocument("JSON document exceeds its byte limit")
    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
            parse_float=_finite_float,
        )
    except InvalidDocument:
        raise
    except (UnicodeError, ValueError, OverflowError, RecursionError) as error:
        raise InvalidDocument(f"Invalid UTF-8 JSON document: {error}") from error
    validate_json(document, limits=limits)
    return document


def encode_json(
    document: JsonValue,
    *,
    limits: JsonLimits,
    options: JsonOptions | None = None,
) -> bytes:
    """Encode a validated JSON tree as UTF-8 with one trailing newline."""

    validate_json(document, limits=limits)
    style = JsonOptions() if options is None else options
    try:
        text = json.dumps(
            document,
            allow_nan=False,
            indent=style.indent,
            sort_keys=style.sort_keys,
            ensure_ascii=style.ensure_ascii,
            separators=(",", ":") if style.indent is None else (",", ": "),
        )
        data = (text + "\n").encode("utf-8")
    except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as error:
        raise InvalidDocument(f"JSON document could not be encoded: {error}") from error
    if len(data) > limits.max_bytes:
        raise InvalidDocument("JSON document exceeds its byte limit")
    return data


def read_json(path: str | Path, *, limits: JsonLimits) -> tuple[JsonValue, bytes]:
    """Read at most the configured byte limit plus one, preserving raw bytes."""

    with Path(path).open("rb") as stream:
        data = stream.read(limits.max_bytes + 1)
    return decode_json(data, limits=limits), data


__all__ = [
    "JsonValue", "JsonLimits", "JsonOptions", "DocumentError", "InvalidDocument",
    "WrongFormatDocument", "UnsupportedDocument", "validate_json", "decode_json",
    "encode_json", "read_json",
]
