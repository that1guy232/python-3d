"""Validated scalar preferences with persistence before publication.

The supplied ``JsonStore`` codec owns the document envelope and must decode a
settings mapping. This module owns neither file formats nor scene callbacks.
Callers load existing preferences, submit edits, and apply ``values`` to their
application only after a committed result. One logical owner supplies edits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from .store import FileState, JsonStore, LoadResult, PersistenceIssue, WriteResult


PreferenceValue: TypeAlias = bool | int | float | str | None
_VALUE_TYPES = (bool, int, float, str, type(None))
_NUMERIC_TYPES = (int, float)


def _is_finite_number(value: object) -> bool:
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


@dataclass(frozen=True, slots=True)
class PreferenceSpec:
    """One immutable JSON scalar and its normalization/adjustment rules.

    Float fields accept integer or floating-point inputs and return floats.
    Integer fields require actual integers; booleans are never numbers here.
    Numeric choices are exact choices, while ``step`` only controls adjustment.
    """

    default: PreferenceValue
    value_type: type
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[PreferenceValue, ...] | None = None
    range_policy: str = "reject"
    step: int | float | None = None
    round_digits: int | None = None

    def __post_init__(self) -> None:
        if self.value_type not in _VALUE_TYPES:
            raise ValueError("preference type must be an immutable JSON scalar type")
        if self.range_policy not in ("reject", "clamp"):
            raise ValueError("range_policy must be 'reject' or 'clamp'")
        numeric = self.value_type in _NUMERIC_TYPES
        if not numeric and (
            self.minimum is not None or self.maximum is not None
            or self.step is not None or self.round_digits is not None
            or self.range_policy != "reject"
        ):
            raise ValueError("bounds, rounding, clamping and steps require a numeric field")
        for name in ("minimum", "maximum", "step"):
            value = getattr(self, name)
            if value is not None and not _is_finite_number(value):
                raise ValueError(f"{name} must be a finite number")
            if self.value_type is int and value is not None and type(value) is not int:
                raise ValueError(f"an integer field requires an integer {name}")
            if self.value_type is float and value is not None:
                try:
                    normalized = float(value)
                except OverflowError as exc:
                    raise ValueError(f"{name} must be a finite float") from exc
                if not math.isfinite(normalized):
                    raise ValueError(f"{name} must be a finite float")
                object.__setattr__(self, name, normalized)
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        if self.step is not None and self.step <= 0:
            raise ValueError("step must be positive")
        if self.round_digits is not None and (
            type(self.round_digits) is not int or self.round_digits < 0
        ):
            raise ValueError("round_digits must be a nonnegative integer")
        if self.choices is not None:
            if not isinstance(self.choices, (tuple, list)) or not self.choices:
                raise ValueError("choices must be a nonempty tuple or list")
            if self.minimum is not None or self.maximum is not None or self.step is not None:
                raise ValueError("choices cannot be combined with bounds or a step")
            choices = tuple(self._normalize_scalar(value, range_policy="reject")
                            for value in self.choices)
            if len(set(choices)) != len(choices):
                raise ValueError("choices must be distinct after normalization")
            object.__setattr__(self, "choices", choices)
        default = self._normalize_scalar(self.default, range_policy="reject")
        if self.choices is not None and default not in self.choices:
            raise ValueError("default must be one of the choices")
        object.__setattr__(self, "default", default)

    def _normalize_scalar(self, value: object, *, range_policy: str) -> PreferenceValue:
        if self.value_type is float:
            if type(value) not in _NUMERIC_TYPES:
                raise ValueError("value must be a finite number")
            try:
                value = float(value)
            except (ValueError, OverflowError) as exc:
                raise ValueError("value must be a finite number") from exc
            if not math.isfinite(value):
                raise ValueError("value must be a finite number")
        elif type(value) is not self.value_type:
            raise ValueError(f"value must have type {self.value_type.__name__}")

        if self.value_type in _NUMERIC_TYPES:
            if self.minimum is not None and value < self.minimum:
                if range_policy == "reject":
                    raise ValueError("value is below its minimum")
                value = self.minimum
            if self.maximum is not None and value > self.maximum:
                if range_policy == "reject":
                    raise ValueError("value is above its maximum")
                value = self.maximum
            if self.round_digits is not None:
                value = round(value, self.round_digits)
                # Rounding an in-range value must not place it beyond an endpoint.
                if self.minimum is not None:
                    value = max(self.minimum, value)
                if self.maximum is not None:
                    value = min(self.maximum, value)
            if self.value_type is float:
                value = float(value)
        return value

    def validate(self, value: object) -> PreferenceValue:
        """Return a normalized value, raising ``ValueError`` for invalid edits."""
        normalized = self._normalize_scalar(value, range_policy=self.range_policy)
        if self.choices is not None and normalized not in self.choices:
            raise ValueError("value is not one of the supported choices")
        return normalized


@dataclass(frozen=True, slots=True, init=False)
class PreferenceSchema:
    """Immutable fields with tolerant loading and all-or-nothing editing."""

    _fields: Mapping[str, PreferenceSpec]
    _defaults: Mapping[str, PreferenceValue]

    def __init__(self, fields: Mapping[str, PreferenceSpec]) -> None:
        if not isinstance(fields, Mapping):
            raise ValueError("preference fields must be a mapping")
        copied = dict(fields)
        for key, spec in copied.items():
            if type(key) is not str or not key:
                raise ValueError("preference keys must be nonempty strings")
            if not isinstance(spec, PreferenceSpec):
                raise ValueError(f"{key!r} must have a PreferenceSpec")
        object.__setattr__(self, "_fields", MappingProxyType(copied))
        object.__setattr__(self, "_defaults", MappingProxyType(
            {key: spec.default for key, spec in copied.items()}))

    @property
    def fields(self) -> Mapping[str, PreferenceSpec]:
        return self._fields

    @property
    def defaults(self) -> Mapping[str, PreferenceValue]:
        return self._defaults

    def normalize_loaded(
        self, values: Mapping[str, object], *, path: Path | None = None,
    ) -> tuple[dict[str, PreferenceValue], tuple[PersistenceIssue, ...]]:
        """Default missing/invalid fields and ignore unrecognized loaded keys.

        The codec must reject an invalid document shape before calling this
        layer; a non-mapping here is a codec contract error, not damaged input.
        """
        if not isinstance(values, Mapping):
            raise ValueError("decoded preferences must be a mapping")
        normalized = dict(self._defaults)
        issues = []
        for key, spec in self._fields.items():
            if key not in values:
                continue
            try:
                normalized[key] = spec.validate(values[key])
            except ValueError as exc:
                issues.append(PersistenceIssue(
                    code="preference_invalid_field", path=path,
                    detail=f"Preference {key!r} returned to its default: {exc}",
                ))
        return normalized, tuple(issues)

    def validate_candidate(self, values: Mapping[str, object]) -> dict[str, PreferenceValue]:
        """Validate a complete candidate without changing the schema or inputs."""
        if not isinstance(values, Mapping):
            raise ValueError("preference values must be a mapping")
        unknown = [key for key in values if key not in self._fields]
        if unknown:
            raise ValueError(f"unknown preference {unknown[0]!r}")
        missing = [key for key in self._fields if key not in values]
        if missing:
            raise ValueError(f"missing preference {missing[0]!r}")
        normalized = {}
        for key, spec in self._fields.items():
            try:
                normalized[key] = spec.validate(values[key])
            except ValueError as exc:
                raise ValueError(f"invalid preference {key!r}: {exc}") from exc
        return normalized

    def adjust(
        self, values: Mapping[str, object], key: str, direction: int | float,
    ) -> dict[str, PreferenceValue]:
        """Move one step or cycle one choice, clamping numerical adjustments."""
        if type(key) is not str or key not in self._fields:
            raise ValueError(f"unknown preference {key!r}")
        if not _is_finite_number(direction):
            raise ValueError("adjustment direction must be a finite number")
        candidate = self.validate_candidate(values)
        if direction == 0:
            return candidate
        spec = self._fields[key]
        sign = 1 if direction > 0 else -1
        current = candidate[key]
        if spec.choices is not None:
            index = spec.choices.index(current)
            value = spec.choices[(index + sign) % len(spec.choices)]
        else:
            if spec.value_type not in _NUMERIC_TYPES or spec.step is None:
                raise ValueError(f"preference {key!r} has no adjustment step or choices")
            value = current + sign * spec.step
            if spec.minimum is not None:
                value = max(spec.minimum, value)
            if spec.maximum is not None:
                value = min(spec.maximum, value)
        candidate[key] = spec.validate(value)
        return candidate


class PreferenceStore:
    """Persist validated candidates before publishing committed preferences.

    ``load()`` never writes. Invalid-field defaults do not change document
    eligibility: the supplied game codec owns full current-file validation.
    Storage always rechecks eligibility, independent of earlier load results.
    ``reset()`` is an explicit archive-and-replace operation, including for an
    unsupported document; ordinary edits can repair only an invalid primary.
    """

    def __init__(
        self, store: JsonStore[dict], *, schema: PreferenceSchema,
        repair_invalid_on_edit: bool = False,
    ) -> None:
        if not isinstance(schema, PreferenceSchema):
            raise ValueError("schema must be a PreferenceSchema")
        if type(repair_invalid_on_edit) is not bool:
            raise ValueError("repair_invalid_on_edit must be a boolean")
        self.store = store
        self.schema = schema
        self.repair_invalid_on_edit = repair_invalid_on_edit
        self._values = MappingProxyType(schema.validate_candidate(schema.defaults))
        self._last_load: LoadResult[dict] | None = None

    @property
    def values(self) -> Mapping[str, PreferenceValue]:
        """Read-only committed values; previously returned snapshots stay fixed."""
        return self._values

    @property
    def last_load(self) -> LoadResult[dict] | None:
        """The last load observation, for diagnostics rather than authorization."""
        return self._last_load

    def load(self) -> LoadResult[dict]:
        loaded = self.store.load()
        if loaded.value is None:
            values, issues = dict(self.schema.defaults), ()
        else:
            values, issues = self.schema.normalize_loaded(loaded.value, path=self.store.path)
        self._values = MappingProxyType(dict(values))
        result = LoadResult(
            value=dict(values), source=loaded.source,
            primary_state=loaded.primary_state, backup_state=loaded.backup_state,
            writable=loaded.writable, needs_save=loaded.needs_save,
            issues=tuple(loaded.issues) + issues,
        )
        self._last_load = result
        return result

    def _persist(self, candidate: dict[str, PreferenceValue], *, reset: bool = False) -> WriteResult:
        if reset:
            result = self.store.replace_with_archive(dict(candidate))
        else:
            result = self.store.write(dict(candidate))
            if (not result.committed and self.repair_invalid_on_edit
                    and result.issue is not None and result.issue.code == "primary_invalid"):
                result = self.store.replace_with_archive(
                    dict(candidate), required_primary_state=FileState.INVALID,
                )
        if result.committed:
            self._values = MappingProxyType(dict(candidate))
        return result

    def update(self, changes: Mapping[str, object]) -> WriteResult:
        if not isinstance(changes, Mapping):
            raise ValueError("preference changes must be a mapping")
        for key in changes:
            if key not in self.schema.fields:
                raise ValueError(f"unknown preference {key!r}")
        candidate = self.schema.validate_candidate({**self._values, **changes})
        return self._persist(candidate)

    def set(self, key: str, value: object) -> WriteResult:
        if type(key) is not str or key not in self.schema.fields:
            raise ValueError(f"unknown preference {key!r}")
        return self.update({key: value})

    def adjust(self, key: str, direction: int | float) -> WriteResult:
        candidate = self.schema.adjust(self._values, key, direction)
        return self._persist(candidate)

    def reset(self) -> WriteResult:
        candidate = self.schema.validate_candidate(self.schema.defaults)
        return self._persist(candidate, reset=True)

    def save(self, values: Mapping[str, object] | None = None) -> WriteResult:
        """Persist current values or a complete replacement candidate."""
        candidate = self.schema.validate_candidate(self._values if values is None else values)
        return self._persist(candidate)


__all__ = ["PreferenceValue", "PreferenceSpec", "PreferenceSchema", "PreferenceStore"]
