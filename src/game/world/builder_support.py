"""Shared support helpers for world construction pipelines."""

from __future__ import annotations


from dataclasses import dataclass

from typing import Callable


@dataclass(frozen=True)
class WorldObjectBuildStep:

    label: str

    action: Callable[[], None]

    message: str | None = None


def _dispose_value(obj) -> None:

    dispose = getattr(obj, "dispose", None)

    if callable(dispose):

        try:

            dispose()

        except Exception:

            pass


def _dispose_values(values) -> None:

    for value in values or ():

        _dispose_value(value)
