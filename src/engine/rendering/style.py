"""Engine-neutral fog, vibrance, and material-shine snapshots."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextureFogState:
    enabled: bool = False
    density: float = 0.0
    color: tuple[float, float, float, float] = (0.7, 0.8, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class TextureVibranceState:
    vibrance: float = 1.0


@dataclass(frozen=True, slots=True)
class TextureShineState:
    enabled: bool = True
    strength: float = 0.38
    power: float = 28.0
    fresnel: float = 0.18
    tint: tuple[float, float, float] = (1.0, 0.96, 0.86)


_fog_state = TextureFogState()
_vibrance_state = TextureVibranceState()
_shine_state = TextureShineState()


def get_render_fog_state() -> TextureFogState:
    return _fog_state


def update_render_fog_state(
    *,
    enabled: bool | None = None,
    density: float | None = None,
    color: tuple[float, float, float, float] | None = None,
) -> TextureFogState:
    global _fog_state
    current = _fog_state
    _fog_state = TextureFogState(
        enabled=bool(enabled) if enabled is not None else current.enabled,
        density=max(0.0, float(density)) if density is not None else current.density,
        color=tuple(float(value) for value in color) if color is not None else current.color,
    )
    return _fog_state


def get_render_vibrance_state() -> TextureVibranceState:
    return _vibrance_state


def update_render_vibrance_state(
    vibrance: float | None = None,
) -> TextureVibranceState:
    global _vibrance_state
    value = _vibrance_state.vibrance if vibrance is None else float(vibrance)
    _vibrance_state = TextureVibranceState(
        vibrance=max(0.0, min(2.0, value)),
    )
    return _vibrance_state


def get_render_shine_state() -> TextureShineState:
    return _shine_state


def update_render_shine_state(
    *,
    enabled: bool | None = None,
    strength: float | None = None,
    power: float | None = None,
    fresnel: float | None = None,
    tint: tuple[float, float, float] | None = None,
) -> TextureShineState:
    global _shine_state
    current = _shine_state
    _shine_state = TextureShineState(
        enabled=bool(enabled) if enabled is not None else current.enabled,
        strength=max(0.0, float(strength)) if strength is not None else current.strength,
        power=max(1.0, float(power)) if power is not None else current.power,
        fresnel=max(0.0, float(fresnel)) if fresnel is not None else current.fresnel,
        tint=tuple(float(value) for value in tint) if tint is not None else current.tint,
    )
    return _shine_state
