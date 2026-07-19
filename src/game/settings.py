"""Durable storage and validation for user-adjustable game settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


SETTINGS_VERSION = 1

DEFAULT_SETTINGS: dict[str, bool | float] = {
    "audio_muted": False,
    "hud_visible": True,
    "compass_visible": True,
    "minimap_visible": True,
    "held_item_visible": True,
    "test_light_visible": True,
    "controls_text_visible": True,
    "fog_enabled": True,
    "clouds_enabled": True,
    "cloud_density": 1.0,
    "cloud_speed": 1.0,
    "cloud_opacity": 0.9,
    "fog_density": 0.0005,
    "fov": 90.0,
    "brightness": 0.8,
    "vibrance": 1.15,
    "mouse_sensitivity": 0.0015,
    "look_smooth": 4.0,
    "walk_speed": 120.0,
    "sprint_speed": 180.0,
    "road_speed_multiplier": 1.5,
    "jump_speed": 250.0,
    "gravity": 800.0,
    "camera_follow_smooth_hz": 5.0,
    "eye_height": 0.0,
    "height_adjust_speed": 50.0,
    "headbob_enabled": True,
    "headbob_frequency": 0.5,
    "headbob_amplitude_y": 4.0,
    "headbob_amplitude_x": 3.0,
    "headbob_damping": 2.0,
    "idle_sway_enabled": True,
    "idle_delay": 1.0,
    "idle_amount": 0.35,
    "breath_amount": 1.0,
    "mouse_sway_enabled": True,
    "sway_scale": 0.01,
    "sway_limit_x": 1.25,
    "sway_limit_y": 0.75,
}

_NUMBER_RANGES: dict[str, tuple[float, float]] = {
    "cloud_density": (0.0, 2.0),
    "cloud_speed": (0.0, 3.0),
    "cloud_opacity": (0.0, 1.0),
    "fog_density": (0.0, 0.002),
    "fov": (70.0, 110.0),
    "brightness": (0.4, 1.2),
    "vibrance": (0.0, 2.0),
    "mouse_sensitivity": (0.0008, 0.003),
    "look_smooth": (0.0, 12.0),
    "walk_speed": (80.0, 220.0),
    "sprint_speed": (120.0, 320.0),
    "road_speed_multiplier": (1.0, 2.0),
    "jump_speed": (150.0, 500.0),
    "gravity": (400.0, 1800.0),
    "camera_follow_smooth_hz": (0.0, 10.0),
    "eye_height": (-25.0, 100.0),
    "height_adjust_speed": (25.0, 200.0),
    "headbob_frequency": (0.25, 1.5),
    "headbob_amplitude_y": (0.0, 8.0),
    "headbob_amplitude_x": (0.0, 5.0),
    "headbob_damping": (1.0, 8.0),
    "idle_delay": (0.0, 4.0),
    "idle_amount": (0.0, 1.2),
    "breath_amount": (0.0, 2.0),
    "sway_scale": (0.0, 0.04),
    "sway_limit_x": (0.0, 2.0),
    "sway_limit_y": (0.0, 1.2),
}


def settings_path() -> Path:
    """Return the per-user settings path, with an override for development."""

    override = os.getenv("PY3D_SETTINGS_PATH")
    if override:
        return Path(override).expanduser()

    if os.name == "nt":
        base = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "python-3d" / "settings.json"


def validate_settings(values: Mapping[str, Any] | None) -> dict[str, bool | float]:
    """Merge recognized, well-typed values over safe defaults."""

    validated = dict(DEFAULT_SETTINGS)
    if not isinstance(values, Mapping):
        return validated

    for name, default in DEFAULT_SETTINGS.items():
        value = values.get(name)
        if isinstance(default, bool):
            if isinstance(value, bool):
                validated[name] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        minimum, maximum = _NUMBER_RANGES[name]
        validated[name] = max(minimum, min(maximum, float(value)))
    return validated


def load_settings(path: Path | None = None) -> dict[str, bool | float]:
    """Load validated settings, falling back safely for missing/corrupt files."""

    target = settings_path() if path is None else Path(path)
    try:
        with target.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        return dict(DEFAULT_SETTINGS)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"[settings] Could not load {target}: {exc}")
        return dict(DEFAULT_SETTINGS)

    if not isinstance(document, dict):
        return dict(DEFAULT_SETTINGS)
    values = document.get("settings", document)
    return validate_settings(values)


def save_settings(
    values: Mapping[str, Any], path: Path | None = None
) -> bool:
    """Atomically write validated settings. Returns whether the save succeeded."""

    target = settings_path() if path is None else Path(path)
    temporary = target.with_name(f"{target.name}.tmp")
    document = {
        "version": SETTINGS_VERSION,
        "settings": validate_settings(values),
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        return True
    except OSError as exc:
        print(f"[settings] Could not save {target}: {exc}")
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return False
