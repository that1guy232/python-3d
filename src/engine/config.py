"""Engine runtime configuration and environment-variable feature flags."""

from __future__ import annotations

import os as _os


def _env_bool(name: str, default: bool = False) -> bool:
    value = _os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer setting without making a bad override fatal at import."""

    value = _os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    """Read a finite float setting, falling back for malformed values."""

    value = _os.getenv(name)
    if value is None:
        return float(default)
    try:
        parsed = float(value.strip())
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed == parsed and abs(parsed) != float("inf") else float(default)


WIDTH = 1600
HEIGHT = 900
FULLSCREEN = False
VIEWDISTANCE = 2000
FOGDENSITY = 0.0005
LIGHT_BLUE = (0.3, 0.6, 1.0, 1.0)
FOV = 90
# Rendering is uncapped by default.  Players can opt into a presentation cap
# (for example PY3D_FPS=144) or driver/display synchronization independently.
# Neither option is used to conceal slow frames in the renderer.
FPS = max(0, _env_int("PY3D_FPS", 0))
VSYNC = _env_bool("PY3D_VSYNC", False)
# Long stalls (loading, breakpoints, window moves) must not become giant
# simulation steps on the next frame. Keep the limit configurable for tuning.
MAX_FRAME_DT = max(1.0 / 240.0, _env_float("PY3D_MAX_FRAME_DT", 0.1))
MUTE = False

# Performance logging.
# Enable in config, press F3 at runtime, or launch with:
#   $env:PY3D_PERF_LOG="1"; py src/main.py
PERFORMANCE_LOGGING = _env_bool("PY3D_PERF_LOG", False)
PERFORMANCE_LOG_INTERVAL = float(_os.getenv("PY3D_PERF_INTERVAL", "15.0"))
PERFORMANCE_LOG_TOP = int(_os.getenv("PY3D_PERF_TOP", "20"))
PERFORMANCE_LOG_WARMUP_FRAMES = int(_os.getenv("PY3D_PERF_WARMUP", "20"))
PERFORMANCE_SETUP_TIMING = _env_bool("PY3D_SETUP_TIMING", PERFORMANCE_LOGGING)

# Scene hook exception handling.
# By default, the engine logs scene event/dispose failures and continues. Enable
# this while developing to make those failures crash at the source.
RERAISE_SCENE_EXCEPTIONS = _env_bool("PY3D_RERAISE_SCENE_EXCEPTIONS", False)
