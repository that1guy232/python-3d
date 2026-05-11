"""Engine runtime configuration and environment-variable feature flags."""

from __future__ import annotations

import os as _os


def _env_bool(name: str, default: bool = False) -> bool:
    value = _os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


WIDTH = 1600
HEIGHT = 900
FULLSCREEN = False
VIEWDISTANCE = 2000
FOGDENSITY = 0.0005
LIGHT_BLUE = (0.3, 0.6, 1.0, 1.0)
FOV = 90
FPS = 61
VSYNC = False
MUTE = False

# Performance logging.
# Enable in config, press F3 at runtime, or launch with:
#   $env:PY3D_PERF_LOG="1"; py src/main.py
PERFORMANCE_LOGGING = _env_bool("PY3D_PERF_LOG", False)
PERFORMANCE_LOG_INTERVAL = float(_os.getenv("PY3D_PERF_INTERVAL", "15.0"))
PERFORMANCE_LOG_TOP = int(_os.getenv("PY3D_PERF_TOP", "20"))
PERFORMANCE_LOG_WARMUP_FRAMES = int(_os.getenv("PY3D_PERF_WARMUP", "20"))
PERFORMANCE_SETUP_TIMING = _env_bool("PY3D_SETUP_TIMING", PERFORMANCE_LOGGING)
