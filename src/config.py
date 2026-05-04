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
LIGHT_BLUE = (0.7, 0.8, 1.0, 1.0)
FOV = 90
STARTING_POS = (200, 30, 200)
FPS = 61
VSYNC = False
BASE_SPEED = 120
SPRINT_SPEED = 180.0
MOUSE_SENSITIVITY = 0.0015
MUTE = True
# Camera follow-ground behavior
CAMERA_GROUND_OFFSET = 25.0  # eye height above ground surface
CAMERA_FOLLOW_SMOOTH_HZ = (
    5.0  # responsiveness for easing camera Y to target (0=instant)
)
# Head-bob settings
HEADBOB_ENABLED = True
# How fast the bobbing cycles when walking (in cycles/sec). Sprint scales this up.
HEADBOB_FREQUENCY = .5
# Vertical bob amplitude in world units (tune to taste based on your world scale)
HEADBOB_AMPLITUDE = 4
# Side-to-side sway amplitude in world units
HEADBOB_AMPLITUDE_SIDE = 3
# How much sprinting increases frequency/amplitude
HEADBOB_SPRINT_MULT = 1.25
# How quickly the offset eases back to rest when stopping (larger = snappier)
HEADBOB_DAMPING = 2.0

# Jump physics
JUMP_SPEED = 250.0
GRAVITY = 800.0

# Player collision body. The camera sits near the top of the body, so the
# head clearance only needs to cover the small space above eye height.
PLAYER_RADIUS = 16.0
PLAYER_HEAD_CLEARANCE = 6.0

# Performance logging.
# Enable in config, press F3 at runtime, or launch with:
#   $env:PY3D_PERF_LOG="1"; py src/main.py
PERFORMANCE_LOGGING = _env_bool("PY3D_PERF_LOG", False)
PERFORMANCE_LOG_INTERVAL = float(_os.getenv("PY3D_PERF_INTERVAL", "3.0"))
PERFORMANCE_LOG_TOP = int(_os.getenv("PY3D_PERF_TOP", "14"))
PERFORMANCE_LOG_WARMUP_FRAMES = int(_os.getenv("PY3D_PERF_WARMUP", "20"))
PERFORMANCE_SETUP_TIMING = _env_bool("PY3D_SETUP_TIMING", PERFORMANCE_LOGGING)
