"""Game runtime configuration.

Game modules import from here so world-specific tuning stays out of engine
packages. Display, audio, and profiler defaults are re-exported from
``engine.config`` for convenience.
"""

from engine.config import (  # noqa: F401
    FOGDENSITY,
    FOV,
    FPS,
    FULLSCREEN,
    HEIGHT,
    LIGHT_BLUE,
    MUTE,
    PERFORMANCE_LOG_INTERVAL,
    PERFORMANCE_LOG_TOP,
    PERFORMANCE_LOG_WARMUP_FRAMES,
    PERFORMANCE_LOGGING,
    PERFORMANCE_SETUP_TIMING,
    VIEWDISTANCE,
    VSYNC,
    WIDTH,
)

# Sky clouds
CLOUDS_ENABLED = True
CLOUD_DENSITY = 1.0
CLOUD_SPEED = 1.0
CLOUD_OPACITY = 0.9

STARTING_POS = (200, 600, 200)
BASE_SPEED = 120
SPRINT_SPEED = 180.0
MOUSE_SENSITIVITY = 0.0015
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

# Goblins
GOBLIN_COUNT = 50
GOBLIN_SPAWN_TREE_RADIUS = 130.0
GOBLIN_SPAWN_CLEARANCE = 24.0
GOBLIN_MIN_SEPARATION = 90.0
GOBLIN_SPAWN_ATTEMPTS = 90
GOBLIN_CHASE_RADIUS = 260.0
GOBLIN_CHASE_GIVE_UP_RADIUS = 380.0
GOBLIN_BATTLE_TRIGGER_DISTANCE = 42.0
GOBLIN_BATTLE_LOOK_SMOOTH_HZ = 5.5
