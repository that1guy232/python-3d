WIDTH = 1600
HEIGHT = 900
FULLSCREEN = False
VIEWDISTANCE = 2500
FOGDENSITY = 0.0007
LIGHT_BLUE = (0.7, 0.8, 1.0, 1.0)
FOV = 90
STARTING_POS = (200, 30, 200)
FPS = 61
VSYNC = False
BASE_SPEED = 60.0
SPRINT_SPEED = 100.0
MOUSE_SENSITIVITY = 0.0015
MUTE = False
# Camera follow-ground behavior
CAMERA_GROUND_OFFSET = 25.0  # eye height above ground surface
CAMERA_FOLLOW_SMOOTH_HZ = (
    10.0  # responsiveness for easing camera Y to target (0=instant)
)
# Head-bob settings
HEADBOB_ENABLED = True
# How fast the bobbing cycles when walking (in cycles/sec). Sprint scales this up.
HEADBOB_FREQUENCY = 1
# Vertical bob amplitude in world units (tune to taste based on your world scale)
HEADBOB_AMPLITUDE = 5
# Side-to-side sway amplitude in world units
HEADBOB_AMPLITUDE_SIDE = 3
# How much sprinting increases frequency/amplitude
HEADBOB_SPRINT_MULT = 1.25
# How quickly the offset eases back to rest when stopping (larger = snappier)
HEADBOB_DAMPING = 2.0
