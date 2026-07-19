"""Explicit world receiver policies introduced during lighting migration."""

from engine.lighting_receiver import (
    LightingEvaluation,
    LightingReceiver,
    LocalLightPolicy,
)
from engine.rendering.decal import DECAL_LIGHTING_RECEIVER
from engine.rendering.sprite import (
    DEFAULT_SPRITE_LIGHTING_RECEIVER as SPRITE_LIGHTING_RECEIVER,
)


GROUND_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.ground",
    directional=True,
    local=True,
    environment=True,
    exposure=True,
    fog=True,
    shine=True,
    point=True,
)

TEXTURED_STATIC_WALL_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.static_wall.textured",
    directional=True,
    local=True,
    environment=False,
    exposure=True,
    fog=True,
    shine=True,
    point=True,
)

UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.static_wall.untextured",
    directional=True,
    local=False,
    environment=False,
    exposure=True,
    fog=True,
    shine=False,
    evaluation=LightingEvaluation.CPU_BAKED,
)

ROAD_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.road",
    directional=True,
    local=True,
    environment=True,
    exposure=True,
    fog=True,
    shine=True,
    point=True,
)

FENCE_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.fence",
    directional=True,
    local=True,
    environment=True,
    exposure=True,
    fog=True,
    shine=True,
    point=True,
)

CPU_BAKED_SLAB_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.slab.cpu_baked",
    directional=True,
    local=False,
    environment=False,
    exposure=False,
    fog=True,
    shine=True,
    evaluation=LightingEvaluation.CPU_BAKED,
)

CPU_BAKED_OBJECT_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.object.cpu_baked",
    directional=True,
    local=False,
    environment=False,
    exposure=False,
    fog=True,
    shine=False,
    evaluation=LightingEvaluation.CPU_BAKED,
)

DYNAMIC_SLAB_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.slab.dynamic",
    directional=True,
    local=False,
    environment=False,
    exposure=False,
    fog=True,
    shine=True,
    point=True,
    clamp_directional_material=True,
)

DYNAMIC_OBJECT_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.object.dynamic",
    directional=True,
    local=False,
    environment=False,
    exposure=False,
    fog=True,
    shine=False,
    point=True,
    clamp_directional_material=True,
)

DYNAMIC_POLYGON_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.polygon.dynamic",
    directional=True,
    local=True,
    environment=False,
    exposure=True,
    fog=True,
    shine=False,
    point=True,
    clamp_lit_material=True,
    local_light_policy=LocalLightPolicy.POINT_QUERY,
)

SKY_CLEAR_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.sky.clear",
    directional=False,
    local=False,
    environment=False,
    exposure=True,
    fog=False,
    shine=False,
    evaluation=LightingEvaluation.FIXED_FUNCTION,
)

SKY_SUN_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.sky.sun",
    directional=False,
    local=False,
    environment=False,
    exposure=True,
    fog=False,
    shine=False,
    evaluation=LightingEvaluation.FIXED_FUNCTION,
)

SKY_CLOUD_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.sky.cloud",
    directional=False,
    local=False,
    environment=False,
    exposure=True,
    fog=False,
    shine=False,
    evaluation=LightingEvaluation.FIXED_FUNCTION,
)


# Receivers eagerly packetized for the normal packet renderer. Compatibility
# representations remain available below as explicit rollback-only contracts;
# they must not become packet-runtime dependencies again.
PACKET_RUNTIME_LIGHTING_RECEIVERS = (
    GROUND_LIGHTING_RECEIVER,
    ROAD_LIGHTING_RECEIVER,
    FENCE_LIGHTING_RECEIVER,
    TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
    DYNAMIC_SLAB_LIGHTING_RECEIVER,
    DYNAMIC_OBJECT_LIGHTING_RECEIVER,
    DYNAMIC_POLYGON_LIGHTING_RECEIVER,
    SPRITE_LIGHTING_RECEIVER,
    DECAL_LIGHTING_RECEIVER,
    SKY_CLEAR_LIGHTING_RECEIVER,
    SKY_SUN_LIGHTING_RECEIVER,
    SKY_CLOUD_LIGHTING_RECEIVER,
)

ROLLBACK_ONLY_LIGHTING_RECEIVERS = (
    UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
    CPU_BAKED_SLAB_LIGHTING_RECEIVER,
    CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
)

ALL_WORLD_LIGHTING_RECEIVERS = (
    *PACKET_RUNTIME_LIGHTING_RECEIVERS,
    *ROLLBACK_ONLY_LIGHTING_RECEIVERS,
)

PACKET_RUNTIME_LIGHTING_RECEIVER_IDS = frozenset(
    receiver.receiver_id for receiver in PACKET_RUNTIME_LIGHTING_RECEIVERS
)
ROLLBACK_ONLY_LIGHTING_RECEIVER_IDS = frozenset(
    receiver.receiver_id for receiver in ROLLBACK_ONLY_LIGHTING_RECEIVERS
)
