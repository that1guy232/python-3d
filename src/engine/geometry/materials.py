"""Default receiver policies for reusable geometry; IDs remain stable across scenes."""

from engine.rendering.lighting.receiver import (
    LightingEvaluation,
    LightingReceiver,
    LocalLightPolicy,
)


TEXTURED_STATIC_WALL_LIGHTING_RECEIVER = LightingReceiver(
    receiver_id="world.static_wall.textured",
    directional=True,
    local=True,
    environment=False,
    exposure=True,
    fog=True,
    shine=False,
    point=True,
    clamp_lit_material=True,
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
