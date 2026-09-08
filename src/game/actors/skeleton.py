"""Undead enemies sharing the goblin's roaming, collision, and pose batching."""

from game.actors.goblin import Goblin, GoblinRenderBatch
from game.actors.skeleton_model import (
    SKELETON_MODEL_DEFAULT_HEIGHT,
    SKELETON_MODEL_POSE_COUNT,
    SKELETON_MODEL_VARIANT_COUNT,
    cached_skeleton_local_pose_vertex_data,
    prewarm_skeleton_pose_cache,
    skeleton_model_pose_index,
    transformed_cached_skeleton_vertex_data,
)
from game.inventory import InventoryItem, ItemType
from game.gameplay.rendering.receivers import SKELETON_LIGHTING_RECEIVER


class Skeleton(Goblin):
    """A taller, slower undead creature with its own geometry and combat stats."""

    DISPLAY_NAME = "Skeleton"
    DEFAULT_HEIGHT = SKELETON_MODEL_DEFAULT_HEIGHT
    DEFAULT_MOVE_SPEED = 27.0
    DEFAULT_MAX_HP = 7
    DEFAULT_ATTACK_DAMAGE = 2
    MODEL_VARIANT_COUNT = SKELETON_MODEL_VARIANT_COUNT
    WALK_CYCLE_DISTANCE = 38.0
    packet_lighting_receiver = SKELETON_LIGHTING_RECEIVER
    transformed_pose_vertex_data = staticmethod(transformed_cached_skeleton_vertex_data)
    pose_index = staticmethod(skeleton_model_pose_index)

    def __init__(self, position, **kwargs) -> None:
        kwargs.setdefault("model_height", self.DEFAULT_HEIGHT)
        kwargs.setdefault("move_speed", self.DEFAULT_MOVE_SPEED)
        kwargs.setdefault("max_hp", self.DEFAULT_MAX_HP)
        kwargs.setdefault("sound_keys", ())
        kwargs.setdefault("shadow_size", (26.0, 21.0))
        super().__init__(position, **kwargs)

    def combat_rewards(self, scene) -> tuple[InventoryItem, ...]:
        return (InventoryItem(
            "Ancient bone", ItemType.MISC,
            "A weathered bone left behind by a defeated skeleton.",
        ),)


class SkeletonRenderBatch(GoblinRenderBatch):
    """Reuse the GPU pose atlas with skeleton-specific geometry and lighting."""

    packet_lighting_receiver = SKELETON_LIGHTING_RECEIVER
    pose_count = SKELETON_MODEL_POSE_COUNT
    cached_pose_vertex_data = staticmethod(cached_skeleton_local_pose_vertex_data)
    prewarm_pose_cache = staticmethod(prewarm_skeleton_pose_cache)
