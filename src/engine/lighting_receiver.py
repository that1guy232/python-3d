"""Explicit lighting-channel contracts for render receivers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReceiverCompatibilityError(ValueError):
    """A receiver contract cannot be represented by the legacy shader."""


class LightingEvaluation(str, Enum):
    """Where a receiver's declared lighting channels are evaluated."""

    DYNAMIC = "dynamic"
    CPU_BAKED = "cpu_baked"
    FIXED_FUNCTION = "fixed_function"


class LocalLightPolicy(str, Enum):
    """How local-light metadata is interpreted for a receiver."""

    SURFACE = "surface"
    POINT_QUERY = "point_query"


@dataclass(frozen=True, slots=True)
class ReceiverShaderFlags:
    """Compatibility-shader flags projected from a receiver contract."""

    scene_lighting: bool
    directional: bool
    environment: bool
    fog: bool
    shine: bool


@dataclass(frozen=True, slots=True)
class LightingReceiver:
    """Declare which lighting and adjacent render channels a surface receives."""

    receiver_id: str
    directional: bool
    local: bool
    environment: bool
    exposure: bool
    fog: bool
    shine: bool
    point: bool = False
    evaluation: LightingEvaluation = LightingEvaluation.DYNAMIC
    clamp_directional_material: bool = False
    clamp_lit_material: bool = False
    local_light_policy: LocalLightPolicy = LocalLightPolicy.SURFACE

    def __post_init__(self) -> None:
        if not str(self.receiver_id).strip():
            raise ValueError("lighting receiver requires a stable receiver_id")
        if not isinstance(self.evaluation, LightingEvaluation):
            raise TypeError("lighting receiver evaluation must be LightingEvaluation")
        if not isinstance(self.local_light_policy, LocalLightPolicy):
            raise TypeError(
                "lighting receiver local_light_policy must be LocalLightPolicy"
            )

    def compatibility_shader_flags(
        self,
        *,
        has_normals: bool,
    ) -> ReceiverShaderFlags:
        """Project this contract into flags supported by the current shader."""

        if (
            self.evaluation is LightingEvaluation.DYNAMIC
            and self.local != self.exposure
        ):
            raise ReceiverCompatibilityError(
                f"receiver {self.receiver_id!r} separates local light and exposure, "
                "but the compatibility shader couples both channels"
            )
        if (
            self.evaluation is LightingEvaluation.DYNAMIC
            and self.local
            and self.local_light_policy is not LocalLightPolicy.SURFACE
        ):
            raise ReceiverCompatibilityError(
                f"receiver {self.receiver_id!r} uses local-light policy "
                f"{self.local_light_policy.value!r}, but the compatibility shader "
                "only supports surface evaluation"
            )
        supports_dynamic_lighting = (
            bool(has_normals)
            and self.evaluation is LightingEvaluation.DYNAMIC
        )
        scene_lighting = (
            supports_dynamic_lighting and self.local and self.exposure
        )
        return ReceiverShaderFlags(
            scene_lighting=scene_lighting,
            directional=supports_dynamic_lighting and self.directional,
            environment=supports_dynamic_lighting and self.environment,
            fog=self.fog,
            shine=bool(has_normals) and self.shine,
        )
