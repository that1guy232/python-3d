"""Encapsulated head-bob logic.

Computes smoothed movement intensity, phase progression, footstep events,
and visual offsets. Visual offsets are zeroed when disabled, but timing and
footstep events still run so audio/hooks continue to work.
"""

from __future__ import annotations

import math
from typing import Callable, Optional


class HeadBob:
    def __init__(
        self,
        *,
        enabled: bool,
        frequency: float,
        amplitude_y: float,
        amplitude_x: float,
        sprint_mult: float,
        damping: float,
        on_footstep: Optional[Callable[[float, bool, float, str], None]] = None,
    ) -> None:
        self.enabled = enabled
        self.frequency = frequency
        self.amplitude_y = amplitude_y
        self.amplitude_x = amplitude_x
        self.sprint_mult = sprint_mult
        self.damping = max(1.0, damping)
        self.on_footstep = on_footstep

        # Internal state
        self._phase = 0.0  # cycles 0..1
        self._prev_phase = 0.0
        self._intensity = 0.0  # 0..1
        self._off_x = 0.0
        self._off_y = 0.0

        # Idle-sway state: camera gentle vertical bob after inactivity.
        # This centralizes idle handling here so callers don't need duplicate
        # logic elsewhere.
        self._mouse_moved = False
        self._any_key_down = False
        self._idle_elapsed = 0.0
        self._idle_threshold = 1.0
        # small sway (gentle drift) plus a slower "breathing" oscillation
        self._idle_phase = 0.0
        self._idle_frequency = 0.4
        self._idle_amplitude = 0.35
        # breathing is a separate, slower vertical motion to simulate chest
        # breathing when the player is idle for a while.
        self._idle_breath_phase = 0.0
        self._idle_breath_frequency = 0.25
        self._idle_breath_amplitude = 1.0
        self._idle_offset_y = 0.0
        self._idle_smooth_hz = 3.0

    @staticmethod
    def _phase_crossed(prev: float, curr: float, target: float) -> bool:
        """Return True if the phase wrapped past `target` between prev->curr."""
        if prev <= curr:
            return prev < target <= curr
        # wrapped around 1->0
        return prev < target or target <= curr

    def notify_mouse_moved(self) -> None:
        """Inform the headbob that the mouse moved this frame."""
        try:
            self._mouse_moved = True
        except Exception:
            pass

    def notify_input_active(self, any_key_down: bool) -> None:
        """Inform the headbob whether any keyboard input is down this frame."""
        try:
            self._any_key_down = bool(any_key_down)
        except Exception:
            pass

    def update(self, *, moving: bool, sprinting: bool, dt: float) -> None:
        # Smooth intensity towards 1 when moving else 0
        target_intensity = 1.0 if moving else 0.0
        self._intensity += (target_intensity - self._intensity) * min(
            1.0, self.damping * dt
        )

        # Advance phase proportionally to intensity (keeps some motion near stop)
        freq = self.frequency * (self.sprint_mult if sprinting else 1.0)
        phase_speed = freq * max(0.2, self._intensity)
        self._prev_phase = self._phase
        self._phase = (self._phase + phase_speed * dt) % 1.0

        # Compute target offsets (local cam space). Slightly boost on sprint.
        amp_scale = 1.0 + (0.25 if sprinting else 0.0)
        target_y = (
            math.sin(self._phase * 2.0 * math.pi)
            * self.amplitude_y
            * self._intensity
            * amp_scale
        )
        target_x = (
            math.sin(self._phase * 2.0 * math.pi + math.pi / 2.0)
            * self.amplitude_x
            * 0.6
            * self._intensity
            * amp_scale
        )

        # Ease offsets when enabled; keep at rest when disabled (but still compute phase)
        if self.enabled:
            lerp_amt = min(1.0, self.damping * dt)
            self._off_y += (target_y - self._off_y) * lerp_amt
            self._off_x += (target_x - self._off_x) * lerp_amt
        else:
            self._off_y = 0.0
            self._off_x = 0.0

        # Footstep trigger around peaks
        if self.on_footstep and moving and self._intensity > 0.2:
            hit_top = self._phase_crossed(self._prev_phase, self._phase, 0.25)
            hit_bottom = self._phase_crossed(self._prev_phase, self._phase, 0.75)
            if hit_top or hit_bottom:
                foot = "left" if hit_top else "right"
                try:
                    self.on_footstep(self._intensity, sprinting, self._phase, foot)
                except Exception:
                    # Keep main loop resilient to hook failures
                    pass

        # Idle camera sway: starts when no keyboard or mouse input for a while.
        try:
            if not getattr(self, "_any_key_down", False) and not getattr(
                self, "_mouse_moved", False
            ):
                self._idle_elapsed += dt
            else:
                self._idle_elapsed = 0.0

            idle_active = self._idle_elapsed >= self._idle_threshold
            if idle_active:
                # regular small idle sway
                self._idle_phase += (2.0 * math.pi * self._idle_frequency) * dt
                sway_y = math.sin(self._idle_phase) * self._idle_amplitude
                # slower breathing motion
                self._idle_breath_phase += (
                    2.0 * math.pi * self._idle_breath_frequency
                ) * dt
                breath_y = (
                    math.sin(self._idle_breath_phase) * self._idle_breath_amplitude
                )
                # combine sway + breathing for final target
                target_idle_y = sway_y + breath_y
            else:
                target_idle_y = 0.0

            if dt > 0.0:
                a_idle = 1.0 - math.exp(-self._idle_smooth_hz * dt)
                self._idle_offset_y += (target_idle_y - self._idle_offset_y) * a_idle
        except Exception:
            # Keep headbob idle update resilient
            pass
        finally:
            # Reset per-frame mouse activity flag; input handler will set it
            try:
                self._mouse_moved = False
            except Exception:
                pass

    def offsets(self) -> tuple[float, float]:
        # Combine regular headbob offsets with idle vertical offset.
        return self._off_x, self._off_y + getattr(self, "_idle_offset_y", 0.0)

    # Expose read-only properties for convenience
    @property
    def intensity(self) -> float:
        return self._intensity

    @property
    def phase(self) -> float:
        return self._phase


__all__ = ["HeadBob"]
