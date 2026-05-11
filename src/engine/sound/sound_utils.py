"""Sound loading and playback utilities for pygame.mixer.

This mirrors the style of `texture_utils.py`: keep a tiny registry so the rest
of the code can reference sounds by key without juggling file paths or Sound
objects.

Usage:

    from engine.sound.sound_utils import Sounds

    Sounds.ensure_init()  # safe to call many times
    Sounds.load_optional("footstep", "sounds/footstep.wav", volume=0.4)
    Sounds.play("footstep")

All operations fail gracefully if the mixer can't initialize or a file is
missing; errors are printed once and calls become no-ops.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional
import pygame
from engine.config import MUTE


class Sounds:
    """Static manager for loading and playing short SFX.

    Notes
    -----
    - Initializes pygame.mixer lazily on first use.
    - Stores sounds by a string key (e.g., "footstep").
    - `play()` uses a free channel (reserving one if necessary) and returns it.
    - All methods are safe even if audio isn't available; they just no-op.
    """

    _inited: bool = False
    _failed_init: bool = False
    _sounds: Dict[str, pygame.mixer.Sound] = {}
    _muted: bool = bool(MUTE)

    @classmethod
    def ensure_init(
        cls,
        *,
        frequency: int = 44100,
        size: int = -16,
        channels: int = 2,
        buffer: int = 512,
    ) -> bool:
        """Initialize pygame.mixer if needed. Returns True on success.

        Safe to call multiple times.
        """
        if cls._inited:
            return True
        if cls._failed_init:
            return False
        try:
            # If pygame.init() was called already, mixer may be ready; check first.
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(
                    frequency=frequency, size=size, channels=channels, buffer=buffer
                )
            cls._inited = pygame.mixer.get_init() is not None
            return cls._inited
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[Sounds] Mixer init failed: {e}")
            cls._failed_init = True
            return False

    @classmethod
    def is_available(cls) -> bool:
        """Return True if audio playback should work."""
        return cls._inited and (pygame.mixer.get_init() is not None)

    @classmethod
    def load(
        cls, key: str, path: str, *, volume: Optional[float] = None
    ) -> Optional[pygame.mixer.Sound]:
        """Load a sound file and register it under `key`.

        Raises FileNotFoundError if the path doesn't exist.
        Returns the Sound object on success, or None on failure.
        """
        if not cls.ensure_init():
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        try:
            snd = pygame.mixer.Sound(path)
            if volume is not None:
                snd.set_volume(max(0.0, min(1.0, float(volume))))
            cls._sounds[key] = snd
            # print(f"[Sounds] Loaded '{key}' from {path}")
            return snd
        except Exception as e:  # pragma: no cover - file/codec dependent
            print(f"[Sounds] Failed to load '{key}' from {path}: {e}")
            return None

    @classmethod
    def load_optional(
        cls, key: str, path: str, *, volume: Optional[float] = None
    ) -> Optional[pygame.mixer.Sound]:
        """Load sound if the file exists; otherwise print a note and return None."""
        if not os.path.exists(path):
            print(f"[Sounds] Skipping missing file for '{key}': {path}")
            return None
        return cls.load(key, path, volume=volume)

    @classmethod
    def unload(cls, key: str) -> None:
        cls._sounds.pop(key, None)

    @classmethod
    def clear(cls) -> None:
        cls._sounds.clear()

    @classmethod
    def get(cls, key: str) -> Optional[pygame.mixer.Sound]:
        return cls._sounds.get(key)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _channel_volume_pair(cls, volume) -> tuple[float, float] | None:
        if volume is None:
            return None
        if isinstance(volume, (tuple, list)):
            if len(volume) < 2:
                return None
            return cls._clamp01(volume[0]), cls._clamp01(volume[1])
        v = cls._clamp01(volume)
        return v, v

    @classmethod
    def _apply_channel_volume(cls, channel: pygame.mixer.Channel, volume) -> bool:
        pair = cls._channel_volume_pair(volume)
        if pair is None:
            return False

        left, right = pair
        try:
            channel.set_volume(left, right)
        except TypeError:
            try:
                channel.set_volume(max(left, right))
            except Exception:
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def _point3(point) -> tuple[float, float, float]:
        if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
            return float(point.x), float(point.y), float(point.z)
        return float(point[0]), float(point[1]), float(point[2])

    @classmethod
    def _listener_position(cls, listener) -> tuple[float, float, float]:
        position = getattr(listener, "position", listener)
        return cls._point3(position)

    @classmethod
    def _pan_for_listener(
        cls,
        dx: float,
        dy: float,
        dz: float,
        listener,
    ) -> float:
        horizontal_distance = math.hypot(dx, dz)
        if horizontal_distance <= 1e-6:
            return 0.0

        right_amount = None
        world_delta_to_view = getattr(listener, "world_delta_to_view", None)
        if callable(world_delta_to_view):
            try:
                right_amount = float(world_delta_to_view(dx, dy, dz)[0])
            except Exception:
                right_amount = None

        if right_amount is None:
            right = getattr(listener, "_right", None)
            if right is None:
                right = getattr(listener, "right", None)
            if right is not None:
                try:
                    right_amount = dx * float(right.x) + dz * float(right.z)
                except Exception:
                    right_amount = None

        if right_amount is None:
            return 0.0
        return max(-1.0, min(1.0, right_amount / horizontal_distance))

    @classmethod
    def spatial_volumes(
        cls,
        source_position,
        listener,
        *,
        audible_radius: float,
        full_volume_radius: float = 0.0,
        max_volume: float = 1.0,
        pan_strength: float = 0.85,
        min_volume: float = 0.01,
    ) -> tuple[float, float] | None:
        """Return stereo channel volumes for a positioned sound source."""
        try:
            sx, sy, sz = cls._point3(source_position)
            lx, ly, lz = cls._listener_position(listener)
        except Exception:
            return None

        dx = sx - lx
        dy = sy - ly
        dz = sz - lz
        distance_sq = dx * dx + dy * dy + dz * dz
        audible_radius = max(0.0, float(audible_radius))
        if audible_radius <= 0.0 or distance_sq >= audible_radius * audible_radius:
            return None

        distance = math.sqrt(distance_sq)
        full_radius = max(0.0, min(audible_radius, float(full_volume_radius)))
        if distance <= full_radius:
            attenuation = 1.0
        else:
            fade_distance = max(1.0, audible_radius - full_radius)
            t = min(1.0, max(0.0, (distance - full_radius) / fade_distance))
            attenuation = (1.0 - t) * (1.0 - t)

        volume = cls._clamp01(float(max_volume) * attenuation)
        if volume < max(0.0, float(min_volume)):
            return None

        pan = cls._pan_for_listener(dx, dy, dz, listener)
        pan *= cls._clamp01(pan_strength)
        left = volume * (1.0 - max(0.0, pan))
        right = volume * (1.0 + min(0.0, pan))
        return cls._clamp01(left), cls._clamp01(right)

    @classmethod
    def set_channel_spatial(
        cls,
        channel: pygame.mixer.Channel,
        source_position,
        listener,
        *,
        audible_radius: float,
        full_volume_radius: float = 0.0,
        max_volume: float = 1.0,
        pan_strength: float = 0.85,
        min_volume: float = 0.01,
    ) -> bool:
        """Update an existing channel with distance falloff and stereo panning."""
        volumes = cls.spatial_volumes(
            source_position,
            listener,
            audible_radius=audible_radius,
            full_volume_radius=full_volume_radius,
            max_volume=max_volume,
            pan_strength=pan_strength,
            min_volume=min_volume,
        )
        if volumes is None:
            return False
        return cls._apply_channel_volume(channel, volumes)

    @classmethod
    def play(
        cls,
        key: str,
        *,
        volume: Optional[float | tuple[float, float]] = None,
        loops: int = 0,
        maxtime: int = 0,
        fade_ms: int = 0,
    ) -> Optional[pygame.mixer.Channel]:
        """Play a registered sound by key.

        Parameters
        ----------
        key : str
            The registry key used in `load`/`load_optional`.
        volume : Optional[float | tuple[float, float]]
            If provided, temporarily sets mono or left/right volume for this playback.
        loops : int
            Number of extra times to repeat after the first play. 0 = play once.
        maxtime : int
            Stop playback after this many milliseconds (0 = no limit).
        fade_ms : int
            Fade-in time in milliseconds.
        """
        if cls._muted:
            return None
        if not cls.is_available():
            return None
        snd = cls._sounds.get(key)
        if snd is None:
            # Soft failure to keep game running.
            # Print only once per missing key to avoid spam.
            if getattr(cls, "_missing_warned", None) is None:
                cls._missing_warned = set()
            if key not in cls._missing_warned:
                print(f"[Sounds] Warning: sound '{key}' not loaded")
                cls._missing_warned.add(key)
            return None

        # Use per-channel volume so this playback respects `volume` without
        # permanently altering the Sound object's base volume.
        ch = pygame.mixer.find_channel(True)
        if ch is None:
            return None
        cls._apply_channel_volume(ch, 1.0 if volume is None else volume)
        ch.play(snd, loops=loops, maxtime=maxtime, fade_ms=fade_ms)
        return ch

    @classmethod
    def play_at(
        cls,
        key: str,
        source_position,
        listener,
        *,
        audible_radius: float,
        full_volume_radius: float = 0.0,
        max_volume: float = 1.0,
        pan_strength: float = 0.85,
        min_volume: float = 0.01,
        loops: int = 0,
        maxtime: int = 0,
        fade_ms: int = 0,
    ) -> Optional[pygame.mixer.Channel]:
        """Play a registered sound from a world position relative to a listener."""
        volumes = cls.spatial_volumes(
            source_position,
            listener,
            audible_radius=audible_radius,
            full_volume_radius=full_volume_radius,
            max_volume=max_volume,
            pan_strength=pan_strength,
            min_volume=min_volume,
        )
        if volumes is None:
            return None
        return cls.play(
            key,
            volume=volumes,
            loops=loops,
            maxtime=maxtime,
            fade_ms=fade_ms,
        )

    @classmethod
    def stop(cls, key: str) -> None:
        snd = cls._sounds.get(key)
        if snd is None or not cls.is_available():
            return
        # Stop any channel currently playing this Sound
        for ch_idx in range(pygame.mixer.get_num_channels()):
            ch = pygame.mixer.Channel(ch_idx)
            if ch.get_sound() == snd:
                ch.stop()

    @classmethod
    def set_volume(cls, key: str, volume: float) -> None:
        snd = cls._sounds.get(key)
        if snd is not None:
            snd.set_volume(max(0.0, min(1.0, float(volume))))

    @classmethod
    def set_playing_volume(
        cls,
        key: str,
        volume: float | tuple[float, float],
    ) -> bool:
        """Set the channel volume for every active playback of a sound."""
        snd = cls._sounds.get(key)
        if snd is None or not cls.is_available():
            return False

        changed = False
        for ch_idx in range(pygame.mixer.get_num_channels()):
            ch = pygame.mixer.Channel(ch_idx)
            if ch.get_sound() == snd and ch.get_busy():
                changed = cls._apply_channel_volume(ch, volume) or changed
        return changed

    @classmethod
    def is_loaded(cls, key: str) -> bool:
        return key in cls._sounds

    @classmethod
    def is_playing(cls, key: str) -> bool:
        snd = cls._sounds.get(key)
        if snd is None:
            return False
        # Check if the sound is currently playing on any channel
        for ch_idx in range(pygame.mixer.get_num_channels()):
            ch = pygame.mixer.Channel(ch_idx)
            if ch.get_sound() == snd and ch.get_busy():
                return True
        return False

    @classmethod
    def is_muted(cls) -> bool:
        return cls._muted

    @classmethod
    def set_muted(cls, muted: bool) -> None:
        cls._muted = bool(muted)
        if cls._muted and cls.is_available():
            try:
                pygame.mixer.stop()
            except Exception:
                pass

    @classmethod
    def toggle_muted(cls) -> bool:
        cls.set_muted(not cls._muted)
        return cls._muted


__all__ = ["Sounds"]
