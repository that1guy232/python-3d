"""Sound loading and playback utilities for pygame.mixer.

This mirrors the style of `texture_utils.py`: keep a tiny registry so the rest
of the code can reference sounds by key without juggling file paths or Sound
objects.

Usage:

    from sound.sound_utils import Sounds

    Sounds.ensure_init()  # safe to call many times
    Sounds.load_optional("footstep", "sounds/footstep.wav", volume=0.4)
    Sounds.play("footstep")

All operations fail gracefully if the mixer can't initialize or a file is
missing; errors are printed once and calls become no-ops.
"""

from __future__ import annotations

from typing import Dict, Optional
import os
import pygame
from config import MUTE


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

    @classmethod
    def play(
        cls,
        key: str,
        *,
        volume: Optional[float] = None,
        loops: int = 0,
        maxtime: int = 0,
        fade_ms: int = 0,
    ) -> Optional[pygame.mixer.Channel]:
        """Play a registered sound by key.

        Parameters
        ----------
        key : str
            The registry key used in `load`/`load_optional`.
        volume : Optional[float]
            If provided, temporarily sets the sound volume (0..1) for this playback.
        loops : int
            Number of extra times to repeat after the first play. 0 = play once.
        maxtime : int
            Stop playback after this many milliseconds (0 = no limit).
        fade_ms : int
            Fade-in time in milliseconds.
        """
        if MUTE:
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
        if volume is not None:
            try:
                v = max(0.0, min(1.0, float(volume)))
                # Set both left/right volumes equally.
                ch.set_volume(v)
            except Exception:
                pass
        ch.play(snd, loops=loops, maxtime=maxtime, fade_ms=fade_ms)
        return ch

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


__all__ = ["Sounds"]
