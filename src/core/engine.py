"""Core engine loop & orchestration.

Separates concerns:
- Engine: sets up window, GL state, main loop.
- Scene: holds world assets & update/draw logic (can have multiple later).
- Overlay: simple 2D overlays (FPS etc.).

Keeps compatibility with legacy fixed-function pipeline for now; easy to swap
in shader pipeline later.
"""

from __future__ import annotations

import pygame
from OpenGL.GL import (
    glEnable,
    glDisable,
    glClearColor,
    glDepthFunc,
    GL_DEPTH_TEST,
    GL_LEQUAL,
    GL_CULL_FACE,
)

from config import *
from core.performance import PerformanceLogger
from ui.text_renderer import TextRenderer


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, *, initial_scene=None, initial_scene_factory=None):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        pygame.display.set_caption("3D Pygame")
        # Build flags once and pass an explicit vsync value to ensure it's
        # disabled when config.VSYNC is False. Some older pygame builds don't
        # accept the vsync kwarg, so fall back to the older call signature.
        flags = pygame.DOUBLEBUF | pygame.OPENGL
        if FULLSCREEN:
            flags |= pygame.FULLSCREEN
        try:
            # vsync: 1 to enable, 0 to disable
            pygame.display.set_mode((WIDTH, HEIGHT), flags, vsync=(1 if VSYNC else 0))
        except (TypeError, pygame.error):
            # Older pygame versions won't accept the vsync kwarg, or vsync
            # was requested but unavailable on this system/driver. Fall back
            # to the older call signature without vsync.
            pygame.display.set_mode((WIDTH, HEIGHT), flags)
        self.clock = pygame.time.Clock()

        # GL state
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDisable(GL_CULL_FACE)
        glClearColor(*LIGHT_BLUE)

        self.text = TextRenderer(WIDTH, HEIGHT)
        self.profiler = PerformanceLogger(
            enabled=PERFORMANCE_LOGGING,
            report_interval=PERFORMANCE_LOG_INTERVAL,
            top_n=PERFORMANCE_LOG_TOP,
            warmup_frames=PERFORMANCE_LOG_WARMUP_FRAMES,
        )

        if initial_scene is None:
            if initial_scene_factory is None:
                raise ValueError("Engine requires an initial_scene or initial_scene_factory")
            initial_scene = initial_scene_factory()

        # Active scene (owns camera & input)
        self.scene = initial_scene
        self._attach_profiler(self.scene)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    @staticmethod
    def _dispose_scene(scene) -> None:
        dispose = getattr(scene, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass

    def _attach_profiler(self, scene) -> None:
        setattr(scene, "profiler", self.profiler)
        target_scene = getattr(scene, "target_scene", None)
        if target_scene is not None:
            setattr(target_scene, "profiler", self.profiler)

    def _is_over_any_mesh(self, pos):
        # Kept for compatibility if needed elsewhere; scene now owns bounds
        if hasattr(self.scene, "contains_horizontal"):
            return self.scene.contains_horizontal(pos)
        return True

    # ------------------------------------------------------------------
    def handle_events(self, dt: float) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                self.profiler.toggle()
                continue
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F4:
                self.profiler.reset()
                if self.profiler.enabled:
                    print("[perf] Performance stats reset.")
                continue
            # Forward events to the active scene
            if hasattr(self.scene, "handle_event"):
                try:
                    self.scene.handle_event(event)
                except Exception:
                    pass
        # mouse look (update target rotation; smoothing applied in update())
        mdx, mdy = pygame.mouse.get_rel()
        # Forward mouse delta and frame dt so handlers can normalize by frame time
        if hasattr(self.scene, "apply_mouse_delta"):
            try:
                self.scene.apply_mouse_delta(mdx, mdy, dt)
            except TypeError:
                # Backwards compatible: older apply_mouse_delta may not accept dt
                self.scene.apply_mouse_delta(mdx, mdy)
        return True

    # ------------------------------------------------------------------
    def update(self, dt: float):
        # Scene owns all gameplay updates
        self.scene.update(dt)
        next_scene = getattr(self.scene, "next_scene", None)
        if next_scene is not None:
            old_scene = self.scene
            self.scene = next_scene
            self._attach_profiler(self.scene)
            self._dispose_scene(old_scene)
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            pygame.mouse.get_rel()

    # ------------------------------------------------------------------
    def render(self):  # pragma: no cover - visual
        # Delegate full render to the scene; pass HUD renderer
        fps_val = self.clock.get_fps()
        if hasattr(self.scene, "render"):
            with self.profiler.section("render.scene"):
                self.scene.render(show_hud=True, text=self.text, fps=fps_val)
        with self.profiler.section("render.flip"):
            pygame.display.flip()

    # ------------------------------------------------------------------
    def run(self):  # pragma: no cover - visual
        running = True
        while running:
            # If VSYNC is disabled we don't want to artificially cap FPS here;
            # calling tick() with no framerate argument returns elapsed ms
            # without sleeping. If VSYNC is enabled, keep the configured FPS
            # cap as a safety net for systems/drivers that don't honor vsync.
            if not VSYNC:
                dt = self.clock.tick() / 1000.0
            else:
                dt = self.clock.tick(FPS) / 1000.0
            self.profiler.begin_frame()
            with self.profiler.section("engine.events"):
                running = self.handle_events(dt)
            if running:
                with self.profiler.section("engine.update"):
                    self.update(dt)
                with self.profiler.section("engine.render"):
                    self.render()
            self.profiler.end_frame()
            if not running:
                break
        self._dispose_scene(self.scene)
        pygame.quit()

    # ------------------------------------------------------------------
