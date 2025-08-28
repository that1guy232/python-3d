import math
import numpy as np
from pygame.math import Vector3
from config import WIDTH, HEIGHT
import pygame


class Camera:
    def __init__(self, position=None, rotation=None, width=800, height=600, fov=75, default_brightness=0.1):
        # keep external API types the same (pygame.Vector3)
        self.position = position or Vector3(0, 0, -300)
        self.rotation = rotation or Vector3(0, 0, 0)  # pitch (x), yaw (y), roll (z)
        self.speed = 5

        # Area-based brightness: a list of circular areas (center, radius, value).
        # If no areas are defined, brightness defaults to `brightness_default` (0.0).
        # Access the current camera brightness with the `brightness` property below.
        self.brightness_areas = []
        self.brightness_default = default_brightness

        # FOV scale (same formula you used)
        self._fov_scale = (height / 2) / math.tan(math.radians(fov / 2))

        # cached trig & direction vectors (kept as Vector3 where movement uses them)
        self._yaw_cos = 1.0
        self._yaw_sin = 0.0
        self._pitch_cos = 1.0
        self._pitch_sin = 0.0


        
        # NumPy rotation matrix (world -> camera)
        self._R = np.eye(3, dtype=np.float64)

        # Manual height offset (adjustable by user for screenshots, added on top of
        # the terrain-following camera Y). WorldScene will add this when computing
        # the target camera height.
        self.manual_height_offset = 0.0

        # Speed used when adjusting manual height offset with Q/E (world units/sec).
        # Chosen to be relatively small so Q/E can be used for fine screenshot tweaks.
        self.height_adjust_speed = 50.0

        # OPTIMIZATION: Cache for brightness lookups
        self._brightness_cache = {}
        self._brightness_cache_size = 1000  # Maximum cache size
        self._brightness_cache_hits = 0
        self._brightness_cache_misses = 0
        
        # OPTIMIZATION: Precomputed squared radii to avoid sqrt in distance checks
        self._brightness_areas_optimized = []

        # initialize
        self.update_rotation(0)

    # OPTIMIZATION: Brightness area helpers with caching and precomputation
    def add_brightness_area(self, center, radius, value, falloff):
        """
        Add a circular brightness area.

        center: pygame.Vector3 or (x,y,z) tuple. Only x/z are used for containment.
        radius: float, world-space radius around center (uses x/z plane).
        value: float, brightness value to apply inside the area.
        """
        if not isinstance(center, Vector3):
            center = Vector3(*center)
        
        area = {
            "center": center, 
            "radius": float(radius), 
            "value": float(value), 
            "falloff": float(falloff)
        }
        self.brightness_areas.append(area)
        
        # OPTIMIZATION: Precompute squared radius and store optimized version
        optimized_area = {
            "center_x": center.x,
            "center_z": center.z,
            "radius_squared": float(radius) ** 2,
            "radius": float(radius),  # Keep original radius for distance calculations
            "value": float(value),
            "falloff": float(falloff),
            "original": area  # Keep reference to original for compatibility
        }
        self._brightness_areas_optimized.append(optimized_area)
        
        # Clear cache when areas change
        self._brightness_cache.clear()

    def clear_brightness_areas(self):
        """Remove all defined brightness areas."""
        self.brightness_areas.clear()
        self._brightness_areas_optimized.clear()
        self._brightness_cache.clear()

    def _get_cache_key(self, point):
        """Generate a cache key for a point (rounded to reduce cache size)."""
        # Round to nearest 10 units to create cache buckets
        return (round(point.x / 10) * 10, round(point.z / 10) * 10)

    def get_brightness_at_with_blending(self, point):
        """
        Return the brightness at the given world-space point with proper area blending.
        This version properly combines overlapping brightness areas by adding their contributions.
        """
        if not self._brightness_areas_optimized:
            return self.brightness_default

        # Check cache first
        cache_key = self._get_cache_key(point)
        if cache_key in self._brightness_cache:
            self._brightness_cache_hits += 1
            cached_result = self._brightness_cache[cache_key]
            return cached_result.get("brightness_blended", self.brightness_default) if cached_result else self.brightness_default

        self._brightness_cache_misses += 1

        # Calculate blended brightness from all overlapping areas
        point_x, point_z = point.x, point.z
        total_brightness = self.brightness_default
        contributing_areas = []
        
        # Find all areas that contain this point and calculate their contributions
        for area in self._brightness_areas_optimized:
            dx = point_x - area["center_x"]
            dz = point_z - area["center_z"]
            dist_sq = dx * dx + dz * dz
            
            if dist_sq <= area["radius_squared"]:
                # Calculate distance-based falloff within the area
                radius = area["radius"]
                falloff = area.get("falloff", 1.0)
                value = area["value"]
                
                if radius > 1e-12:
                    # Calculate normalized distance (0 at center, 1 at edge)
                    distance = math.sqrt(dist_sq)
                    norm_distance = min(1.0, distance / radius)
                    
                    # Apply falloff (higher falloff = sharper edge)
                    attenuation = (1.0 - norm_distance) ** max(falloff, 0.0)
                    
                    # Calculate this area's contribution
                    contribution = value * attenuation
                    contributing_areas.append({
                        "area": area,
                        "contribution": contribution,
                        "attenuation": attenuation
                    })

        # Blend contributions from all overlapping areas
        if contributing_areas:
            # Method 1: Additive blending (areas add together)
            # This is good for light sources that should combine
            total_contribution = sum(ca["contribution"] for ca in contributing_areas)
            total_brightness = self.brightness_default + total_contribution
            

        # Cache the result with all the data we computed
        best_area = max(contributing_areas, key=lambda x: x["contribution"])["area"] if contributing_areas else None
        result = {
            "brightness_blended": total_brightness,
            "brightness": max(ca["area"]["value"] for ca in contributing_areas) if contributing_areas else self.brightness_default,
            "area": best_area,
            "contributing_areas": contributing_areas
        }
        self._cache_result(cache_key, result)
        
        return total_brightness

    def get_brightness_at(self, point):
        """
        Return the brightness at the given world-space point (pygame.Vector3).
        FIXED: Now uses proper blending for overlapping areas.
        """
        return self.get_brightness_at_with_blending(point)

    def get_brightness_area_with_blending(self, point):
        """
        Get detailed brightness information at a point including all contributing areas.
        Returns a dict with blended brightness and list of contributing areas.
        """
        if not self._brightness_areas_optimized:
            return None

        # Check cache first
        cache_key = self._get_cache_key(point)
        if cache_key in self._brightness_cache:
            self._brightness_cache_hits += 1
            cached_result = self._brightness_cache[cache_key]
            if cached_result and "contributing_areas" in cached_result:
                return {
                    "brightness": cached_result["brightness_blended"],
                    "contributing_areas": cached_result["contributing_areas"],
                    "primary_area": cached_result["area"]["original"] if cached_result["area"] else None
                }
            return None

        # Force calculation by calling get_brightness_at_with_blending
        brightness = self.get_brightness_at_with_blending(point)
        
        # Get the cached result that was just computed
        cached_result = self._brightness_cache.get(cache_key)
        if cached_result and "contributing_areas" in cached_result:
            return {
                "brightness": brightness,
                "contributing_areas": cached_result["contributing_areas"],
                "primary_area": cached_result["area"]["original"] if cached_result["area"] else None
            }
        
        return None

    def get_brightness_area(self, point):
        """
        OPTIMIZED: Uses caching and precomputed distances.
        Returns the primary (strongest) brightness area at this point.
        """
        blend_info = self.get_brightness_area_with_blending(point)
        return blend_info["primary_area"] if blend_info else None

    def _cache_result(self, key, result):
        """Cache a result with size management."""
        # Simple LRU-like behavior: if cache is full, clear it
        if len(self._brightness_cache) >= self._brightness_cache_size:
            # Keep only the most recently accessed items (simple approach)
            self._brightness_cache.clear()
        
        self._brightness_cache[key] = result

    def get_cache_stats(self):
        """Return cache performance statistics for debugging."""
        total = self._brightness_cache_hits + self._brightness_cache_misses
        hit_rate = self._brightness_cache_hits / total if total > 0 else 0
        return {
            "hits": self._brightness_cache_hits,
            "misses": self._brightness_cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._brightness_cache)
        }

    def clear_cache_stats(self):
        """Reset cache statistics."""
        self._brightness_cache_hits = 0
        self._brightness_cache_misses = 0

    # BATCH PROCESSING OPTIMIZATION
    def get_brightness_batch(self, points):
        """
        Get brightness for multiple points at once with proper blending.
        More efficient than calling get_brightness_at() multiple times.
        
        Args:
            points: List of pygame.Vector3 points
            
        Returns:
            List of brightness values corresponding to each point
        """
        if not self._brightness_areas_optimized:
            return [self.brightness_default] * len(points)

        results = []
        for point in points:
            # Use the blending version for batch processing too
            brightness = self.get_brightness_at_with_blending(point)
            results.append(brightness)

        return results

    @property
    def brightness(self):
        """Convenience property: brightness at the camera's current position."""
        return self.get_brightness_at(self.position)


    def update_rotation(self, dt):
        """Precompute trig, direction vectors (Vector3) and rotation matrix (numpy)."""
        cp = math.cos(self.rotation.x)
        sp = math.sin(self.rotation.x)
        cy = math.cos(self.rotation.y)
        sy = math.sin(self.rotation.y)

        self._yaw_cos = cy
        self._yaw_sin = sy
        self._pitch_cos = cp
        self._pitch_sin = sp

        # Right vector (matches your previous expression)
        self._right = Vector3(self._yaw_cos, 0, -self._yaw_sin)
        # Full forward vector (with vertical component when pitched)
        self._forward = Vector3(
            -self._pitch_cos * self._yaw_sin,
            self._pitch_sin,
            -self._pitch_cos * self._yaw_cos,
        )
        # Ground forward (horizontal component only — keep same formula you used)
        self._ground_forward = -Vector3(-self._yaw_sin, 0, -self._yaw_cos)

        # Build rotation matrices with the same yaw-then-pitch order as your original transform:
        # yaw (around Y) then pitch (around X). Apply yaw first, then pitch.
        Ry = np.array(
            [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], dtype=np.float64
        )

        Rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64
        )

        # world -> camera matrix: pitch * yaw (so application is R @ vector)
        self._R = Rx @ Ry



    def move_camera(self, keys, speed, dt):
        """Movement using precomputed direction vectors (keeps same calls/semantics)."""
        # keys is expected to be pygame.key.get_pressed() or similar mapping
        if keys[pygame.K_w]:
            self.position -= self._ground_forward * speed * dt
        if keys[pygame.K_s]:
            self.position += self._ground_forward * speed * dt
        if keys[pygame.K_a]:
            self.position -= self._right * speed * dt
        if keys[pygame.K_d]:
            self.position += self._right * speed * dt


        


        if keys[pygame.K_q]:
            delta = self.height_adjust_speed * dt
            self.manual_height_offset += delta
        if keys[pygame.K_e]:
            delta = self.height_adjust_speed * dt
            self.manual_height_offset -= delta

        return self.position
