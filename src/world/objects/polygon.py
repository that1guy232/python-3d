from pygame.math import Vector3
from core.object3d import Object3D
from OpenGL.GL import (
    glEnable,
    glDisable,
    glBindTexture,
    glTexParameteri,
    glBegin,
    glEnd,
    glTexCoord2f,
    glVertex3f,
    glColor3f,
    glColor4f,
    glBlendFunc,
    glAlphaFunc,
    GL_TEXTURE_2D,
    GL_QUADS,
    GL_BLEND,
    GL_ALPHA_TEST,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_GREATER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_REPEAT,
    GL_TRIANGLE_FAN
)
from textures.texture_utils import get_texture_size


class Polygon(Object3D):
    def __init__(self, position=None, rotation=None, points_2d=None, thickness=1.0, texture=None):
        self.points_2d = points_2d
        if len(self.points_2d) < 3:
            raise ValueError("A polygon must have at least 3 points.")
        self.thickness = thickness
        self.texture = texture

        super().__init__(position, rotation)

        self.vertices = self._generate_vertices()
        self.faces = self._generate_faces()
        self.local_vertices = self.vertices


    def _generate_faces(self):
        """Generate faces for a 3D extruded polygon.

        This method triangulates the front and back faces with an ear-clipping
        algorithm so concave polygons (for example an 'L' shape) render
        correctly. Side faces remain quads connecting front and back.
        """
        faces = []
        n_points = len(self.points_2d)

        # Ensure polygon is CCW for consistent triangulation
        def signed_area(points):
            a = 0.0
            for i in range(len(points)):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % len(points)]
                a += (x1 * y2) - (x2 * y1)
            return a * 0.5

        pts2d = list(self.points_2d)
        if signed_area(pts2d) < 0:
            pts2d.reverse()

        # Ear-clipping triangulation on 2D points
        def is_convex(a, b, c):
            # returns True if angle abc is convex (assuming CCW polygon)
            ax, ay = a
            bx, by = b
            cx, cy = c
            return ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) > 0

        def point_in_tri(pt, a, b, c):
            # barycentric technique
            px, py = pt
            ax, ay = a
            bx, by = b
            cx, cy = c
            v0x, v0y = cx - ax, cy - ay
            v1x, v1y = bx - ax, by - ay
            v2x, v2y = px - ax, py - ay
            dot00 = v0x * v0x + v0y * v0y
            dot01 = v0x * v1x + v0y * v1y
            dot02 = v0x * v2x + v0y * v2y
            dot11 = v1x * v1x + v1y * v1y
            dot12 = v1x * v2x + v1y * v2y
            denom = dot00 * dot11 - dot01 * dot01
            if denom == 0:
                return False
            inv = 1.0 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv
            v = (dot00 * dot12 - dot01 * dot02) * inv
            return (u >= 0) and (v >= 0) and (u + v <= 1)

        # index list into pts2d
        idxs = list(range(n_points))
        triangles = []

        # Work on a copy of pts2d for geometric checks
        loop_guard = 0
        while len(idxs) > 3 and loop_guard < 10000:
            ear_found = False
            for i in range(len(idxs)):
                i_prev = idxs[(i - 1) % len(idxs)]
                i_curr = idxs[i]
                i_next = idxs[(i + 1) % len(idxs)]

                a = pts2d[i_prev]
                b = pts2d[i_curr]
                c = pts2d[i_next]

                if not is_convex(a, b, c):
                    continue

                # check no other point is inside triangle abc
                any_inside = False
                for j in idxs:
                    if j in (i_prev, i_curr, i_next):
                        continue
                    if point_in_tri(pts2d[j], a, b, c):
                        any_inside = True
                        break
                if any_inside:
                    continue

                # ear found
                triangles.append([i_prev, i_curr, i_next])
                idxs.pop(i)
                ear_found = True
                break

            if not ear_found:
                # fallback: polygon may be degenerate or numeric issues — break to avoid infinite loop
                break
            loop_guard += 1

        # add the final triangle
        if len(idxs) == 3:
            triangles.append([idxs[0], idxs[1], idxs[2]])

        # Front face triangles (indices 0..n_points-1)
        for tri in triangles:
            faces.append(list(tri))

        # Back face triangles (offset by n_points) - reverse to keep outward normal
        for tri in triangles:
            faces.append([t + n_points for t in reversed(tri)])

        # Side faces (quads) connecting front and back
        for i in range(n_points):
            next_i = (i + 1) % n_points
            face = [
                i,                    # current front vertex
                next_i,               # next front vertex
                next_i + n_points,    # next back vertex
                i + n_points          # current back vertex
            ]
            faces.append(face)

        return faces

    def _generate_vertices(self):
        """
        Generate 3D vertices from 2D points by extruding along Z-axis.
        Creates front face at z=0 and back face at z=-thickness.
        """
        vertices = []
        
        # Front face vertices (z = 0) - FIXED: Using Vector3 instead of lists
        for point in self.points_2d:
            vertices.append(Vector3(point[0], point[1], 0.0))
        
        # Back face vertices (z = -thickness) - FIXED: Using Vector3 instead of lists
        for point in self.points_2d:
            vertices.append(Vector3(point[0], point[1], -self.thickness))
        
        return vertices

    # Alternative face generation for triangulated faces (if needed)
    def _generate_faces_triangulated(self):
        """
        Alternative method that generates triangulated faces.
        Useful for rendering systems that prefer triangles over quads.
        """
        faces = []
        n_points = len(self.points_2d)
        
        # Triangulate front face (fan triangulation from first vertex)
        for i in range(1, n_points - 1):
            faces.append([0, i, i + 1])
        
        # Triangulate back face (fan triangulation, reversed for correct normal)
        for i in range(1, n_points - 1):
            faces.append([n_points, n_points + i + 1, n_points + i])
        
        # Side faces as triangles
        for i in range(n_points):
            next_i = (i + 1) % n_points
            
            # First triangle of the quad
            faces.append([i, next_i, next_i + n_points])
            # Second triangle of the quad
            faces.append([i, next_i + n_points, i + n_points])
        
        return faces
    

    def draw(self):
        """Draw the polygon with OpenGL immediate mode."""
        world_verts = self.get_world_vertices()
        if not world_verts:
            print("No valid world vertices available for drawing.")
            return

        n_points = len(self.points_2d)

        if self.texture:
            # Textured rendering
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glEnable(GL_ALPHA_TEST)
            glAlphaFunc(GL_GREATER, 0.01)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glColor4f(1.0, 1.0, 1.0, 1.0)

            # Compute a single UV map for the entire polygon so the texture
            # spans the whole front/back faces instead of per-triangle.
            n_points = len(self.points_2d)
            xs = [p[0] for p in self.points_2d]
            ys = [p[1] for p in self.points_2d]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            spanx = maxx - minx if maxx - minx != 0 else 1.0
            spany = maxy - miny if maxy - miny != 0 else 1.0

            # Normalize to [0,1] across the polygon. If you prefer pixel-based
            # tiling you can factor in get_texture_size(self.texture).
            uv_map = []
            for (x, y) in self.points_2d:
                u = (x - minx) / spanx
                v = (y - miny) / spany
                uv_map.append((u, v))

            side_idx = 0
            for face_idx, face in enumerate(self.faces):
                if len(face) != 4:
                    # Front/back faces (triangles after triangulation)
                    self._draw_polygon_face_textured(world_verts, face, uv_map)
                else:
                    # Side faces (quads)
                    self._draw_quad_face_textured(world_verts, face, side_idx)
                    side_idx += 1

            glDisable(GL_BLEND)
            glDisable(GL_ALPHA_TEST)
            glDisable(GL_TEXTURE_2D)
        else:
            # Untextured rendering with flat colors
            side_idx = 0
            for face_idx, face in enumerate(self.faces):
                # Get color for this face
                color = (
                    self.face_colors[face_idx]
                    if hasattr(self, 'face_colors') and face_idx < len(self.face_colors)
                    else (1.0, 1.0, 1.0)
                )

                # Normalize color if in 0-255 range
                if any(x > 2.0 for x in color):
                    color = tuple(x / 255.0 for x in color)

                glColor3f(*color)

                if len(face) != 4:
                    # Front/back faces (triangles)
                    self._draw_polygon_face(world_verts, face)
                else:
                    # Side faces (quads)
                    self._draw_quad_face(world_verts, face)
                    side_idx += 1

    def _draw_polygon_face(self, world_verts, face):
        """Draw a polygon face (front or back) as a triangle fan."""
        glBegin(GL_TRIANGLE_FAN)
        for idx in face:
            v = world_verts[idx]
            glVertex3f(v.x, v.y, v.z)
        glEnd()

    def _draw_quad_face(self, world_verts, face):
        """Draw a quad face (side face)."""
        glBegin(GL_QUADS)
        for idx in face:
            v = world_verts[idx]
            glVertex3f(v.x, v.y, v.z)
        glEnd()

    def _draw_polygon_face_textured(self, world_verts, face, uv_map):
        """Draw a triangulated polygon face using a global uv_map.

        uv_map is a list of per-vertex (u,v) coordinates computed from the
        original 2D points. Faces reference indices into the original vertex
        list, so we look up UVs directly.
        """
        # Faces here are triangles from triangulation. Use GL_TRIANGLES to be
        # explicit about triangle lists (tri-fan is not needed).
        glBegin(GL_TRIANGLE_FAN)
        for idx in face:
            v = world_verts[idx]
            # uv_map corresponds to the front-face vertex order (0..n_points-1)
            if idx < len(uv_map):
                u, v_coord = uv_map[idx]
            else:
                # Back-face uses offset indices; map back to front vertex
                u, v_coord = uv_map[idx - len(uv_map)]

            glTexCoord2f(u, v_coord)
            glVertex3f(v.x, v.y, v.z)
        glEnd()

    def _draw_quad_face_textured(self, world_verts, face, side_idx):
        """Draw a textured quad face (side face) with edge-sampling for thin faces."""
        if len(face) != 4:
            return
            
        a, b, c, d = face
        va, vb, vc, vd = [world_verts[i] for i in face]
        
        # Compute face dimensions
        edge1 = vb - va
        edge2 = vc - vb
        span_u = edge1.length()
        span_v = edge2.length()
        
        # FIXED: Get texture size for proper tiling - removed incorrect hasattr check
        tex_size = get_texture_size(self.texture) if self.texture else None
        if tex_size:
            tex_w, tex_h = tex_size
            u_repeat = span_u / max(1e-6, float(tex_w))
            v_repeat = span_v / max(1e-6, float(tex_h))
            
            # For thin side faces, sample edge pixels to avoid stretching
            if self.thickness > 0.0:
                strip_u = max(1.0 / max(1.0, float(tex_w)) * u_repeat, 0.001 * u_repeat)
                strip_v = max(1.0 / max(1.0, float(tex_h)) * v_repeat, 0.001 * v_repeat)
                
                # Sample appropriate edge based on side
                if side_idx % 2 == 0:  # Even sides - sample thin vertical strip
                    u_min = 0.0
                    u_max = min(strip_u, u_repeat)
                else:  # Odd sides - sample from opposite edge
                    u_min = max(0.0, u_repeat - strip_u)
                    u_max = u_repeat
                    
                face_uvs = [
                    (u_min, 0.0),
                    (u_max, 0.0), 
                    (u_max, v_repeat),
                    (u_min, v_repeat)
                ]
            else:
                # Normal UV mapping for non-extruded faces
                face_uvs = [
                    (0.0, 0.0),
                    (u_repeat, 0.0),
                    (u_repeat, v_repeat),
                    (0.0, v_repeat)
                ]
        else:
            # Default UV mapping
            face_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        
        glBegin(GL_QUADS)
        for vi, idx in enumerate(face):
            uv = face_uvs[vi]
            v = world_verts[idx]
            glTexCoord2f(uv[0], uv[1])
            glVertex3f(v.x, v.y, v.z)
        glEnd()