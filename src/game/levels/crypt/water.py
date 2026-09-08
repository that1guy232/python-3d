"""A shallow translucent tide over the crypt's mosaic pavement."""

from OpenGL import GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader


_VERTEX = """
#version 120
varying vec3 world;
void main() {
    world = gl_Vertex.xyz;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""
_FRAGMENT = """
#version 120
varying vec3 world;
uniform float elapsed;
uniform vec3 eye;
uniform float exposure;
uniform float fogDensity;
uniform vec3 fogColor;
uniform int lightCount;
uniform vec3 lightPosition[16];
uniform vec3 lightColor[16];
void main() {
    vec2 p = world.xz;
    float a = sin(p.x * .10 + p.y * .071 + elapsed * .60);
    float b = sin(p.y * .19 - p.x * .047 - elapsed * .46);
    float crest = pow(max(0., a * b), 14.);
    float ripple = abs(sin(p.x*.16 + sin(p.y*.11+elapsed*.42))
                     + cos(p.y*.17 + sin(p.x*.09-elapsed*.35)));
    float caustic = 1.-smoothstep(.015,.10,ripple);
    caustic *= min(1.,.10/max(.10,fwidth(ripple)));
    vec3 normal = normalize(vec3(.045*a, 1., .035*b));
    vec3 view = normalize(eye-world);
    float fresnel = pow(1.-max(0.,dot(normal,view)), 3.);
    vec3 color = vec3(.055,.29,.27) + crest*vec3(.20,.31,.23);
    color += fresnel*vec3(.04,.09,.10);
    color += caustic*vec3(.18,.32,.25);
    for (int i=0; i<16; i++) {
        if (i >= lightCount) break;
        vec3 delta = lightPosition[i]-world;
        float falloff = max(0.,1.-length(delta)/220.);
        vec3 halfway = normalize(normalize(delta)+view);
        float spec = pow(max(0.,dot(normal,halfway)),96.);
        color += lightColor[i] * falloff * (spec*.85 + crest*.06);
    }
    float fog = exp(-pow(length(eye-world)*fogDensity,2.));
    gl_FragColor = vec4(mix(fogColor,color*exposure,fog), .30+fresnel*.28+caustic*.10);
}
"""


class CryptWater:
    def __init__(self, rooms):
        self.rooms = rooms
        self.program = None
        self.locations = {}

    def draw(self, scene):
        if self.program is None:
            self.program = compileProgram(compileShader(_VERTEX,gl.GL_VERTEX_SHADER),
                                          compileShader(_FRAGMENT,gl.GL_FRAGMENT_SHADER))
            self.locations = {name: gl.glGetUniformLocation(self.program,name) for name in
                ("elapsed","eye","exposure","fogDensity","fogColor","lightCount",
                 "lightPosition","lightColor")}
        previous = int(gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM))
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        try:
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_ALPHA_TEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glUseProgram(self.program)
            loc = self.locations
            gl.glUniform1f(loc['elapsed'],scene.elapsed)
            gl.glUniform3f(loc['eye'],*scene.camera.position)
            gl.glUniform1f(loc['exposure'],scene.camera.brightness_default)
            gl.glUniform1f(loc['fogDensity'],scene.fog_density if scene.fog_enabled else 0.)
            gl.glUniform3f(loc['fogColor'],*scene.renderer._sky_rgba()[:3])
            lights = scene.lighting.point_lights[:16] if scene.lighting_controller.enabled else ()
            gl.glUniform1i(loc['lightCount'],len(lights))
            if lights:
                gl.glUniform3fv(loc['lightPosition'],len(lights),[l.position for l in lights])
                gl.glUniform3fv(loc['lightColor'],len(lights),[l.color for l in lights])
            gl.glBegin(gl.GL_QUADS)
            for room in self.rooms:
                x0,x1,z0,z1 = room.bounds
                for x,z in ((x0,z0),(x0,z1),(x1,z1),(x1,z0)):
                    gl.glVertex3f(x,3.2,z)
            gl.glEnd()
        finally:
            gl.glUseProgram(previous)
            gl.glPopAttrib()

    def dispose(self):
        if self.program is not None:
            gl.glDeleteProgram(self.program)
            self.program = None
            self.locations.clear()
