"""Standalone GLSL source owned by the packet lighting backend."""

PACKET_LIGHTING_VERTEX_SOURCE = r"""#version 120

uniform vec3 u_light_dir;
uniform mat4 u_sun_shadow_matrix;
uniform int u_directional_normal_stream;
uniform float u_local_reference;
uniform int u_local_point_query_policy;
uniform int u_brightness_area_count;
uniform sampler2D u_local_light_records;
uniform float u_local_light_record_width;

varying vec4 v_color;
varying vec2 v_uv;
varying float v_emissive;
varying vec3 v_normal;
varying vec3 v_directional_normal;
varying vec3 v_eye_normal;
varying vec3 v_eye_light_dir;
varying vec3 v_eye_pos;
varying vec3 v_world_pos;
varying float v_fog_distance;
varying float v_point_query_brightness;
varying vec4 v_sun_shadow_coord;

vec4 vertex_local_light_record(int light_index, int field_index)
{
    float index = float(light_index * 3 + field_index);
    return texture2D(
        u_local_light_records,
        vec2((index + 0.5) / max(u_local_light_record_width, 1.0), 0.5)
    );
}

float point_query_brightness_at_vertex(vec3 world_pos)
{
    float brightness = 1.0;
    for (int i = 0; i < u_brightness_area_count; ++i) {
        vec4 area = vertex_local_light_record(i, 0);
        vec4 params = vertex_local_light_record(i, 1);
        vec4 bounds = vertex_local_light_record(i, 2);
        if (params.w > 0.5) {
            if (
                world_pos.x < bounds.x ||
                world_pos.x > bounds.y ||
                world_pos.z < bounds.z ||
                world_pos.z > bounds.w
            ) {
                continue;
            }
        }
        float radius = max(area.z, 0.000001);
        float dist = distance(world_pos.xz, area.xy);
        if (dist <= radius) {
            float norm_dist = clamp(dist / radius, 0.0, 1.0);
            float attenuation = pow(1.0 - norm_dist, max(params.x, 0.0));
            float target = area.w;
            float relative = u_local_reference == 0.0
                ? target
                : target / u_local_reference;
            brightness *= 1.0 + (relative - 1.0) * attenuation;
        }
    }
    return brightness;
}

void main()
{
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    v_color = gl_Color;
    v_uv = gl_MultiTexCoord0.xy;
    v_emissive = clamp(gl_MultiTexCoord0.z, 0.0, 1.0);
    v_normal = gl_Normal;
    v_directional_normal = u_directional_normal_stream != 0
        ? gl_MultiTexCoord1.xyz
        : gl_Normal;
    vec3 light_dir = u_light_dir;
    if (length(light_dir) <= 0.00001) {
        light_dir = vec3(0.0, 1.0, 0.0);
    }
    v_eye_normal = gl_NormalMatrix * gl_Normal;
    v_eye_light_dir = gl_NormalMatrix * normalize(light_dir);
    v_world_pos = gl_Vertex.xyz;
    v_sun_shadow_coord = u_sun_shadow_matrix * gl_Vertex;
    v_point_query_brightness = u_local_point_query_policy == 0
        ? 1.0
        : point_query_brightness_at_vertex(gl_Vertex.xyz);
    vec4 eye_pos = gl_ModelViewMatrix * gl_Vertex;
    v_eye_pos = eye_pos.xyz;
    v_fog_distance = length(eye_pos.xyz);
}
"""

PACKET_LIGHTING_FRAGMENT_SOURCE = r"""#version 120

uniform sampler2D u_texture;
uniform float u_exposure;
uniform float u_local_reference;
uniform int u_local_lighting_enabled;
uniform int u_local_point_query_policy;
uniform int u_directional_enabled;
uniform int u_clamp_directional_material;
uniform int u_clamp_lit_material;
uniform vec3 u_light_dir;
uniform float u_light_ambient;
uniform float u_light_diffuse;
uniform float u_light_max_factor;
uniform int u_sun_shadow_enabled;
uniform sampler2D u_sun_shadow_map;
uniform vec2 u_sun_shadow_texel_size;
uniform float u_sun_shadow_bias;
const int MAX_POINT_LIGHTS = 16;
uniform int u_point_light_count;
uniform vec4 u_point_light_position_ranges[MAX_POINT_LIGHTS];
uniform vec4 u_point_light_color_intensities[MAX_POINT_LIGHTS];
uniform int u_point_light_shadow_slots[MAX_POINT_LIGHTS];
uniform samplerCube u_point_shadow_map0;
uniform samplerCube u_point_shadow_map1;
uniform float u_point_shadow_bias0;
uniform float u_point_shadow_bias1;
uniform int u_brightness_area_count;
uniform sampler2D u_local_light_records;
uniform float u_local_light_record_width;
uniform int u_environment_enabled;
uniform int u_environment_region_count;
uniform sampler2D u_environment_region_records;
uniform float u_environment_region_record_width;
uniform int u_environment_opening_count;
uniform sampler2D u_environment_portal_records;
uniform float u_environment_portal_record_width;
uniform int u_fog_enabled;
uniform float u_fog_density;
uniform vec4 u_fog_color;
uniform float u_vibrance;
uniform int u_shine_enabled;
uniform float u_shine_strength;
uniform float u_shine_power;
uniform float u_shine_fresnel;
uniform vec3 u_shine_tint;

varying vec4 v_color;
varying vec2 v_uv;
varying float v_emissive;
varying vec3 v_normal;
varying vec3 v_directional_normal;
varying vec3 v_eye_normal;
varying vec3 v_eye_light_dir;
varying vec3 v_eye_pos;
varying vec3 v_world_pos;
varying float v_fog_distance;
varying float v_point_query_brightness;
varying vec4 v_sun_shadow_coord;

float smooth01(float value)
{
    value = clamp(value, 0.0, 1.0);
    return value * value * (3.0 - 2.0 * value);
}

float indoor_light_contribution_weight(float receiver_factor)
{
    float indoor_factor = 0.34;
    float receiver = clamp(receiver_factor, 0.0, 1.0);
    if (receiver <= indoor_factor) {
        return 1.0;
    }
    if (receiver >= 1.0) {
        return 0.0;
    }
    return 1.0 - smooth01((receiver - indoor_factor) / (1.0 - indoor_factor));
}

vec4 record_texel(sampler2D records, float width, float index)
{
    return texture2D(records, vec2((index + 0.5) / max(width, 1.0), 0.5));
}

vec4 local_light_record(int light_index, int field_index)
{
    float index = float(light_index * 3 + field_index);
    return record_texel(u_local_light_records, u_local_light_record_width, index);
}

vec4 environment_region_record(int region_index, int field_index)
{
    float index = float(region_index * 2 + field_index);
    return record_texel(
        u_environment_region_records,
        u_environment_region_record_width,
        index
    );
}

vec4 environment_portal_record(int portal_index, int field_index)
{
    float index = float(portal_index * 2 + field_index);
    return record_texel(
        u_environment_portal_records,
        u_environment_portal_record_width,
        index
    );
}

float opening_region_factor(
    vec4 opening,
    vec4 params,
    vec4 region,
    float region_factor,
    vec3 world_pos
)
{
    if (opening.x < -0.5) {
        return region_factor;
    }

    float side = opening.y;
    float center_x = opening.z;
    float center_z = opening.w;
    float width = max(1.0, params.x);
    float depth = max(1.0, params.y);
    float side_fade = max(1.0, params.z);
    float edge_factor = clamp(params.w, 0.0, 1.0);
    float inward_depth = 0.0;
    float lateral = 0.0;

    if (side < 0.5) {
        inward_depth = region.w - world_pos.z;
        lateral = world_pos.x - center_x;
    } else if (side < 1.5) {
        inward_depth = region.y - world_pos.x;
        lateral = world_pos.z - center_z;
    } else if (side < 2.5) {
        inward_depth = world_pos.z - region.z;
        lateral = world_pos.x - center_x;
    } else if (side < 3.5) {
        inward_depth = world_pos.x - region.x;
        lateral = world_pos.z - center_z;
    } else {
        return region_factor;
    }

    if (inward_depth < 0.0 || inward_depth > depth) {
        return region_factor;
    }

    float half_width = width * 0.5;
    float lateral_abs = abs(lateral);
    if (lateral_abs >= half_width + side_fade) {
        return region_factor;
    }

    float width_influence = lateral_abs <= half_width
        ? 1.0
        : 1.0 - smooth01((lateral_abs - half_width) / side_fade);
    float depth_influence = 1.0 - smooth01(inward_depth / depth);
    float influence = clamp(width_influence * depth_influence, 0.0, 1.0);
    return mix(region_factor, edge_factor, influence);
}

float environment_factor_at(vec3 world_pos)
{
    if (u_environment_enabled == 0) {
        return 1.0;
    }

    float factor = 1.0;
    for (int i = 0; i < u_environment_region_count; ++i) {
        vec4 region = environment_region_record(i, 0);
        if (
            world_pos.x >= region.x &&
            world_pos.x <= region.y &&
            world_pos.z >= region.z &&
            world_pos.z <= region.w
        ) {
            float base_region_factor = clamp(
                environment_region_record(i, 1).x,
                0.0,
                1.0
            );
            float region_opening_factor = base_region_factor;
            for (int j = 0; j < u_environment_opening_count; ++j) {
                vec4 opening = environment_portal_record(j, 0);
                if (abs(opening.x - float(i)) > 0.5) {
                    continue;
                }
                region_opening_factor = max(
                    region_opening_factor,
                    opening_region_factor(
                        opening,
                        environment_portal_record(j, 1),
                        region,
                        base_region_factor,
                        world_pos
                    )
                );
            }
            factor = min(factor, region_opening_factor);
        }
    }
    return factor;
}

float brightness_at(vec3 world_pos, float receiver_factor, vec3 surface_normal)
{
    float brightness = 1.0;
    for (int i = 0; i < u_brightness_area_count; ++i) {
        vec4 area = local_light_record(i, 0);
        vec4 params = local_light_record(i, 1);
        vec4 bounds = local_light_record(i, 2);
        float indoor_only = params.y;
        if (params.w > 0.5) {
            if (
                world_pos.x < bounds.x ||
                world_pos.x > bounds.y ||
                world_pos.z < bounds.z ||
                world_pos.z > bounds.w
            ) {
                continue;
            }
        }
        float radius = max(area.z, 0.000001);
        float dist = distance(world_pos.xz, area.xy);
        if (dist <= radius) {
            float norm_dist = clamp(dist / radius, 0.0, 1.0);
            float attenuation = pow(1.0 - norm_dist, max(params.x, 0.0));
            if (indoor_only > 0.5 && u_local_point_query_policy == 0) {
                attenuation *= indoor_light_contribution_weight(receiver_factor);
                if (attenuation <= 0.000001) {
                    continue;
                }
            }
            float target = area.w;
            if (surface_normal.y > 0.55 && u_local_point_query_policy == 0) {
                target = mix(
                    u_local_reference,
                    target,
                    clamp(params.z, 0.0, 1.0)
                );
            }
            float relative = u_local_reference == 0.0 ? target : target / u_local_reference;
            brightness *= 1.0 + (relative - 1.0) * attenuation;
        }
    }
    return brightness;
}

float sun_visibility(vec3 surface_normal)
{
    if (u_sun_shadow_enabled == 0) {
        return 1.0;
    }
    vec3 shadow_coord = v_sun_shadow_coord.xyz / v_sun_shadow_coord.w;
    shadow_coord = shadow_coord * 0.5 + 0.5;
    if (
        shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
        shadow_coord.y < 0.0 || shadow_coord.y > 1.0 ||
        shadow_coord.z < 0.0 || shadow_coord.z > 1.0
    ) {
        return 1.0;
    }

    float slope = 1.0 - max(0.0, dot(surface_normal, normalize(u_light_dir)));
    float bias = u_sun_shadow_bias * (1.0 + slope * 2.0);
    float visibility = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(float(x), float(y)) * u_sun_shadow_texel_size;
            float closest_depth = texture2D(
                u_sun_shadow_map,
                shadow_coord.xy + offset
            ).r;
            visibility += shadow_coord.z - bias <= closest_depth ? 1.0 : 0.0;
        }
    }
    return visibility / 9.0;
}

float sunlight_factor(vec3 surface_normal, float visibility)
{
    if (u_directional_enabled == 0) {
        return 1.0;
    }
    vec3 normal = normalize(v_directional_normal);
    float dot_light = max(0.0, dot(normal, normalize(u_light_dir)));
    return clamp(
        u_light_ambient + u_light_diffuse * dot_light * visibility,
        0.0,
        u_light_max_factor
    );
}

float point_shadow_visibility(
    int shadow_slot,
    vec3 light_to_fragment,
    float distance_to_light,
    float light_range
)
{
    if (shadow_slot == -1) {
        return 1.0;
    }
    if (shadow_slot < -1) {
        return 0.0;
    }
    float normalized_depth = shadow_slot == 0
        ? textureCube(u_point_shadow_map0, light_to_fragment).r
        : textureCube(u_point_shadow_map1, light_to_fragment).r;
    float closest_distance = normalized_depth * light_range;
    float bias = shadow_slot == 0 ? u_point_shadow_bias0 : u_point_shadow_bias1;
    return distance_to_light - bias <= closest_distance ? 1.0 : 0.0;
}

vec3 point_light_contribution(vec3 surface_normal)
{
    vec3 result = vec3(0.0);
    vec3 normal = normalize(surface_normal);
    for (int i = 0; i < u_point_light_count; ++i) {
        vec4 position_range = u_point_light_position_ranges[i];
        vec4 color_intensity = u_point_light_color_intensities[i];
        vec3 to_light = position_range.xyz - v_world_pos;
        float distance_to_light = length(to_light);
        float range = max(position_range.w, 0.0001);
        if (distance_to_light >= range || distance_to_light <= 0.0001) {
            continue;
        }
        vec3 light_dir = to_light / distance_to_light;
        float diffuse = max(0.0, dot(normal, light_dir));
        float normalized_distance = clamp(distance_to_light / range, 0.0, 1.0);
        float attenuation = 1.0 - normalized_distance;
        attenuation *= attenuation;
        float visibility = point_shadow_visibility(
            u_point_light_shadow_slots[i],
            v_world_pos - position_range.xyz,
            distance_to_light,
            range
        );
        result += color_intensity.rgb
            * max(color_intensity.a, 0.0)
            * diffuse
            * attenuation
            * visibility;
    }
    return result;
}

vec3 safe_normalize(vec3 value, vec3 fallback)
{
    float len = length(value);
    if (len <= 0.00001) {
        return fallback;
    }
    return value / len;
}

float shine_factor(
    vec3 tex_rgb,
    float receiver_factor,
    float environment_factor,
    vec3 surface_normal,
    float direct_sun_visibility
)
{
    if (u_shine_enabled == 0 || u_shine_strength <= 0.0) {
        return 0.0;
    }

    vec3 normal = safe_normalize(v_eye_normal, vec3(0.0, 0.0, 1.0));
    vec3 light_dir = safe_normalize(v_eye_light_dir, vec3(0.0, 0.0, 1.0));
    vec3 view_dir = safe_normalize(-v_eye_pos, vec3(0.0, 0.0, 1.0));
    vec3 half_dir = safe_normalize(light_dir + view_dir, view_dir);
    float direct = max(0.0, dot(normal, light_dir));
    float facing = max(0.0, dot(normal, view_dir));
    float specular = pow(
        max(0.0, dot(normal, half_dir)),
        max(u_shine_power, 1.0)
    );
    float fresnel = pow(1.0 - facing, 3.0) * clamp(u_shine_fresnel, 0.0, 1.0);
    float luma = dot(tex_rgb, vec3(0.299, 0.587, 0.114));
    float material = mix(0.55, 1.20, clamp(luma, 0.0, 1.0));
    float covered_visibility = 1.0;
    if (u_environment_enabled != 0) {
        covered_visibility = smooth01((environment_factor - 0.62) / 0.38);
    }
    float receiver_visibility = smooth01((receiver_factor - 0.24) / 0.76);
    return clamp(
        (specular + fresnel * 0.55)
            * u_shine_strength
            * material
            * (0.35 + 0.65 * direct)
            * direct_sun_visibility
            * covered_visibility
            * receiver_visibility,
        0.0,
        0.75
    );
}

vec3 apply_fog(vec3 rgb)
{
    if (u_fog_enabled == 0) {
        return rgb;
    }
    float density = max(u_fog_density, 0.0);
    float fog_factor = exp(-pow(density * v_fog_distance, 2.0));
    fog_factor = clamp(fog_factor, 0.0, 1.0);
    return mix(u_fog_color.rgb, rgb, fog_factor);
}

vec3 apply_vibrance(vec3 rgb)
{
    float vibrance = clamp(u_vibrance, 0.0, 2.0);
    if (abs(vibrance - 1.0) <= 0.0001) {
        return rgb;
    }

    float luma = dot(rgb, vec3(0.299, 0.587, 0.114));
    vec3 gray = vec3(luma);
    float max_channel = max(max(rgb.r, rgb.g), rgb.b);
    float min_channel = min(min(rgb.r, rgb.g), rgb.b);
    float saturation = clamp(max_channel - min_channel, 0.0, 1.0);
    float mix_factor = vibrance < 1.0
        ? vibrance
        : 1.0 + (vibrance - 1.0) * (1.0 - saturation * 0.75);
    return clamp(mix(gray, rgb, mix_factor), 0.0, 1.0);
}

void main()
{
    vec4 texel = texture2D(u_texture, v_uv);
    vec3 surface_normal = normalize(v_normal);
    float vertex_factor = max(max(v_color.r, v_color.g), v_color.b);
    // A completed geometry shadow map is the normal raster-lighting path.
    // Scalar X/Z lights and covered-region visibility are retained only for
    // explicit no-shadow compatibility draws during rollback.
    bool raster_visibility = u_sun_shadow_enabled != 0;
    float environment_factor = raster_visibility
        ? 1.0
        : environment_factor_at(v_world_pos);
    float receiver_factor = min(vertex_factor, environment_factor);
    float environment_scale = vertex_factor <= 0.0001
        ? environment_factor
        : receiver_factor / vertex_factor;
    vec3 environment_receiver_rgb = v_color.rgb * environment_scale;
    vec3 receiver_rgb = v_color.rgb;
    float local_brightness = raster_visibility
        ? 1.0
        : (
            u_local_point_query_policy == 0
                ? brightness_at(v_world_pos, receiver_factor, surface_normal)
                : v_point_query_brightness
        );
    float brightness = (
        u_local_lighting_enabled == 0 || raster_visibility
    ) ? u_exposure : local_brightness * u_exposure;
    float direct_sun_visibility = sun_visibility(surface_normal);
    vec3 directional_receiver = environment_receiver_rgb
        * sunlight_factor(surface_normal, direct_sun_visibility);
    if (u_clamp_directional_material != 0) {
        directional_receiver = clamp(directional_receiver, 0.0, 1.0);
    }
    vec3 lit_receiver = directional_receiver * brightness;
    lit_receiver += point_light_contribution(surface_normal)
        * receiver_rgb
        * u_exposure;
    lit_receiver = mix(
        lit_receiver,
        receiver_rgb * u_exposure,
        v_emissive
    );
    if (u_clamp_lit_material != 0) {
        lit_receiver = clamp(lit_receiver, 0.0, 1.0);
    }
    vec3 rgb = texel.rgb * lit_receiver;
    float shine = shine_factor(
        texel.rgb,
        receiver_factor,
        environment_factor,
        surface_normal,
        direct_sun_visibility
    ) * texel.a;
    rgb = min(rgb + u_shine_tint * shine, vec3(1.0));
    rgb = apply_vibrance(rgb);
    rgb = apply_fog(rgb);
    gl_FragColor = vec4(rgb, texel.a * v_color.a);
}
"""
