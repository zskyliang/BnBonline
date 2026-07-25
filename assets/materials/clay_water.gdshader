shader_type spatial;
render_mode cull_back, depth_draw_opaque, diffuse_burley, specular_schlick_ggx;

uniform vec4 shallow_color : source_color = vec4(0.31, 0.67, 0.82, 1.0);
uniform vec4 deep_color : source_color = vec4(0.16, 0.45, 0.66, 1.0);
uniform sampler2D detail_normal : hint_normal, repeat_enable;


void vertex() {
	float slow_wave = sin((VERTEX.x + TIME * 0.55) * 0.72);
	float cross_wave = sin((VERTEX.z - TIME * 0.42) * 0.91);
	VERTEX.y += (slow_wave + cross_wave) * 0.025;
}


void fragment() {
	vec2 uv_a = UV * 5.0 + vec2(TIME * 0.012, TIME * 0.006);
	vec2 uv_b = UV * 7.5 + vec2(-TIME * 0.008, TIME * 0.014);
	vec3 normal_a = texture(detail_normal, uv_a).rgb;
	vec3 normal_b = texture(detail_normal, uv_b).rgb;
	float wave = clamp((normal_a.r + normal_b.g) * 0.5, 0.0, 1.0);
	ALBEDO = mix(deep_color.rgb, shallow_color.rgb, 0.42 + wave * 0.2);
	NORMAL_MAP = normalize(normal_a + normal_b);
	NORMAL_MAP_DEPTH = 0.18;
	METALLIC = 0.0;
	SPECULAR = 0.16;
	ROUGHNESS = 0.7;
}
