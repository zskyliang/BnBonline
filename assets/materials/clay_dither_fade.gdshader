shader_type spatial;
render_mode cull_back, depth_draw_opaque, diffuse_burley, specular_schlick_ggx;

uniform vec4 albedo_color : source_color = vec4(0.85, 0.4, 0.3, 1.0);
uniform sampler2D normal_texture : hint_normal;
uniform float normal_strength : hint_range(0.0, 1.0) = 0.22;
uniform float visible_ratio : hint_range(0.0, 1.0) = 1.0;
uniform float roughness_value : hint_range(0.0, 1.0) = 0.92;


float bayer_4x4(vec2 pixel_position) {
	ivec2 pixel = ivec2(mod(floor(pixel_position), 4.0));
	int index = pixel.x + pixel.y * 4;
	float pattern[16] = float[](
		0.0, 8.0, 2.0, 10.0,
		12.0, 4.0, 14.0, 6.0,
		3.0, 11.0, 1.0, 9.0,
		15.0, 7.0, 13.0, 5.0
	);
	return (pattern[index] + 0.5) / 16.0;
}


void fragment() {
	if (visible_ratio < bayer_4x4(FRAGCOORD.xy)) {
		discard;
	}
	ALBEDO = albedo_color.rgb;
	METALLIC = 0.0;
	SPECULAR = 0.12;
	ROUGHNESS = roughness_value;
	NORMAL_MAP = texture(normal_texture, UV * 7.5).rgb;
	NORMAL_MAP_DEPTH = normal_strength;
}
