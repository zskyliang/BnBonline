shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_opaque, blend_mix;

uniform sampler2D base_texture : source_color, filter_linear_mipmap_anisotropic;
uniform sampler2D tint_mask : filter_linear_mipmap;
uniform vec4 team_color : source_color = vec4(0.31, 0.68, 0.78, 1.0);
uniform float team_strength : hint_range(0.0, 1.0) = 0.92;
uniform float alpha_threshold : hint_range(0.0, 1.0) = 0.055;


void fragment() {
	vec4 base = texture(base_texture, UV);
	float mask_value = texture(tint_mask, UV).r * team_strength;
	float paper_value = dot(base.rgb, vec3(0.2126, 0.7152, 0.0722));
	vec3 watercolor_team = team_color.rgb * mix(0.76, 1.14, paper_value);
	ALBEDO = mix(base.rgb, watercolor_team, mask_value);
	ALPHA = base.a;
	ALPHA_SCISSOR_THRESHOLD = alpha_threshold;
}
