shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_opaque;

uniform sampler2D tile_atlas : source_color, filter_linear_mipmap_anisotropic;
varying vec4 tile_tint;
varying vec2 tile_metadata;


void vertex() {
	tile_tint = COLOR;
	tile_metadata = INSTANCE_CUSTOM.rg;
}


void fragment() {
	float variant = floor(clamp(tile_metadata.r, 0.0, 0.999) * 6.0);
	vec2 atlas_cell = vec2(mod(variant, 3.0), floor(variant / 3.0));
	vec2 atlas_uv = (UV + atlas_cell) / vec2(3.0, 2.0);
	vec4 base = texture(tile_atlas, atlas_uv);
	float paper_value = dot(base.rgb, vec3(0.2126, 0.7152, 0.0722));
	vec3 painted = tile_tint.rgb * mix(0.72, 1.12, paper_value);
	ALBEDO = mix(base.rgb, painted, tile_metadata.g * 0.78);
	ALPHA = base.a;
	ALPHA_SCISSOR_THRESHOLD = 0.035;
}
