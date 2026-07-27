shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_never;

varying vec4 ripple_color;


void vertex() {
	ripple_color = COLOR;
}


void fragment() {
	float radius = length(UV - vec2(0.5)) * 2.0;
	float outer_ring = 1.0 - smoothstep(0.035, 0.105, abs(radius - 0.76));
	float inner_ring = 1.0 - smoothstep(0.03, 0.09, abs(radius - 0.48));
	float watercolor_edge = 0.9 + sin(atan(UV.y - 0.5, UV.x - 0.5) * 13.0) * 0.1;
	ALBEDO = ripple_color.rgb;
	ALPHA = max(outer_ring, inner_ring * 0.38) * watercolor_edge * ripple_color.a;
}
