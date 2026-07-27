shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_never;

varying vec4 streak_color;


void vertex() {
	streak_color = COLOR;
}


void fragment() {
	float horizontal = 1.0 - smoothstep(0.08, 0.95, abs(UV.x - 0.5) * 2.0);
	float tapered_tail = smoothstep(0.0, 0.2, UV.y);
	float softened_tip = 1.0 - smoothstep(0.76, 1.0, UV.y);
	float paper_breakup = 0.86 + sin(UV.y * 37.0) * 0.08;
	ALBEDO = mix(vec3(0.76, 0.91, 0.96), streak_color.rgb, 0.35);
	ALPHA = horizontal * tapered_tail * softened_tip * paper_breakup * streak_color.a;
}
