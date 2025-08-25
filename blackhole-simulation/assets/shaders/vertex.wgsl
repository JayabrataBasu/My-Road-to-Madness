// Fullscreen triangle vertex shader
struct VSOut {
	@builtin(position) pos: vec4<f32>,
	@location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
	var positions = array<vec2<f32>,3>(
		vec2<f32>(-1.0, -3.0),
		vec2<f32>(-1.0,  1.0),
		vec2<f32>( 3.0,  1.0)
	);
	let p = positions[vid];
	var out: VSOut;
	out.pos = vec4<f32>(p, 0.0, 1.0);
	// Map from triangle positions to uv
	out.uv = (p * 0.5) + vec2<f32>(0.5, 0.5);
	return out;
}
