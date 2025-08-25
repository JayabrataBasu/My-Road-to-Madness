@group(0) @binding(0) var tex: texture_2d<f32>;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
	// Manual load (assumes framebuffer-sized UV mapping)
	let dims = textureDimensions(tex);
	let px = vec2<i32>(clamp(i32(uv.x * f32(dims.x)), 0, dims.x - 1), clamp(i32(uv.y * f32(dims.y)), 0, dims.y - 1));
	return textureLoad(tex, px, 0);
}
