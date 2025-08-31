pub trait Fractal {
    /// Advance internal state (dt in seconds; may be ignored).
    fn update(&mut self, _dt: f32) {}
    /// Render a partial pass. `stride` >= 1 indicates pixel sampling interval.
    fn render_partial(&mut self, frame: &mut [u8], width: u32, height: u32, stride: u32);
    /// Called to pan view (if supported).
    fn pan(&mut self, _dx: f64, _dy: f64) {}
    /// Called to zoom (if supported).
    fn zoom(&mut self, _factor: f64) {}
    /// Name of the fractal.
    fn name(&self) -> &'static str;
    /// Short info string for overlay (iterations, zoom, etc.).
    fn info_string(&self) -> String { String::new() }
    /// Increments whenever a change invalidates previously rendered passes.
    fn detail_epoch(&self) -> u64 { 0 }
}
