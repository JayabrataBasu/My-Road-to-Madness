#[derive(Clone, Copy, Debug)]
pub struct FloatTween { pub start:f32, pub end:f32, pub duration:f32, pub t:f32 }

impl FloatTween {
	pub fn new(start:f32, end:f32, duration:f32) -> Self { Self { start, end, duration, t:0.0 } }
	pub fn update(&mut self, dt:f32) { self.t = (self.t + dt / self.duration).clamp(0.0, 1.0); }
	pub fn value(&self) -> f32 { self.start + (self.end - self.start) * self.ease_in_out_cubic(self.t) }
	fn ease_in_out_cubic(&self, x:f32) -> f32 { if x < 0.5 { 4.0 * x * x * x } else { 1.0 - (-2.0*x +2.0).powf(3.0)/2.0 } }
	pub fn finished(&self) -> bool { (self.t - 1.0).abs() < 1e-6 }
}
