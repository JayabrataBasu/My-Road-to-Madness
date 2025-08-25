use std::sync::Arc;
use glam::Vec3;
use crate::physics::black_hole::BlackHole;

pub struct BlackHoleObject {
	pub bh: Arc<BlackHole>,
	pub position: Vec3,
}

impl BlackHoleObject {
	pub fn new(bh: Arc<BlackHole>, position: Vec3) -> Self { Self { bh, position } }
}

pub enum SimObject {
	BlackHole(BlackHoleObject),
	// Future: AccretionDisk, StarField, ParticleEmitter
}

impl SimObject {
	pub fn position(&self) -> Vec3 {
		match self { SimObject::BlackHole(o) => o.position }
	}
}
