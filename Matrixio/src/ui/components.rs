// Common UI components and utilities
use eframe::egui;

/// Custom button styles and components
pub struct CustomButton;

impl CustomButton {
    pub fn primary(text: &str) -> egui::Button {
        egui::Button::new(text)
            .fill(egui::Color32::from_rgb(0, 122, 255))
    }

    pub fn secondary(text: &str) -> egui::Button {
        egui::Button::new(text)
            .fill(egui::Color32::from_rgb(108, 117, 125))
    }

    pub fn danger(text: &str) -> egui::Button {
        egui::Button::new(text)
            .fill(egui::Color32::from_rgb(220, 53, 69))
    }

    pub fn success(text: &str) -> egui::Button {
        egui::Button::new(text)
            .fill(egui::Color32::from_rgb(40, 167, 69))
    }
}

/// Progress bar component
pub struct ProgressBar {
    progress: f32,
    label: String,
}

impl ProgressBar {
    pub fn new(progress: f32, label: String) -> Self {
        Self { progress, label }
    }

    pub fn render(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(&self.label);
            ui.add(egui::ProgressBar::new(self.progress).show_percentage());
        });
    }
}

/// Matrix size indicator
pub fn render_matrix_size_indicator(ui: &mut egui::Ui, rows: usize, cols: usize) {
    let color = if rows * cols > 10000 {
        egui::Color32::RED
    } else if rows * cols > 1000 {
        egui::Color32::YELLOW
    } else {
        egui::Color32::GREEN
    };

    ui.colored_label(color, format!("{}×{}", rows, cols));
}

/// Memory usage indicator
pub fn render_memory_indicator(ui: &mut egui::Ui, bytes: usize) {
    let (value, unit, color) = if bytes < 1024 {
        (bytes as f64, "B", egui::Color32::GREEN)
    } else if bytes < 1024 * 1024 {
        (bytes as f64 / 1024.0, "KB", egui::Color32::GREEN)
    } else if bytes < 1024 * 1024 * 1024 {
        (bytes as f64 / (1024.0 * 1024.0), "MB", egui::Color32::YELLOW)
    } else {
        (bytes as f64 / (1024.0 * 1024.0 * 1024.0), "GB", egui::Color32::RED)
    };

    ui.colored_label(color, format!("{:.1} {}", value, unit));
}