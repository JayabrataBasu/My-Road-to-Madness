use anyhow::Result;
use eframe::egui;
use log::info;

mod data;
#[cfg(feature = "gpu")]
mod gpu;
mod matrix;
mod ui;

use ui::MatrixioApp;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    info!("Starting Matrixio - High-Performance Matrix Calculator");

    // Configure the native app
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_icon(eframe::icon_data::from_png_bytes(&[]).unwrap_or_default()),
        ..Default::default()
    };

    // Run the application
    eframe::run_native(
        "Matrixio - Matrix Calculator",
        options,
        Box::new(|cc| {
            // Setup custom fonts if needed
            setup_custom_fonts(&cc.egui_ctx);

            // Configure egui style
            configure_style(&cc.egui_ctx);

            Ok(Box::new(MatrixioApp::new(cc)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run application: {}", e))?;

    Ok(())
}

fn setup_custom_fonts(ctx: &egui::Context) {
    let fonts = egui::FontDefinitions::default();

    // Add custom fonts here if needed
    // fonts.font_data.insert("custom_font".to_owned(), egui::FontData::from_static(include_bytes!("../assets/font.ttf")));

    ctx.set_fonts(fonts);
}

fn configure_style(ctx: &egui::Context) {
    let mut style = egui::Style::default();

    // Configure colors for a professional look
    style.visuals.window_fill = egui::Color32::from_rgb(40, 44, 52);
    style.visuals.panel_fill = egui::Color32::from_rgb(33, 37, 43);
    style.visuals.extreme_bg_color = egui::Color32::from_rgb(25, 28, 33);
    style.visuals.faint_bg_color = egui::Color32::from_rgb(50, 56, 66);

    // Button styling
    style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(60, 67, 78);
    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(70, 77, 88);
    style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(80, 87, 98);

    // Text colors
    style.visuals.text_color();
    style.visuals.strong_text_color();

    ctx.set_style(style);
}
