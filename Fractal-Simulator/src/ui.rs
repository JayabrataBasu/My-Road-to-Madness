//! Simple 5x7 bitmap font text blitter for overlay.

const FONT: [[u8; 7]; 96] = include!("tiny5x7_font.incl"); // ASCII 32..127

pub fn draw_text(
    frame: &mut [u8],
    width: u32,
    height: u32,
    mut x: i32,
    y: i32,
    text: &str,
    color: [u8; 3],
) {
    let w = width as i32;
    let h = height as i32;
    for ch in text.chars() {
        if ch == '\n' {
            x = 0;
            continue;
        }
        if ch < ' ' || ch > '~' {
            x += 6;
            continue;
        }
        let glyph = FONT[(ch as u8 - 32) as usize];
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..5 {
                // width 5 bits (use lower 5 of each row byte)
                if bits & (1 << (4 - col)) != 0 {
                    let px = x + col as i32;
                    let py = y + row as i32;
                    if px >= 0 && px < w && py >= 0 && py < h {
                        let idx = ((py as u32 * width + px as u32) * 4) as usize;
                        frame[idx] = color[0];
                        frame[idx + 1] = color[1];
                        frame[idx + 2] = color[2];
                        frame[idx + 3] = 0xFF;
                    }
                }
            }
        }
        x += 6; // 5 + 1 spacing
    }
}
