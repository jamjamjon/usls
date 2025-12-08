use aksr::Builder;

use crate::{Color, ColorSource, Palette, TextLoc, TextStyle, TextStyleMode};

/// Prob-specific style configuration.
#[derive(Debug, Clone, Builder, PartialEq)]
pub struct ProbStyle {
    visible: bool,
    text_x_pos: f32,
    text_y_pos: f32,
    text_style: TextStyle,
    palette: Vec<Color>,
}

impl Default for ProbStyle {
    fn default() -> Self {
        Self {
            visible: true,
            text_x_pos: 0.05,
            text_y_pos: 0.05,
            text_style: TextStyle::default()
                .with_loc(TextLoc::InnerTopLeft)
                .with_mode(TextStyleMode::rect(5.0))
                .with_thickness(2)
                .with_draw_fill(true)
                .with_draw_outline(true)
                .with_bg_fill_color(ColorSource::AutoAlpha(180))
                .with_bg_outline_color(ColorSource::Custom(Color::black())),
            palette: Color::palette_base_20(),
        }
    }
}

impl Palette for ProbStyle {
    fn palette(&self) -> &[Color] {
        &self.palette
    }
}
