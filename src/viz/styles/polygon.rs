use aksr::Builder;

use crate::{Color, ColorSource, Palette, TextLoc, TextStyle};

/// Polygon-specific style configuration.
#[derive(Debug, Clone, Builder, PartialEq)]
pub struct PolygonStyle {
    visible: bool,
    text_visible: bool,
    draw_fill: bool,
    draw_outline: bool,
    fill_color: ColorSource,
    outline_color: ColorSource,
    thickness: usize,
    /// Maximum thickness as ratio of polygon's min(width, height)
    /// Prevents thick outlines from filling small polygons
    thickness_max_ratio: f32,

    /// Background overlay color (applied once before drawing any polygons)
    /// Useful for segmentation visualization to make polygons stand out
    background_overlay: Option<Color>,
    text_style: TextStyle,
    palette: Vec<Color>,
}

impl Default for PolygonStyle {
    fn default() -> Self {
        Self {
            visible: true,
            text_visible: false,
            draw_fill: true,
            draw_outline: true,
            fill_color: ColorSource::Custom(Color::white().with_alpha(100)),
            outline_color: ColorSource::Custom(Color::black()),
            thickness: 3,
            thickness_max_ratio: 0.15,
            background_overlay: Some(Color::white().with_alpha(120)),
            text_style: TextStyle::default().with_loc(TextLoc::Center),
            palette: Color::palette_base_20(),
        }
    }
}

impl Palette for PolygonStyle {
    fn palette(&self) -> &[Color] {
        &self.palette
    }
}

impl PolygonStyle {
    /// Set show_confidence in text_style
    pub fn show_confidence(mut self, show: bool) -> Self {
        self.text_style = self.text_style.with_confidence(show);
        self
    }

    /// Set show_id in text_style
    pub fn show_id(mut self, show: bool) -> Self {
        self.text_style = self.text_style.with_id(show);
        self
    }

    /// Set show_name in text_style
    pub fn show_name(mut self, show: bool) -> Self {
        self.text_style = self.text_style.with_name(show);
        self
    }
}
