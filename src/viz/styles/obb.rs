use aksr::Builder;

use crate::{Color, ColorSource, Palette, TextLoc, TextStyle, ThicknessDirection};

/// Drawing mode for oriented bounding boxes (OBB).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObbStyleMode {
    /// Solid line (default)
    Solid,
    /// Dashed line
    Dashed {
        /// Length of each dash segment in pixels
        length: usize,
        /// Gap between dashes in pixels
        gap: usize,
    },
    /// Corner brackets - only draw short lines at four corners
    Corners {
        /// Corner length ratio along the longer side (0.0-0.5)
        ratio_long: f32,
        /// Corner length ratio along the shorter side (0.0-0.5)
        ratio_short: f32,
    },
    /// Rounded corners
    Rounded {
        /// Corner radius as ratio of the shorter side (0.0-0.5)
        ratio: f32,
    },
}

impl Default for ObbStyleMode {
    fn default() -> Self {
        Self::Solid
    }
}

impl ObbStyleMode {
    /// Create a dashed mode with default values
    pub fn dashed() -> Self {
        Self::Dashed { length: 10, gap: 5 }
    }

    /// Create a corners mode with default ratios (20% of each side)
    pub fn corners() -> Self {
        Self::Corners {
            ratio_long: 0.2,
            ratio_short: 0.2,
        }
    }

    /// Create a rounded mode with default ratio (10% of shorter side)
    pub fn rounded() -> Self {
        Self::Rounded { ratio: 0.1 }
    }
}

/// OBB-specific style configuration.
#[derive(Debug, Clone, Builder, PartialEq)]
pub struct ObbStyle {
    visible: bool,
    text_visible: bool,
    draw_fill: bool,
    draw_outline: bool,
    fill_color: ColorSource,
    outline_color: ColorSource,
    mode: ObbStyleMode,
    thickness: usize,
    thickness_max_ratio: f32,
    thickness_direction: ThicknessDirection,
    text_style: TextStyle,
    palette: Vec<Color>,
}

impl Default for ObbStyle {
    fn default() -> Self {
        Self {
            visible: true,
            text_visible: true,
            draw_fill: false,
            draw_outline: true,
            fill_color: ColorSource::Auto,
            outline_color: ColorSource::Auto,
            mode: ObbStyleMode::rounded(),
            thickness: 3,
            thickness_max_ratio: 0.3,
            thickness_direction: ThicknessDirection::default(),
            text_style: TextStyle::default().with_loc(TextLoc::OuterTopRight),
            palette: Color::palette_base_20(),
        }
    }
}

impl Palette for ObbStyle {
    fn palette(&self) -> &[Color] {
        &self.palette
    }
}

impl ObbStyle {
    /// OBB with dashed outline
    pub fn dashed() -> Self {
        Self {
            mode: ObbStyleMode::dashed(),
            ..Default::default()
        }
    }

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
