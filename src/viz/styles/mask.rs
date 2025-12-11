use aksr::Builder;

use crate::{Color, ColorMap256};

/// Glow radius specification for halo effect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GlowRadius {
    /// Fixed radius in pixels
    Pixels(usize),
    /// Percentage of mask diagonal (0.0-1.0)
    Percent(f32),
}

impl Default for GlowRadius {
    fn default() -> Self {
        Self::Percent(0.05) // 5% of diagonal by default
    }
}

/// Drawing mode for mask rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaskStyleMode {
    /// Default: overlay mask color on original image
    Overlay,
    /// Halo effect: grayscale background + colored glow around mask edges
    Halo {
        /// Glow radius (pixels or percentage of mask diagonal)
        glow_radius: GlowRadius,
        /// Glow color
        glow_color: Color,
    },
}

impl Default for MaskStyleMode {
    fn default() -> Self {
        Self::Overlay
    }
}

impl MaskStyleMode {
    /// Create halo mode with purple glow using percentage-based radius (proportional)
    pub fn halo_purple() -> Self {
        Self::Halo {
            glow_radius: GlowRadius::Percent(0.05), // 5% of diagonal
            glow_color: [128, 0, 255, 180].into(),  // Purple
        }
    }

    /// Create halo mode with custom color and percentage-based radius
    pub fn halo_with(glow_percent: f32, color: Color) -> Self {
        Self::Halo {
            glow_radius: GlowRadius::Percent(glow_percent),
            glow_color: color,
        }
    }

    /// Create halo mode with fixed pixel radius
    pub fn halo_pixels(glow_pixels: usize, color: Color) -> Self {
        Self::Halo {
            glow_radius: GlowRadius::Pixels(glow_pixels),
            glow_color: color,
        }
    }
}

/// Mask-specific style configuration.
#[derive(Debug, Clone, Builder, PartialEq)]
pub struct MaskStyle {
    visible: bool,
    mode: MaskStyleMode,
    draw_polygons: bool,
    draw_polygon_largest: bool,
    draw_hbbs: bool,
    draw_obbs: bool,
    cutout: bool,
    cutout_original: bool,
    cutout_background_color: Color,
    colormap256: Option<ColorMap256>,
    palette: Vec<Color>,
}

impl Default for MaskStyle {
    fn default() -> Self {
        Self {
            visible: true,
            mode: MaskStyleMode::default(),
            draw_polygons: false,
            draw_polygon_largest: false,
            draw_hbbs: false,
            draw_obbs: false,
            cutout: true,
            cutout_original: false,
            cutout_background_color: Color::green(),
            colormap256: None,
            palette: Color::palette_base_20(),
        }
    }
}

impl crate::Palette for MaskStyle {
    fn palette(&self) -> &[Color] {
        &self.palette
    }
}

impl MaskStyle {
    pub fn halo() -> Self {
        Self {
            mode: MaskStyleMode::halo_purple(),
            ..Default::default()
        }
    }
}
