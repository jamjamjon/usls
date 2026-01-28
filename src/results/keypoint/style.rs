use aksr::Builder;

use crate::{Color, ColorSource, Palette, Skeleton, TextLoc, TextStyle, TextStyleMode};

/// Drawing mode for keypoint rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeypointStyleMode {
    /// Circle (default)
    Circle,
    /// Star shape (e.g., 5-pointed star)
    Star {
        /// Number of points (e.g., 5 for five-pointed star)
        points: usize,
        /// Ratio of inner radius to outer radius (0.0-1.0, typically 0.4)
        inner_ratio: f32,
    },
    /// Square
    Square,
    /// Cross/Plus sign
    Cross {
        /// Thickness of cross arms in pixels
        thickness: usize,
    },
    /// Diamond shape
    Diamond,
    /// Triangle with optional rotation angle (in radians)
    /// Default points up (angle = 0), positive angle rotates clockwise
    Triangle {
        /// Rotation angle in radians (0 = pointing up)
        angle: f32,
    },
    /// X shape (diagonal cross)
    X {
        /// Thickness of X arms in pixels
        thickness: usize,
    },
    /// Rounded square
    RoundedSquare {
        /// Corner radius ratio (0.0-0.5, percentage of side length)
        corner_ratio: f32,
    },
    /// Glow effect: radial gradient emanating from the keypoint center
    /// Color fades from center outward (like a heatmap hotspot)
    Glow {
        /// Glow radius multiplier (relative to keypoint radius)
        glow_multiplier: f32,
    },
}

impl Default for KeypointStyleMode {
    fn default() -> Self {
        Self::Circle
    }
}

impl KeypointStyleMode {
    /// Create a 5-pointed star with default inner ratio
    pub fn star() -> Self {
        Self::Star {
            points: 5,
            inner_ratio: 0.5,
        }
    }

    /// Create a cross with default thickness
    pub fn cross() -> Self {
        Self::Cross { thickness: 2 }
    }

    /// Create an X shape with default thickness
    pub fn x() -> Self {
        Self::X { thickness: 2 }
    }

    /// Create a triangle pointing up (default)
    pub fn triangle() -> Self {
        Self::Triangle { angle: 0.0 }
    }

    /// Create a triangle with custom rotation angle (in radians)
    pub fn triangle_with_angle(angle: f32) -> Self {
        Self::Triangle { angle }
    }

    /// Create a rounded square with default corner ratio
    pub fn rounded_square() -> Self {
        Self::RoundedSquare { corner_ratio: 0.3 }
    }

    /// Create glow mode (2x radius)
    pub fn glow() -> Self {
        Self::Glow {
            glow_multiplier: 2.0,
        }
    }

    /// Create glow mode with custom multiplier
    pub fn glow_with(glow_multiplier: f32) -> Self {
        Self::Glow { glow_multiplier }
    }
}

/// Keypoint-specific style configuration.
#[derive(Debug, Clone, Builder, PartialEq)]
pub struct KeypointStyle {
    visible: bool,
    text_visible: bool,
    draw_fill: bool,
    draw_outline: bool,
    fill_color: ColorSource,
    outline_color: ColorSource,
    mode: KeypointStyleMode,
    radius: usize,
    /// Outline thickness (extends outward from radius boundary)
    thickness: usize,
    skeleton: Option<Skeleton>,
    skeleton_thickness: usize,
    text_style: TextStyle,
    palette: Vec<Color>,
}

impl Default for KeypointStyle {
    fn default() -> Self {
        Self {
            visible: true,
            text_visible: true,
            draw_fill: true,
            draw_outline: true,
            fill_color: ColorSource::Auto,
            outline_color: ColorSource::Auto,
            mode: KeypointStyleMode::default(),
            radius: 4,
            thickness: 2,
            skeleton: None,
            skeleton_thickness: 2,
            text_style: TextStyle::default()
                .with_mode(TextStyleMode::rounded(2.0, 3.0))
                .with_loc(TextLoc::OuterTopRight)
                .with_thickness(2)
                .with_draw_fill(true)
                .with_draw_outline(true)
                .with_bg_fill_color(ColorSource::InheritFillAlpha(220))
                .with_bg_outline_color(ColorSource::Custom(Color::black()))
                .with_id(true)
                .with_name(false)
                .with_confidence(false),
            palette: Color::palette_base_20(),
        }
    }
}

impl Palette for KeypointStyle {
    fn palette(&self) -> &[Color] {
        &self.palette
    }
}

impl KeypointStyle {
    /// Keypoint with star shape
    pub fn star() -> Self {
        Self {
            mode: KeypointStyleMode::star(),
            ..Default::default()
        }
    }

    /// Keypoint with glow effect
    pub fn glow() -> Self {
        Self {
            mode: KeypointStyleMode::glow(),
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
