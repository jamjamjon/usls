use aksr::Builder;

use crate::Color;

/// Color configuration for different visual elements.
///
/// This struct is kept for backward compatibility. New code should use
/// `ColorSource` directly on each shape style.
///
/// # Shape colors
/// - `outline`: Shape outline/border color
/// - `fill`: Shape fill color
///
/// # Text colors
/// - `text`: Text character color
/// - `text_bg_fill`: Text background box fill color
/// - `text_bg_outline`: Text background box outline color
#[derive(Debug, Builder, Default, Clone, PartialEq, Copy)]
pub struct ColorStyle {
    /// Shape outline color
    pub outline: Option<Color>,
    /// Shape fill color
    pub fill: Option<Color>,
    /// Text character color
    pub text: Option<Color>,
    /// Text background box fill color
    pub text_bg_fill: Option<Color>,
    /// Text background box outline color
    pub text_bg_outline: Option<Color>,
}

impl ColorStyle {
    /// Backward compat alias for text_bg_fill
    pub fn text_bg(&self) -> Option<Color> {
        self.text_bg_fill
    }

    /// Backward compat builder for text_bg_fill
    pub fn with_text_bg(mut self, color: Color) -> Self {
        self.text_bg_fill = Some(color);
        self
    }
}

/// Color source for shapes and text - can inherit from shape or be custom.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ColorSource {
    /// Automatically determined (palette for shapes, black for text)
    #[default]
    Auto,
    /// Auto with custom alpha (0-255)
    AutoAlpha(u8),
    /// Inherit from shape's outline color (for text)
    InheritOutline,
    /// Inherit outline with custom alpha (0-255)
    InheritOutlineAlpha(u8),
    /// Inherit from shape's fill color (for text)
    InheritFill,
    /// Inherit fill with custom alpha (0-255)
    InheritFillAlpha(u8),
    /// Custom color
    Custom(Color),
}

impl ColorSource {
    /// Create a custom color source
    pub fn custom(color: Color) -> Self {
        Self::Custom(color)
    }

    /// Create auto with custom alpha
    pub fn auto_alpha(alpha: u8) -> Self {
        Self::AutoAlpha(alpha)
    }

    /// Create inherit outline with custom alpha
    pub fn inherit_outline_alpha(alpha: u8) -> Self {
        Self::InheritOutlineAlpha(alpha)
    }

    /// Create inherit fill with custom alpha
    pub fn inherit_fill_alpha(alpha: u8) -> Self {
        Self::InheritFillAlpha(alpha)
    }
}

/// Trait for styles that have a color palette.
pub trait Palette {
    /// Get the palette colors.
    fn palette(&self) -> &[Color];

    /// Get color from palette by index (wraps around if index > palette length).
    fn color_from_palette(&self, i: usize) -> Color {
        let palette = self.palette();
        palette[i % palette.len()]
    }
}
