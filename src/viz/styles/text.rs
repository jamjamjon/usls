use aksr::Builder;

use crate::ColorSource;

/// Shape mode for text background box.
///
/// This enum defines the shape of the text background box.
/// Use `draw_fill` and `draw_outline` in TextStyle to control rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TextStyleMode {
    /// Rectangle background
    Rect {
        /// Padding around text
        padding: f32,
    },
    /// Rounded rectangle background
    Rounded {
        /// Padding around text
        padding: f32,
        /// Corner radius
        radius: f32,
    },
}

impl Default for TextStyleMode {
    fn default() -> Self {
        Self::Rect { padding: 4.0 }
    }
}

impl TextStyleMode {
    /// Create a rectangle background with padding
    pub fn rect(padding: f32) -> Self {
        Self::Rect { padding }
    }

    /// Create a rounded rectangle background
    pub fn rounded(padding: f32, radius: f32) -> Self {
        Self::Rounded { padding, radius }
    }

    /// Get padding value
    pub fn padding(&self) -> f32 {
        match self {
            Self::Rect { padding } | Self::Rounded { padding, .. } => *padding,
        }
    }

    /// Get corner radius (0.0 for non-rounded modes)
    pub fn corner_radius(&self) -> f32 {
        match self {
            Self::Rounded { radius, .. } => *radius,
            Self::Rect { .. } => 0.0,
        }
    }

    /// Check if this is a rounded mode
    pub fn is_rounded(&self) -> bool {
        matches!(self, Self::Rounded { .. })
    }
}

/// Text-specific style configuration.
///
/// Text rendering has three visual elements:
/// - **Text characters**: colored by `color`
/// - **Background box fill**: colored by `bg_fill_color`
/// - **Background box outline**: colored by `bg_outline_color`
#[derive(Debug, Clone, Builder, PartialEq)]
pub struct TextStyle {
    /// Whether to show text
    visible: bool,
    /// Text position relative to shape
    loc: TextLoc,
    /// Background box shape mode (Rect or Rounded)
    mode: TextStyleMode,

    /// Font size in pixels (None = use global default)
    font_size: Option<f32>,

    /// Whether to draw background fill
    draw_fill: bool,
    /// Whether to draw background outline
    draw_outline: bool,
    /// Background box outline thickness
    thickness: usize,

    /// Text character color
    color: ColorSource,
    /// Background box fill color
    bg_fill_color: ColorSource,
    /// Background box outline color
    bg_outline_color: ColorSource,

    /// Show confidence value
    #[args(setter_prefix = "show")]
    confidence: bool,
    /// Show name/label
    #[args(setter_prefix = "show")]
    name: bool,
    /// Show ID
    #[args(setter_prefix = "show")]
    id: bool,
    /// Decimal places for confidence
    decimal_places: usize,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            visible: true,
            loc: TextLoc::OuterTopLeft,
            mode: TextStyleMode::Rect { padding: 4.0 },
            font_size: None, // Use global default from TextRenderer
            draw_fill: true,
            draw_outline: false,
            thickness: 0,
            color: ColorSource::Auto,
            bg_fill_color: ColorSource::Auto,
            bg_outline_color: ColorSource::Auto,
            confidence: true,
            name: true,
            id: true,
            decimal_places: 3,
        }
    }
}

impl TextStyle {
    /// Check if any text content should be shown
    pub fn should_draw(&self) -> bool {
        self.visible && (self.name || self.confidence || self.id)
    }

    /// Alias for bg_fill_color (backward compat)
    pub fn bg_color(&self) -> ColorSource {
        self.bg_fill_color
    }

    /// Alias for with_bg_fill_color (backward compat)
    pub fn with_bg_color(self, color: ColorSource) -> Self {
        self.with_bg_fill_color(color)
    }
}

/// Text positioning options relative to visual elements.
///
/// Text anchor point convention:
/// - The returned (x, y) represents the **bottom-left** corner of the text box
/// - The text renderer will draw text with its bottom edge at y
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TextLoc {
    InnerTopLeft,
    InnerTopCenter,
    InnerTopRight,
    InnerBottomLeft,
    InnerBottomCenter,
    InnerBottomRight,
    InnerCenterLeft,
    InnerCenterRight,
    Center,
    #[default]
    OuterTopLeft,
    OuterTopCenter,
    OuterTopRight,
    OuterBottomLeft,
    OuterBottomCenter,
    OuterBottomRight,
    OuterCenterLeft,
    OuterCenterRight,
}

impl TextLoc {
    /// Calculate the text anchor position (bottom-left of text box).
    ///
    /// # Arguments
    /// * `bbox` - Bounding box as (xmin, ymin, xmax, ymax)
    /// * `text_size` - Text dimensions as (width, height)
    /// * `canvas_size` - Canvas dimensions as (width, height)
    /// * `padding` - Optional padding from edges (default 2.0)
    /// * `offset` - Optional vertical offset (e.g., for thickness adjustment)
    ///
    /// # Returns
    /// (x, y) anchor point where text bottom-left should be positioned
    pub fn compute_anchor(
        &self,
        bbox: (f32, f32, f32, f32),
        text_size: (u32, u32),
        canvas_size: (u32, u32),
        padding: Option<f32>,
        offset: Option<f32>,
    ) -> (f32, f32) {
        let (xmin, ymin, xmax, ymax) = bbox;
        let (text_w, text_h) = (text_size.0 as f32, text_size.1 as f32);
        let (canvas_w, canvas_h) = (canvas_size.0 as f32, canvas_size.1 as f32);
        let pad = padding.unwrap_or(2.0);
        let off = offset.unwrap_or(0.0);
        let cx = (xmin + xmax) / 2.0;
        let cy = (ymin + ymax) / 2.0;

        // Compute raw position based on alignment
        let (mut x, mut y) = match self {
            // Inner positions (text inside the bounding box)
            TextLoc::InnerTopLeft => (xmin + pad, ymin + text_h + pad),
            TextLoc::InnerTopCenter => (cx - text_w / 2.0, ymin + text_h + pad),
            TextLoc::InnerTopRight => (xmax - text_w - pad, ymin + text_h + pad),
            TextLoc::InnerBottomLeft => (xmin + pad, ymax - pad),
            TextLoc::InnerBottomCenter => (cx - text_w / 2.0, ymax - pad),
            TextLoc::InnerBottomRight => (xmax - text_w - pad, ymax - pad),
            TextLoc::InnerCenterLeft => (xmin + pad, cy + text_h / 2.0),
            TextLoc::InnerCenterRight => (xmax - text_w - pad, cy + text_h / 2.0),
            TextLoc::Center => (cx - text_w / 2.0, cy + text_h / 2.0),
            // Outer positions (text outside the bounding box)
            // Apply offset for OuterTop positions to avoid overlapping with shape edges
            TextLoc::OuterTopLeft => (xmin, ymin - off),
            TextLoc::OuterTopCenter => (cx - text_w / 2.0, ymin - off),
            TextLoc::OuterTopRight => (xmax - text_w, ymin - off),
            TextLoc::OuterBottomLeft => (xmin, ymax + text_h + pad + off),
            TextLoc::OuterBottomCenter => (cx - text_w / 2.0, ymax + text_h + pad + off),
            TextLoc::OuterBottomRight => (xmax - text_w, ymax + text_h + pad + off),
            TextLoc::OuterCenterLeft => (xmin - text_w - pad, cy + text_h / 2.0),
            TextLoc::OuterCenterRight => (xmax + pad, cy + text_h / 2.0),
        };

        // Clamp to canvas boundaries
        // Ensure text doesn't go off the left edge
        if x < 0.0 {
            x = 0.0;
        }
        // Ensure text doesn't go off the right edge
        if x + text_w > canvas_w {
            x = (canvas_w - text_w).max(0.0);
        }
        // Ensure text doesn't go off the top edge (y is bottom of text, so y - text_h >= 0)
        if y - text_h < 0.0 {
            y = text_h;
        }
        // Ensure text doesn't go off the bottom edge
        if y > canvas_h {
            y = canvas_h;
        }

        (x, y)
    }
}
