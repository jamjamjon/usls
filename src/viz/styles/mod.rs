//! Style configurations for visualization elements.
//!
//! This module contains all style-related types organized by category:
//! - `hbb` - Horizontal bounding box styles
//! - `obb` - Oriented bounding box styles
//! - `keypoint` - Keypoint styles
//! - `polygon` - Polygon styles
//! - `mask` - Mask styles
//! - `prob` - Probability/classification styles
//! - `text` - Text rendering styles
//! - `color` - Color configuration (backward compat)

mod color;
mod hbb;
mod keypoint;
mod mask;
mod obb;
mod polygon;
mod prob;
mod text;

// Re-export all types (Style removed - use XxxStyle directly)
pub use color::*;
pub use hbb::*;
pub use keypoint::*;
pub use mask::*;
pub use obb::*;
pub use polygon::*;
pub use prob::*;
pub use text::*;

use crate::Color;

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

/// Direction for thickness expansion when drawing outlines.
///
/// Controls whether the outline grows inward, outward, or centered
/// relative to the specified coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ThicknessDirection {
    /// Expand inward (stays within the original bounds)
    Inward,
    /// Expand outward (default, current behavior)
    #[default]
    Outward,
    /// Expand in both directions (centered on the edge)
    Centered,
}
