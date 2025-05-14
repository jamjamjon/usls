use aksr::Builder;

use crate::{Color, ColorMap256, Skeleton};

#[derive(Debug, Clone, Builder, PartialEq)]
pub struct Style {
    visible: bool,                    // For ALL
    text_visible: bool,               // For ALL
    draw_fill: bool,                  // For ALL
    draw_outline: bool,               // For ALL
    color_fill_alpha: Option<u8>,     // Alpha for fill
    radius: usize,                    // For Keypoint
    text_x_pos: f32,                  // For Probs
    text_y_pos: f32,                  // For Probs
    thickness: usize,                 // For Hbb
    thickness_threshold: f32,         // For Hbb
    draw_mask_polygons: bool,         // For Masks
    draw_mask_polygon_largest: bool,  // For Masks
    draw_mask_hbbs: bool,             // For Masks
    draw_mask_obbs: bool,             // For Masks
    text_loc: TextLoc,                // For ALL
    color: StyleColors,               // For ALL
    palette: Vec<Color>,              // For ALL
    skeleton: Option<Skeleton>,       // For Keypoints
    colormap256: Option<ColorMap256>, // For Masks
    decimal_places: usize,
    #[args(set_pre = "show")]
    confidence: bool,
    #[args(set_pre = "show")]
    name: bool,
    #[args(set_pre = "show")]
    id: bool,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            visible: true,
            text_visible: true,
            draw_outline: true,
            draw_fill: false,
            color: Default::default(),
            color_fill_alpha: None,
            draw_mask_polygons: false,
            draw_mask_polygon_largest: false,
            draw_mask_hbbs: false,
            draw_mask_obbs: false,
            radius: 3,
            text_x_pos: 0.05,
            text_y_pos: 0.05,
            thickness: 3,
            thickness_threshold: 0.3,
            text_loc: TextLoc::OuterTopLeft,
            decimal_places: 3,
            confidence: true,
            name: true,
            id: true,
            palette: Color::palette_base_20(),
            skeleton: None,
            colormap256: None,
        }
    }
}

impl Style {
    pub fn color_from_palette(&self, i: usize) -> Color {
        self.palette[i % self.palette.len()]
    }

    pub fn draw_text(&self) -> bool {
        self.text_visible() && (self.name() || self.confidence() || self.id())
    }

    pub fn prob() -> Self {
        Self {
            text_loc: TextLoc::InnerTopLeft,
            ..Default::default()
        }
    }

    pub fn hbb() -> Style {
        Style {
            text_loc: TextLoc::OuterTopLeft,
            ..Default::default()
        }
    }

    pub fn obb() -> Self {
        Self {
            text_loc: TextLoc::OuterTopRight,
            ..Default::default()
        }
    }

    pub fn keypoint() -> Style {
        Style {
            id: true,
            name: false,
            confidence: false,
            text_visible: true,
            draw_fill: true,
            draw_outline: true,
            text_loc: TextLoc::OuterTopRight,
            radius: 5,
            ..Default::default()
        }
    }

    pub fn polygon() -> Self {
        Self {
            draw_fill: true,
            draw_outline: true,
            id: false,
            name: false,
            confidence: false,
            text_visible: false,
            text_loc: TextLoc::Center,
            ..Default::default()
        }
    }

    pub fn mask() -> Self {
        Self {
            text_visible: false,
            ..Self::polygon()
        }
    }
}

#[derive(Debug, Builder, Default, Clone, PartialEq, Copy)]
pub struct StyleColors {
    pub outline: Option<Color>,
    pub fill: Option<Color>,
    pub text: Option<Color>,
    pub text_bg: Option<Color>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
    OuterTopLeft,
    OuterTopCenter,
    OuterTopRight,
    OuterBottomLeft,
    OuterBottomCenter,
    OuterBottomRight,
    OuterCenterLeft,
    OuterCenterRight,
}
