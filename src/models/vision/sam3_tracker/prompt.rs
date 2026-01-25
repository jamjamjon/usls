use crate::{
    Color, ColorSource, Hbb, HbbStyle, HbbStyleMode, Keypoint, KeypointStyle, KeypointStyleMode,
    TextStyle, TextStyleMode,
};

/// SAM3 Prompt - supports text, boxes, and points
///
/// # CLI Format (FromStr parsing rules)
/// ```text
/// # Text only
/// -p "cat"
///
/// # Text + box (xywh)
/// -p "shoe;pos:480,290,110,360"
///
/// # Geometry only - auto "visual" text
/// -p "pos:480,290,110,360;neg:370,280,115,375"
///
/// # Point (2 coords) - auto-detect
/// -p "pos:500,375"
///
/// # Box (4 coords) - auto-detect
/// -p "pos:480,290,110,360"
///
/// # Multiple geometry
/// -p "pos:500,375;pos:1125,625;neg:300,400"
/// ```
///
/// ## Parsing Rules:
/// 1. First part without `pos:`/`neg:` prefix → text prompt
/// 2. Parts with `pos:`/`neg:` prefix → geometry (box or point)
/// 3. 2 coords → point, 4 coords → box (xywh)
/// 4. If only geometry (no text), "visual" is auto-set
#[derive(Debug, Clone, Default)]
pub struct Sam3Prompt {
    /// Text prompt (use "visual" for geometry-only mode)
    pub text: String,
    pub is_auto_text: bool,
    /// Bounding boxes (xywh format, with name "positive"/"negative")
    pub boxes: Vec<Hbb>,
    /// Points (with name "positive"/"negative")
    pub points: Vec<Keypoint>,
}

impl Sam3Prompt {
    /// Default class name for geometry-only prompts
    pub const VISUAL: &'static str = "visual";
    /// Name for positive geometry
    pub const POSITIVE: &'static str = "positive";
    /// Name for negative geometry
    pub const NEGATIVE: &'static str = "negative";

    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            is_auto_text: false,
            boxes: Vec::new(),
            points: Vec::new(),
        }
    }

    /// Create a geometry-only prompt (text defaults to "visual")
    pub fn visual() -> Self {
        Self {
            text: Self::VISUAL.to_string(),
            is_auto_text: true,
            boxes: Vec::new(),
            points: Vec::new(),
        }
    }

    pub fn with_text(mut self, text: &str) -> Self {
        self.text = text.to_string();
        self.is_auto_text = false;
        self
    }

    // ==================== Box Methods (xywh format) ====================

    /// Add a bounding box (xywh format) with style (green=positive, red=negative)
    pub fn with_box(mut self, x: f32, y: f32, w: f32, h: f32, positive: bool) -> Self {
        let (name, color) = if positive {
            (Self::POSITIVE, Color::cyan())
        } else {
            (Self::NEGATIVE, Color::red())
        };
        self.boxes.push(
            Hbb::from_xywh(x, y, w, h)
                .with_name(name)
                .with_confidence(1.0) // Required for drawing
                .with_style(
                    HbbStyle::default()
                        .with_mode(HbbStyleMode::dashed())
                        .with_thickness(6)
                        .with_draw_fill(true)
                        .with_draw_outline(true)
                        .with_outline_color(ColorSource::Custom(color))
                        .with_text_visible(true)
                        .with_text_style(
                            TextStyle::default()
                                .with_mode(TextStyleMode::rect(5.))
                                .with_draw_fill(true)
                                .with_bg_fill_color(ColorSource::Custom(color)),
                        )
                        .show_id(false)
                        .show_confidence(false),
                ),
        );
        self
    }

    /// Add a positive box (xywh)
    pub fn with_positive_box(self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.with_box(x, y, w, h, true)
    }

    /// Add a negative box (xywh)
    pub fn with_negative_box(self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.with_box(x, y, w, h, false)
    }

    // ==================== Point Methods ====================

    /// Add a point with style (green=positive, red=negative)
    pub fn with_point(mut self, x: f32, y: f32, positive: bool) -> Self {
        let (name, color) = if positive {
            (Self::POSITIVE, Color::green())
        } else {
            (Self::NEGATIVE, Color::red())
        };
        self.points.push(
            Keypoint::default()
                .with_xy(x, y)
                .with_name(name)
                .with_confidence(1.0) // Required for drawing (confidence > 0)
                .with_style(
                    KeypointStyle::default()
                        .with_mode(KeypointStyleMode::star())
                        .with_radius(15)
                        .with_draw_fill(true)
                        .with_draw_outline(true)
                        .with_fill_color(ColorSource::Custom(color))
                        .with_text_visible(false),
                ),
        );
        self
    }

    /// Add a positive point (foreground)
    pub fn with_positive_point(self, x: f32, y: f32) -> Self {
        self.with_point(x, y, true)
    }

    /// Add a negative point (background)
    pub fn with_negative_point(self, x: f32, y: f32) -> Self {
        self.with_point(x, y, false)
    }

    // ==================== Query Methods ====================

    /// Check if prompt has any boxes
    pub fn has_boxes(&self) -> bool {
        !self.boxes.is_empty()
    }

    /// Check if prompt has any points
    pub fn has_points(&self) -> bool {
        !self.points.is_empty()
    }

    /// Check if prompt has at least one positive box
    pub fn has_positive_box(&self) -> bool {
        self.boxes.iter().any(|b| b.name() == Some(Self::POSITIVE))
    }

    /// Check if this is a geometry-only prompt (text is "visual")
    pub fn is_visual(&self) -> bool {
        self.text == Self::VISUAL
    }

    /// Check if geometry should be used (for sam3-image):
    /// - "visual" text: requires at least one positive box
    /// - Other text: any boxes are valid
    pub fn should_use_geometry(&self) -> bool {
        if self.is_visual() {
            self.has_positive_box()
        } else {
            self.has_boxes()
        }
    }

    /// Get box labels (1=positive, 0=negative)
    pub fn box_labels(&self) -> Vec<i64> {
        self.boxes
            .iter()
            .map(|b| {
                if b.name() == Some(Self::POSITIVE) {
                    1
                } else {
                    0
                }
            })
            .collect()
    }

    /// Get point labels (1=positive, 0=negative)
    pub fn point_labels(&self) -> Vec<i64> {
        self.points
            .iter()
            .map(|p| {
                if p.name() == Some(Self::POSITIVE) {
                    1
                } else {
                    0
                }
            })
            .collect()
    }

    // ==================== Conversion Methods ====================

    /// Normalize boxes for SAM3-Image v2 (scale to model input space, then normalize to cxcywh)
    /// This is needed when model input resolution differs from original image resolution
    pub fn normalized_boxes_scaled(
        &self,
        scale_x: f32,
        scale_y: f32,
        model_width: f32,
        model_height: f32,
    ) -> Vec<[f32; 4]> {
        self.boxes
            .iter()
            .map(|hbb| {
                let (x, y, w, h) = hbb.xywh();
                let cx = ((x + w / 2.0) * scale_x) / model_width;
                let cy = ((y + h / 2.0) * scale_y) / model_height;
                let nw = (w * scale_x) / model_width;
                let nh = (h * scale_y) / model_height;
                [cx, cy, nw, nh]
            })
            .collect()
    }

    /// Scale points by ratio (for SAM3-Tracker)
    pub fn scaled_points(&self, scale_x: f32, scale_y: f32) -> Vec<[f32; 2]> {
        self.points
            .iter()
            .map(|kpt| [kpt.x() * scale_x, kpt.y() * scale_y])
            .collect()
    }

    /// Scale boxes and convert to xyxy (for SAM3-Tracker)
    pub fn scaled_boxes_xyxy(&self, scale_x: f32, scale_y: f32) -> Vec<[f32; 4]> {
        self.boxes
            .iter()
            .map(|hbb| {
                let (x1, y1, x2, y2) = hbb.xyxy();
                [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            })
            .collect()
    }

    pub fn class_name(&self) -> &str {
        &self.text
    }

    /// Parse coordinates from string
    fn parse_coords(s: &str) -> std::result::Result<Vec<f32>, String> {
        s.split(',')
            .map(|x| {
                x.trim()
                    .parse::<f32>()
                    .map_err(|e| format!("Invalid coordinate '{}': {}", x.trim(), e))
            })
            .collect()
    }
}

impl std::str::FromStr for Sam3Prompt {
    type Err = String;

    /// Parse from CLI format:
    /// - `"text"` - text only
    /// - `"text;pos:x,y,w,h"` - text + box
    /// - `"pos:x,y,w,h"` - box only, auto "visual"
    /// - `"pos:x,y"` - point (2 coords)
    /// - `"pos:x,y,w,h"` - box (4 coords, xywh)
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(';').collect();
        if parts.is_empty() {
            return Err("Empty prompt string".to_string());
        }

        let first = parts[0].trim();

        // Check if first part is geometry (pos:/neg:) or text
        let (text, is_auto_text, geo_parts) =
            if first.starts_with("pos:") || first.starts_with("neg:") {
                // No text provided, use "visual"
                (Self::VISUAL, true, parts.as_slice())
            } else {
                // First part is text
                (first, false, &parts[1..])
            };

        let mut prompt = Self::new(text);
        prompt.is_auto_text = is_auto_text;

        for part in geo_parts {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            if let Some(coords_str) = part.strip_prefix("pos:") {
                let coords = Self::parse_coords(coords_str)?;
                match coords.len() {
                    2 => prompt = prompt.with_positive_point(coords[0], coords[1]),
                    4 => {
                        prompt =
                            prompt.with_positive_box(coords[0], coords[1], coords[2], coords[3])
                    }
                    n => return Err(format!("pos: expects 2 (point) or 4 (box) coords, got {n}")),
                }
            } else if let Some(coords_str) = part.strip_prefix("neg:") {
                let coords = Self::parse_coords(coords_str)?;
                match coords.len() {
                    2 => prompt = prompt.with_negative_point(coords[0], coords[1]),
                    4 => {
                        prompt =
                            prompt.with_negative_box(coords[0], coords[1], coords[2], coords[3])
                    }
                    n => return Err(format!("neg: expects 2 (point) or 4 (box) coords, got {n}")),
                }
            } else {
                return Err(format!(
                    "Invalid format: '{part}'. Use 'pos:x,y' (point) or 'pos:x,y,w,h' (box)"
                ));
            }
        }

        Ok(prompt)
    }
}
