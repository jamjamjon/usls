/// SAM3 Prompt - ONE text + N boxes
///
/// Each prompt represents a single query:
/// - `text`: The text description (use "visual" for box-only mode)
/// - `boxes`: Bounding boxes in xywh format
/// - `labels`: Box labels (1=positive, 0=negative)
///
/// # Examples
/// ```
/// // Text only
/// Sam3Prompt::new("cat");
///
/// // Box only (defaults to "visual")
/// Sam3Prompt::new("visual")
///     .with_box(59.0, 144.0, 17.0, 19.0, true);
///
/// // Text + negative box
/// Sam3Prompt::new("handle")
///     .with_box(40.0, 183.0, 278.0, 21.0, false);
/// ```
#[derive(Debug, Clone)]
pub struct Sam3Prompt {
    /// Text prompt (single text per query)
    pub text: String,
    /// Bounding boxes in xywh format: [x, y, width, height]
    pub boxes: Vec<[f32; 4]>,
    /// Box labels: 1 = positive, 0 = negative
    pub labels: Vec<i64>,
}

impl Sam3Prompt {
    /// Default class name for box-only prompts
    pub const VISUAL: &'static str = "visual";
}

impl Default for Sam3Prompt {
    fn default() -> Self {
        Self {
            text: Self::VISUAL.to_string(),
            boxes: Vec::new(),
            labels: Vec::new(),
        }
    }
}

impl Sam3Prompt {
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            boxes: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// Create a box-only prompt (text defaults to "visual")
    pub fn visual() -> Self {
        Self::new(Self::VISUAL)
    }

    pub fn with_text(mut self, text: &str) -> Self {
        self.text = text.to_string();
        self
    }

    /// Add a bounding box with label (xywh: x, y, width, height)
    pub fn with_box(mut self, x: f32, y: f32, w: f32, h: f32, positive: bool) -> Self {
        self.boxes.push([x, y, w, h]);
        self.labels.push(if positive { 1 } else { 0 });
        self
    }

    /// Add a positive bounding box
    pub fn with_positive_box(self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.with_box(x, y, w, h, true)
    }

    /// Add a negative bounding box
    pub fn with_negative_box(self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.with_box(x, y, w, h, false)
    }

    /// Check if prompt has any boxes
    pub fn has_boxes(&self) -> bool {
        !self.boxes.is_empty()
    }

    /// Check if prompt has at least one positive box
    pub fn has_positive_box(&self) -> bool {
        self.labels.iter().any(|&l| l == 1)
    }

    /// Check if this is a box-only prompt (text is "visual")
    pub fn is_visual(&self) -> bool {
        self.text == Self::VISUAL
    }

    /// Check if geometry should be used:
    /// - Box-only ("visual"): requires at least one positive box
    /// - Text + boxes: any boxes are valid (text provides anchor)
    pub fn should_use_geometry(&self) -> bool {
        if self.is_visual() {
            self.has_positive_box()
        } else {
            self.has_boxes()
        }
    }

    pub fn normalized_boxes(&self, image_width: f32, image_height: f32) -> Vec<[f32; 4]> {
        self.boxes
            .iter()
            .map(|&[x, y, w, h]| {
                let cx = (x + w / 2.0) / image_width;
                let cy = (y + h / 2.0) / image_height;
                let nw = w / image_width;
                let nh = h / image_height;
                [cx, cy, nw, nh]
            })
            .collect()
    }

    pub fn class_name(&self) -> &str {
        &self.text
    }

    fn parse_coords(s: &str) -> std::result::Result<[f32; 4], String> {
        let coords: Vec<f32> = s
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| format!("Invalid coordinates '{}': {}", s, e))?;

        if coords.len() != 4 {
            return Err(format!(
                "Expected 4 coordinates (x,y,w,h), got {}",
                coords.len()
            ));
        }

        Ok([coords[0], coords[1], coords[2], coords[3]])
    }
}

impl std::str::FromStr for Sam3Prompt {
    type Err = String;

    /// Parse from CLI format: "text;pos:x,y,w,h;neg:x,y,w,h"
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(';').collect();
        if parts.is_empty() {
            return Err("Empty prompt string".to_string());
        }

        let text = parts[0].trim();
        let mut prompt = Self::new(text);

        for part in parts.iter().skip(1) {
            let part = part.trim();
            if let Some(coords) = part.strip_prefix("pos:") {
                let [x, y, w, h] = Self::parse_coords(coords)?;
                prompt = prompt.with_positive_box(x, y, w, h);
            } else if let Some(coords) = part.strip_prefix("neg:") {
                let [x, y, w, h] = Self::parse_coords(coords)?;
                prompt = prompt.with_negative_box(x, y, w, h);
            } else {
                return Err(format!(
                    "Invalid box format: '{}'. Use 'pos:x,y,w,h' or 'neg:x,y,w,h'",
                    part
                ));
            }
        }

        Ok(prompt)
    }
}
