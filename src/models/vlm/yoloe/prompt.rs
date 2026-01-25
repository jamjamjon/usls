use crate::{Color, ColorSource, Hbb, HbbStyle, HbbStyleMode, Image, TextStyle, TextStyleMode};

/// YOLOE Prompt - supports either textual OR visual prompts (mutually exclusive).
///
/// # CLI Format (FromStr parsing rules)
/// ```text
/// # Textual prompt (class names) - no images needed
/// -p "person"
/// -p "car"
///
/// # Visual prompt with box (xyxy format) - requires --visual-image
/// -p "xyxy:221.52,405.8,344.98,857.54,person"
///
/// # Visual prompt with box (xywh format)
/// -p "xywh:221.52,405.8,123.46,451.74,person"
///
/// # Visual prompt with box (cxcywh format)
/// -p "cxcywh:283.25,631.67,123.46,451.74,person"
///
/// # Visual prompt without class name (defaults to "visual")
/// -p "xyxy:221.52,405.8,344.98,857.54"
/// ```
///
/// ## Parsing Rules:
/// 1. If starts with `xyxy:`/`xywh:`/`cxcywh:` → visual prompt (box)
/// 2. Otherwise → textual prompt (class name)
/// 3. Textual and visual prompts are mutually exclusive - cannot mix them
/// 4. Visual prompts: `format:x1,y1,x2,y2[,class_name]`
/// 5. Class name is optional, defaults to "visual"
/// 6. Visual prompt uses single image with multiple boxes
/// 7. Boxes with duplicate class names share the same class ID (order-based)
#[derive(Debug, Clone)]
pub enum YOLOEPrompt {
    /// Textual prompt with class names
    Textual(Vec<String>),
    /// Visual prompt: (boxes, image) - single image with multiple boxes
    Visual(Vec<Hbb>, Image),
}

impl YOLOEPrompt {
    /// Default class name for visual prompts without explicit name
    pub const VISUAL: &'static str = "visual";

    /// Create an empty textual prompt.
    ///
    /// Use with builder methods like `.with_text()` or `.with_texts()` to add class names.
    ///
    /// # Examples
    /// ```ignore
    /// let prompt = YOLOEPrompt::textual()
    ///     .with_text("person")?
    ///     .with_texts(&["car", "bus"])?;
    /// ```
    pub fn textual() -> Self {
        Self::Textual(vec![])
    }

    /// Create an empty visual prompt with an image.
    ///
    /// Use with builder methods like `.with_hbb()` or `.with_hbbs()` to add bounding boxes.
    ///
    /// # Arguments
    /// * `image` - The reference image for visual prompting
    ///
    /// # Examples
    /// ```ignore
    /// let image = Image::try_read("prompt.jpg")?;
    /// let prompt = YOLOEPrompt::visual(image)
    ///     .with_hbb(hbb1)?
    ///     .with_hbbs(vec![hbb2, hbb3])?;
    /// ```
    pub fn visual(image: impl Into<Image>) -> Self {
        Self::Visual(vec![], image.into())
    }

    /// Append a single class name to textual prompt (builder pattern).
    ///
    /// Automatically deduplicates - if the text already exists, it won't be added again.
    ///
    /// # Arguments
    /// * `text` - Class name to add
    ///
    /// # Errors
    /// Returns error if called on a Visual prompt
    ///
    /// # Examples
    /// ```ignore
    /// let prompt = YOLOEPrompt::textual()
    ///     .with_text("person")?
    ///     .with_text("person")?;  // Won't duplicate
    /// ```
    pub fn with_text(self, text: impl Into<String>) -> anyhow::Result<Self> {
        match self {
            Self::Visual(..) => {
                anyhow::bail!("Visual prompt cannot be converted to textual prompt")
            }
            Self::Textual(mut texts) => {
                let text_str = text.into();
                if !texts.contains(&text_str) {
                    texts.push(text_str);
                }
                Ok(Self::Textual(texts))
            }
        }
    }

    /// Extend textual prompt with multiple class names (builder pattern).
    ///
    /// Automatically deduplicates - only new class names will be added.
    ///
    /// # Arguments
    /// * `texts` - Collection of class names to add
    ///
    /// # Errors
    /// Returns error if called on a Visual prompt
    ///
    /// # Examples
    /// ```ignore
    /// let prompt = YOLOEPrompt::textual()
    ///     .with_texts(vec!["person".to_string(), "car".to_string()])?;
    /// ```
    pub fn with_texts(self, texts: impl Into<Vec<String>>) -> anyhow::Result<Self> {
        match self {
            Self::Visual(..) => {
                anyhow::bail!("Visual prompt cannot be converted to textual prompt")
            }
            Self::Textual(mut existing) => {
                for text in texts.into() {
                    if !existing.contains(&text) {
                        existing.push(text);
                    }
                }
                Ok(Self::Textual(existing))
            }
        }
    }

    /// Replace the image of a visual prompt (builder pattern).
    ///
    /// Keeps existing bounding boxes but replaces the reference image.
    ///
    /// # Arguments
    /// * `image` - New reference image
    ///
    /// # Errors
    /// Returns error if called on a Textual prompt
    ///
    /// # Examples
    /// ```ignore
    /// let prompt = YOLOEPrompt::visual(old_image)
    ///     .with_hbb(hbb)?
    ///     .with_image(new_image)?;  // Replace image, keep boxes
    /// ```
    pub fn with_image(self, image: impl Into<Image>) -> anyhow::Result<Self> {
        match self {
            Self::Visual(boxes, _) => Ok(Self::Visual(boxes, image.into())),
            Self::Textual(_) => {
                anyhow::bail!("Textual prompt cannot be converted to visual prompt")
            }
        }
    }

    /// Append a single bounding box to visual prompt (builder pattern).
    ///
    /// # Arguments
    /// * `hbb` - Bounding box to add (must have a class name set)
    ///
    /// # Errors
    /// Returns error if called on a Textual prompt
    ///
    /// # Examples
    /// ```ignore
    /// let hbb = Hbb::from_xyxy(100.0, 100.0, 200.0, 200.0)
    ///     .with_name("person");
    /// let prompt = YOLOEPrompt::visual(image)
    ///     .with_hbb(hbb)?;
    /// ```
    pub fn with_hbb(self, hbb: impl Into<Hbb>) -> anyhow::Result<Self> {
        match self {
            Self::Textual(_) => {
                anyhow::bail!("Textual prompt cannot be converted to visual prompt")
            }
            Self::Visual(mut boxes, image) => {
                boxes.push(hbb.into());
                Ok(Self::Visual(boxes, image))
            }
        }
    }

    /// Extend visual prompt with multiple bounding boxes (builder pattern).
    ///
    /// # Arguments
    /// * `hbbs` - Collection of bounding boxes to add
    ///
    /// # Errors
    /// Returns error if called on a Textual prompt
    ///
    /// # Examples
    /// ```ignore
    /// let boxes = vec![hbb1, hbb2, hbb3];
    /// let prompt = YOLOEPrompt::visual(image)
    ///     .with_hbbs(boxes)?;
    /// ```
    pub fn with_hbbs(self, hbbs: impl Into<Vec<Hbb>>) -> anyhow::Result<Self> {
        match self {
            Self::Textual(_) => {
                anyhow::bail!("Textual prompt cannot be converted to visual prompt")
            }
            Self::Visual(mut boxes, image) => {
                boxes.extend(hbbs.into());
                Ok(Self::Visual(boxes, image))
            }
        }
    }

    /// Clear all content (texts or boxes) while preserving prompt type.
    ///
    /// For textual prompts, clears all class names.
    /// For visual prompts, clears all boxes and resets image to default.
    ///
    /// # Examples
    /// ```ignore
    /// let prompt = YOLOEPrompt::textual()
    ///     .with_text("person")?
    ///     .clear();  // Empty textual prompt
    /// ```
    pub fn clear(self) -> Self {
        match self {
            Self::Textual(_) => Self::Textual(vec![]),
            Self::Visual(_, _) => Self::Visual(vec![], Image::default()),
        }
    }

    /// Check if this is a textual prompt
    pub fn is_textual(&self) -> bool {
        matches!(self, Self::Textual(_))
    }

    /// Check if this is a visual prompt
    pub fn is_visual(&self) -> bool {
        matches!(self, Self::Visual(..))
    }

    /// Get class names preserving input order (works for both textual and visual prompts)
    /// Duplicates are removed but first occurrence order is maintained
    pub fn class_names(&self) -> Vec<String> {
        match self {
            Self::Textual(texts) => texts.clone(),
            Self::Visual(boxes, _) => {
                let mut names = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for hbb in boxes {
                    if let Some(name) = hbb.name() {
                        let name_str = name.to_string();
                        if seen.insert(name_str.clone()) {
                            names.push(name_str);
                        }
                    }
                }
                names
            }
        }
    }

    /// Get bounding boxes from visual prompt.
    ///
    /// # Returns
    /// * `Some(&[Hbb])` - Boxes if this is a visual prompt
    /// * `None` - If this is a textual prompt
    pub fn boxes(&self) -> Option<&[Hbb]> {
        match self {
            Self::Visual(boxes, _) => Some(boxes),
            Self::Textual(_) => None,
        }
    }

    /// Get reference image from visual prompt.
    ///
    /// # Returns
    /// * `Some(&Image)` - Image if this is a visual prompt
    /// * `None` - If this is a textual prompt
    pub fn image(&self) -> Option<&Image> {
        match self {
            Self::Visual(_, image) => Some(image),
            Self::Textual(_) => None,
        }
    }

    /// Draw visual prompt boxes on the reference image.
    ///
    /// Creates an annotated copy of the reference image with all bounding boxes drawn.
    ///
    /// # Arguments
    /// * `annotator` - Annotator instance for rendering boxes
    ///
    /// # Returns
    /// Annotated image with all boxes drawn
    ///
    /// # Errors
    /// Returns error if called on a Textual prompt or if annotation fails
    ///
    /// # Examples
    /// ```ignore
    /// let annotated = prompt.draw(&annotator)?;
    /// annotated.save("prompt_visualization.jpg")?;
    /// ```
    #[cfg(feature = "annotator")]
    pub fn draw(&self, annotator: &crate::Annotator) -> anyhow::Result<Image> {
        match self {
            Self::Textual(_) => anyhow::bail!("Textual prompt cannot be drawn"),
            Self::Visual(boxes, image) => {
                let mut annotated = image.clone();
                for hbb in boxes {
                    annotated = annotator.annotate(&annotated, hbb)?;
                }
                Ok(annotated)
            }
        }
    }

    /// Parse box format: "x1,y1,x2,y2[,class_name]"
    fn parse_box_with_name(s: &str) -> anyhow::Result<([f32; 4], Option<String>)> {
        let parts: Vec<&str> = s.splitn(5, ',').collect();
        if parts.len() < 4 {
            anyhow::bail!(format!(
                "Box requires at least 4 coordinates, got {}",
                parts.len()
            ));
        }

        let coords: Vec<f32> = parts[..4]
            .iter()
            .map(|p| {
                p.trim()
                    .parse::<f32>()
                    .map_err(|e| anyhow::anyhow!("Invalid coordinate '{}': {}", p.trim(), e))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let name = if parts.len() > 4 {
            let name_str = parts[4].trim();
            if name_str.is_empty() {
                None
            } else {
                Some(name_str.to_string())
            }
        } else {
            None
        };

        Ok(([coords[0], coords[1], coords[2], coords[3]], name))
    }
}

impl std::str::FromStr for YOLOEPrompt {
    type Err = anyhow::Error;

    /// Parse a single prompt string:
    /// - `"person"` - textual prompt (class name)
    /// - `"xyxy:x1,y1,x2,y2[,class_name]"` - visual prompt box (requires image via parse)
    /// - `"xywh:x,y,w,h[,class_name]"` - visual prompt box
    /// - `"cxcywh:cx,cy,w,h[,class_name]"` - visual prompt box
    ///
    /// Note: This only parses the string format. Visual prompts need image set via parse().
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            anyhow::bail!("Empty prompt string");
        }

        let create_visual = |coords_str: &str,
                             constructor: fn(f32, f32, f32, f32) -> Hbb|
         -> Result<Self, anyhow::Error> {
            let (coords, name) = Self::parse_box_with_name(coords_str)?;
            // .map_err(|e| anyhow::anyhow!(e))?;
            let class_name = name.unwrap_or_else(|| Self::VISUAL.to_string());
            let color = Color::cyan();

            let hbb = constructor(coords[0], coords[1], coords[2], coords[3])
                .with_name(&class_name)
                .with_confidence(1.0)
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
                );

            Ok(Self::Visual(vec![hbb], Image::default()))
        };

        if let Some(coords_str) = s.strip_prefix("xyxy:") {
            create_visual(coords_str, Hbb::from_xyxy)
        } else if let Some(coords_str) = s.strip_prefix("xywh:") {
            create_visual(coords_str, Hbb::from_xywh)
        } else if let Some(coords_str) = s.strip_prefix("cxcywh:") {
            create_visual(coords_str, Hbb::from_cxcywh)
        } else {
            // Textual prompt (class name)
            Ok(Self::Textual(vec![s.to_string()]))
        }
    }
}

impl YOLOEPrompt {
    /// Create from CLI arguments (multiple -p strings + optional image path)
    ///
    /// Rules:
    /// - All prompts must be either textual OR visual (mutually exclusive)
    /// - Textual: no image needed
    /// - Visual: requires single image with multiple boxes
    /// - Boxes with duplicate class names share the same class ID (order-based)
    pub fn parse(prompts: &[String], visual_image: Option<&str>) -> anyhow::Result<Self> {
        if prompts.is_empty() {
            anyhow::bail!("No prompts provided");
        }

        let parsed: Vec<YOLOEPrompt> = prompts
            .iter()
            .map(|s| s.parse::<YOLOEPrompt>())
            .collect::<Result<Vec<_>, _>>()?;

        // Check if all are textual or all are visual
        let all_textual = parsed.iter().all(|p| p.is_textual());
        let all_visual = parsed.iter().all(|p| p.is_visual());

        if !all_textual && !all_visual {
            anyhow::bail!(
                "Cannot mix textual and visual prompts. Use either class names or boxes, not both"
            );
        }

        if all_textual {
            // Merge all textual prompts using builder pattern
            let mut texts = Vec::new();
            for p in parsed {
                if let Self::Textual(t) = p {
                    texts.extend(t);
                }
            }
            Ok(Self::Textual(texts))
        } else {
            // Merge all visual prompts (preserve input order)
            let mut boxes = Vec::new();
            for p in parsed {
                if let Self::Visual(b, _) = p {
                    boxes.extend(b);
                }
            }

            // Load single image
            let image_path = visual_image.ok_or_else(|| {
                anyhow::anyhow!("Visual prompts require --prompt-image to be specified")
            })?;
            let image = Image::try_read(image_path)?;

            // Use builder pattern (class IDs will be assigned by the model based on class_names() order)
            Self::visual(image).with_hbbs(boxes)
        }
    }
}
