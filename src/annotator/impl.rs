//! Annotator module for visualizing inference results with customizable styles

use crate::{
    DrawContext, Drawable, HbbStyle, Image, KeypointStyle, MaskStyle, ObbStyle, PolygonStyle,
    ProbStyle, TextRenderer,
};

/// Visualize inference results by drawing bounding boxes, masks, and keypoints directly onto images.
///
/// # 🖍️ Annotator
///
/// A comprehensive annotation engine for computer vision inference results. Provides
/// customizable styles for different annotation types and supports complex visualizations
/// with multiple drawable objects on a single image.
///
/// ## Supported Annotations
///
/// - **Horizontal Bounding Boxes (HBB)**: Standard axis-aligned rectangles with customizable colors, thickness, and labels
/// - **Oriented Bounding Boxes (OBB)**: Rotated rectangles for precise object boundaries
/// - **Keypoints**: Individual points with optional skeleton connections for pose estimation
/// - **Polygons**: Arbitrary shapes for segmentation and contour visualization
/// - **Masks**: Alpha-blended segmentation masks with customizable opacity
/// - **Probabilities**: Classification confidence scores with positioning options
///
/// ## Features
///
/// - **Styles**: Customizable colors, thickness, transparency, and label positions
/// - **Support**: Alpha-blended masks, oriented boxes, and skeleton drawing
/// - **Performance**: Optimized rendering with minimal memory allocation
/// - **Flexibility**: Builder pattern for easy configuration and method chaining
///
/// ## Examples
///
/// ### Basic Usage
///
/// ```no_run
/// use usls::{Annotator, HbbStyle, KeypointStyle, Y, Image};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let annotator = Annotator::default()
///     .with_hbb_style(HbbStyle::default())
///     .with_keypoint_style(KeypointStyle::default());
///
/// let image = Image::default();
/// let y = Y::default();
/// let annotated = annotator.annotate(&image, &y)?;
/// # Ok(())
/// # }
/// ```
///
/// ### Custom Styling
///
/// ```no_run
/// use usls::{Annotator, HbbStyle, ColorSource, Image, Y};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let annotator = Annotator::default()
///     .with_hbb_style(
///         HbbStyle::default()
///             .with_thickness(4)
///             .with_draw_fill(true)
///             .with_fill_color(ColorSource::AutoAlpha(50))
///     )
///     .with_font_size(16.0);
///
/// let image = Image::default();
/// let detections = Y::default();
/// let result = annotator.annotate(&image, &detections)?;
/// # Ok(())
/// # }
/// ```
///
/// ### Combined Annotations
///
/// ```no_run
/// use usls::{Annotator, HbbStyle, KeypointStyle, PolygonStyle, ProbStyle, SKELETON_COCO_19, Y, Image, Hbb, Keypoint, Polygon, Prob};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let detections = vec![Hbb::default()];
/// let pose_keypoints = vec![Keypoint::default()];
/// let segmentations = vec![Polygon::default()];
/// let classifications = vec![Prob::default()];
///
/// let y = Y::default()
///     .with_hbbs(&detections)
///     .with_keypoints(&pose_keypoints)
///     .with_polygons(&segmentations)
///     .with_probs(&classifications);
///
/// let annotator = Annotator::default()
///     .with_hbb_style(HbbStyle::default().with_thickness(3))
///     .with_keypoint_style(
///         KeypointStyle::default()
///             .with_skeleton(SKELETON_COCO_19.into())
///             .with_radius(4)
///     )
///     .with_polygon_style(PolygonStyle::default())
///     .with_prob_style(ProbStyle::default());
///
/// let image = Image::default();
/// let result = annotator.annotate(&image, &y)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Performance
///
/// - Uses efficient RGBA image conversion for minimal overhead
/// - Supports batch rendering of multiple annotation types
/// - Built-in performance metrics via `elapsed_annotator!` macros
/// - Memory-efficient style sharing across multiple annotations
///
/// ## Customization
///
/// Each annotation type supports extensive customization:
/// - **Colors**: Auto-generated, fixed, or alpha-blended color sources
/// - **Thickness**: Configurable line widths with dynamic scaling
/// - **Labels**: Customizable text positioning and visibility
/// - **Transparency**: Alpha blending for masks and filled shapes
/// - **Fonts**: Custom font loading and size configuration
///
/// 📘 **Guide**: [`docs/annotator.md`](../../docs/annotator.md)
#[derive(Clone, aksr::Builder)]
pub struct Annotator {
    prob_style: Option<ProbStyle>,
    hbb_style: Option<HbbStyle>,
    obb_style: Option<ObbStyle>,
    keypoint_style: Option<KeypointStyle>,
    polygon_style: Option<PolygonStyle>,
    mask_style: Option<MaskStyle>,
    text_renderer: TextRenderer,
}

impl Default for Annotator {
    fn default() -> Self {
        Self {
            prob_style: Some(ProbStyle::default()),
            hbb_style: Some(HbbStyle::default()),
            obb_style: Some(ObbStyle::default()),
            keypoint_style: Some(KeypointStyle::default()),
            polygon_style: Some(PolygonStyle::default()),
            mask_style: Some(MaskStyle::default()),
            text_renderer: TextRenderer::default(),
        }
    }
}

impl Annotator {
    /// Renders annotations onto an image with the configured styles.
    ///
    /// This is the core method that applies all configured annotation styles
    /// to draw drawable objects onto the provided image.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image to annotate (will not be modified)
    /// * `drawable` - Any object implementing the `Drawable` trait (e.g., `Y`, `Hbb`, `Keypoint`, etc.)
    ///
    /// # Returns
    ///
    /// Returns a new `Image` with the annotations rendered.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use usls::{Annotator, Y, Image};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let annotator = Annotator::default();
    /// let image = Image::default();
    /// let y = Y::default();
    /// let result = annotator.annotate(&image, &y)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn annotate<T: Drawable>(&self, image: &Image, drawable: &T) -> anyhow::Result<Image> {
        crate::elapsed_annotator!("annotate_total", {
            let mut rgba8 = crate::elapsed_annotator!("RgbImage->RgbaImage", image.to_rgba8());
            self.draw_inplace(&mut rgba8, drawable)?;
            Ok(rgba8.into())
        })
    }

    pub fn draw_inplace<T: Drawable>(
        &self,
        rgba8: &mut image::RgbaImage,
        drawable: &T,
    ) -> anyhow::Result<()> {
        let ctx = crate::elapsed_annotator!("context_creation", self.create_context());
        crate::elapsed_annotator!("drawable_render", drawable.draw(&ctx, rgba8)?);
        Ok(())
    }

    fn create_context(&self) -> DrawContext {
        DrawContext {
            text_renderer: &self.text_renderer,
            prob_style: self.prob_style.as_ref(),
            hbb_style: self.hbb_style.as_ref(),
            obb_style: self.obb_style.as_ref(),
            keypoint_style: self.keypoint_style.as_ref(),
            polygon_style: self.polygon_style.as_ref(),
            mask_style: self.mask_style.as_ref(),
        }
    }

    /// Sets a custom font for text rendering.
    ///
    /// Loads a font file from the specified path and configures the text renderer
    /// to use it for all annotation labels and text.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the font file (TTF, OTF, etc.)
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` for method chaining, or an error if font loading fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use usls::Annotator;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let annotator = Annotator::default()
    ///     .with_font("./fonts/arial.ttf")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Notes
    ///
    /// - Font loading is performed synchronously
    /// - The font file must be accessible at the specified path
    /// - Prints the loaded font path on success
    pub fn with_font(mut self, path: &str) -> anyhow::Result<Self> {
        self.text_renderer = self.text_renderer.with_font(path)?;
        println!("font: {path:?}");
        Ok(self)
    }

    /// Sets the font size for text rendering.
    ///
    /// Configures the size of text used in annotation labels, confidence scores,
    /// and other text elements.
    ///
    /// # Arguments
    ///
    /// * `x` - Font size in points (e.g., 12.0, 16.0, 24.0)
    ///
    /// # Returns
    ///
    /// Returns `self` for method chaining.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use usls::Annotator;
    ///
    /// let annotator = Annotator::default()
    ///     .with_font_size(16.0); // 16pt font
    /// ```
    ///
    /// # Notes
    ///
    /// - Larger font sizes may require more space for text positioning
    /// - Font size affects all text elements in annotations
    /// - Recommended range: 10.0 - 32.0 for most use cases
    pub fn with_font_size(mut self, x: f32) -> Self {
        self.text_renderer = self.text_renderer.with_font_size(x);
        self
    }
}
