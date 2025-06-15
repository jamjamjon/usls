use aksr::Builder;
use anyhow::Result;

use crate::{DrawContext, Drawable, Image, Style, TextRenderer};

/// Annotator provides configuration for drawing annotations on images,
/// including styles, color palettes, and text rendering config.
#[derive(Clone, Builder)]
pub struct Annotator {
    prob_style: Option<Style>,
    hbb_style: Option<Style>,
    obb_style: Option<Style>,
    keypoint_style: Option<Style>,
    polygon_style: Option<Style>,
    mask_style: Option<Style>,
    text_renderer: TextRenderer,
}

impl Default for Annotator {
    fn default() -> Self {
        Self {
            text_renderer: TextRenderer::default(),
            prob_style: Some(Style::prob()),
            hbb_style: Some(Style::hbb()),
            obb_style: Some(Style::obb()),
            keypoint_style: Some(Style::keypoint()),
            polygon_style: Some(Style::polygon()),
            mask_style: Some(Style::mask()),
        }
    }
}

impl Annotator {
    /// Annotate an image with drawable objects
    pub fn annotate<T: Drawable>(&self, image: &Image, drawable: &T) -> Result<Image> {
        crate::elapsed_annotator!("annotate_total", {
            let ctx = crate::elapsed_annotator!("context_creation", {
                DrawContext {
                    text_renderer: &self.text_renderer,
                    prob_style: self.prob_style.as_ref(),
                    hbb_style: self.hbb_style.as_ref(),
                    obb_style: self.obb_style.as_ref(),
                    keypoint_style: self.keypoint_style.as_ref(),
                    polygon_style: self.polygon_style.as_ref(),
                    mask_style: self.mask_style.as_ref(),
                }
            });
            let mut rgba8 = crate::elapsed_annotator!("image_conversion", image.to_rgba8());
            crate::elapsed_annotator!("drawable_render", drawable.draw(&ctx, &mut rgba8)?);
            Ok(rgba8.into())
        })
    }

    pub fn with_font(mut self, path: &str) -> Result<Self> {
        self.text_renderer = self.text_renderer.with_font(path)?;
        Ok(self)
    }

    pub fn with_font_size(mut self, x: f32) -> Self {
        self.text_renderer = self.text_renderer.with_font_size(x);
        self
    }
}
