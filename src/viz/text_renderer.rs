use ab_glyph::{FontArc, PxScale};
use aksr::Builder;
use anyhow::Result;
use image::{Rgba, RgbaImage};

use crate::{Color, Hub};

/// Text rendering engine with font management and styling capabilities.
#[derive(Builder, Clone, Debug)]
pub struct TextRenderer {
    #[args(except(setter))]
    font: FontArc,
    font_size: f32,
    _scale: f32,
}

impl Default for TextRenderer {
    fn default() -> Self {
        Self {
            font: Self::load_font(None)
                .unwrap_or_else(|err| panic!("Failed to load font: {}", err)),
            font_size: 24.0,
            _scale: 6.666667,
        }
    }
}

impl TextRenderer {
    /// Load custom font
    fn load_font(path: Option<&str>) -> Result<FontArc> {
        let path_font = match path {
            None => Hub::default().try_fetch("fonts/JetBrainsMono-Regular.ttf")?,
            Some(p) => p.into(),
        };
        let buf = std::fs::read(path_font)?;
        let font = FontArc::try_from_vec(buf)?;

        Ok(font)
    }

    pub fn with_font(mut self, path: &str) -> Result<Self> {
        self.font = Self::load_font(Some(path))?;

        Ok(self)
    }

    pub fn text_size(&self, text: &str) -> (u32, u32) {
        let scale = PxScale::from(self.font_size);
        let (text_w, text_h) = imageproc::drawing::text_size(scale, &self.font, text);
        let text_h = text_h + text_h / 3;

        (text_w, text_h)
    }

    pub fn render(
        &self,
        img: &mut RgbaImage,
        text: &str,
        x: f32,
        y: f32,
        color: Color,
        background_color: Color,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let scale = PxScale::from(self.font_size);
        let (text_w, text_h) = imageproc::drawing::text_size(scale, &self.font, text);
        let text_h = text_h + text_h / 3;
        let (left, top) = self.calculate_position(x, y, text_w, text_h, img.width());

        imageproc::drawing::draw_filled_rect_mut(
            img,
            imageproc::rect::Rect::at(left, top).of_size(text_w, text_h),
            Rgba(background_color.into()),
        );

        imageproc::drawing::draw_text_mut(
            img,
            Rgba(color.into()),
            left,
            top + self.calculate_text_offset(),
            scale,
            &self.font,
            text,
        );

        Ok(())
    }

    fn calculate_position(
        &self,
        x: f32,
        y: f32,
        text_w: u32,
        text_h: u32,
        img_width: u32,
    ) -> (i32, i32) {
        let top = if y > text_h as f32 {
            (y.round() as u32 - text_h) as i32
        } else {
            0
        };

        let mut left = x as i32;
        if left + text_w as i32 > img_width as i32 {
            left = img_width as i32 - text_w as i32;
        }

        (left, top)
    }

    fn calculate_text_offset(&self) -> i32 {
        -(self.font_size / self._scale).floor() as i32 + 1
    }
}
