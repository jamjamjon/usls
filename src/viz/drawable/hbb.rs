use crate::Drawable;
use anyhow::Result;
use image::{Rgba, RgbaImage};

use crate::{DrawContext, Hbb, Style, TextLoc};

impl Drawable for [Hbb] {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}

impl Drawable for Hbb {
    fn get_local_style(&self) -> Option<&Style> {
        self.style()
    }

    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        ctx.hbb_style
    }

    fn get_id(&self) -> Option<usize> {
        self.id()
    }

    fn draw_shapes_with_style(
        &self,
        ctx: &DrawContext,
        canvas: &mut RgbaImage,
        style: &Style,
    ) -> Result<()> {
        if style.draw_fill() {
            let mut overlay = RgbaImage::new(canvas.width(), canvas.height());
            imageproc::drawing::draw_filled_rect_mut(
                &mut overlay,
                imageproc::rect::Rect::at(self.xmin().round() as i32, self.ymin().round() as i32)
                    .of_size(
                        (self.width().round() as u32).max(1),
                        (self.height().round() as u32).max(1),
                    ),
                Rgba(style.color().fill.unwrap().into()),
            );
            image::imageops::overlay(canvas, &overlay, 0, 0);
        }

        if style.draw_outline() {
            let short_side_threshold =
                self.width().min(self.height()) * style.thickness_threshold();
            let thickness = style.thickness().min(short_side_threshold as usize).max(1);
            for i in 0..thickness {
                imageproc::drawing::draw_hollow_rect_mut(
                    canvas,
                    imageproc::rect::Rect::at(
                        (self.xmin().round() as i32) - (i as i32),
                        (self.ymin().round() as i32) - (i as i32),
                    )
                    .of_size(
                        ((self.width().round() as u32) + (2 * i as u32)).max(1),
                        ((self.height().round() as u32) + (2 * i as u32)).max(1),
                    ),
                    Rgba(style.color().outline.unwrap().into()),
                );
            }
        }

        // keypoints
        if let Some(keypoints) = self.keypoints() {
            keypoints.to_vec().draw(ctx, canvas)?;
        }

        Ok(())
    }

    fn draw_texts_with_style(
        &self,
        ctx: &DrawContext,
        canvas: &mut RgbaImage,
        style: &Style,
    ) -> Result<()> {
        // label
        if style.draw_text() {
            let short_side_threshold =
                self.width().min(self.height()) * style.thickness_threshold();
            let thickness = style.thickness().min(short_side_threshold as usize).max(1);

            let label = self.meta().label(
                style.id(),
                style.name(),
                style.confidence(),
                style.decimal_places(),
            );

            let (x, y) = match style.text_loc() {
                TextLoc::OuterTopLeft => (
                    (self.xmin().round() as i32 - (thickness - 1) as i32).max(0) as f32,
                    (self.ymin().round() as i32 - (thickness - 1) as i32).max(0) as f32,
                ),
                TextLoc::Center => (self.cx().round(), self.cy().round()),
                _ => todo!(),
            };

            ctx.text_renderer.render(
                canvas,
                &label,
                x,
                y,
                style.color().text.unwrap(),
                style.color().text_bg.unwrap(),
            )?;
        }

        Ok(())
    }
}
