use crate::Drawable;
use anyhow::Result;
use image::{Rgba, RgbaImage};

use crate::{Color, DrawContext, Keypoint, Style, TextLoc};

impl Drawable for Keypoint {
    fn get_local_style(&self) -> Option<&Style> {
        self.style()
    }

    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        ctx.keypoint_style
    }

    fn get_id(&self) -> Option<usize> {
        self.id()
    }

    fn draw_shapes_with_style(
        &self,
        _ctx: &DrawContext,
        canvas: &mut RgbaImage,
        style: &Style,
    ) -> Result<()> {
        if self.confidence().is_none() || self.confidence().unwrap() == 0.0 {
            return Ok(());
        }

        if style.draw_fill() {
            let mut overlay = RgbaImage::new(canvas.width(), canvas.height());
            imageproc::drawing::draw_filled_circle_mut(
                &mut overlay,
                (self.x() as i32, self.y() as i32),
                style.radius() as i32,
                Rgba(style.color().fill.unwrap().into()),
            );
            image::imageops::overlay(canvas, &overlay, 0, 0);
        }

        if style.draw_outline() {
            imageproc::drawing::draw_hollow_circle_mut(
                canvas,
                (self.x() as i32, self.y() as i32),
                style.radius() as i32,
                Rgba(style.color().outline.unwrap().into()),
            );
        }

        Ok(())
    }

    fn draw_texts_with_style(
        &self,
        ctx: &DrawContext,
        canvas: &mut RgbaImage,
        style: &Style,
    ) -> Result<()> {
        if self.confidence().is_none() || self.confidence().unwrap() == 0.0 {
            return Ok(());
        }

        let (x, y) = match style.text_loc() {
            TextLoc::OuterTopRight => (
                self.x() + style.radius() as f32,
                self.y() - style.radius() as f32,
            ),
            _ => todo!(),
        };

        // label
        if style.draw_text() {
            let label = self.meta().label(
                style.id(),
                style.name(),
                style.confidence(),
                style.decimal_places(),
            );

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

impl Drawable for Vec<Keypoint> {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let nk = self.len();
        if nk > 0 {
            if let Some(skeleton) = ctx.skeleton {
                for connection in skeleton.iter() {
                    let (i, ii) = connection.indices;
                    let color = connection.color.unwrap_or(Color::white()); // TODO: default color
                    if i >= nk || ii >= nk {
                        continue;
                    }
                    let kpt1: &_ = &self[i];
                    let kpt2: &_ = &self[ii];

                    if kpt1.confidence().is_none()
                        || kpt1.confidence().unwrap() == 0.0
                        || kpt2.confidence().is_none()
                        || kpt2.confidence().unwrap() == 0.0
                    {
                        continue;
                    }

                    imageproc::drawing::draw_line_segment_mut(
                        canvas,
                        (kpt1.x(), kpt1.y()),
                        (kpt2.x(), kpt2.y()),
                        Rgba(color.into()),
                    );
                }
            }
        }

        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}

impl Drawable for Vec<Vec<Keypoint>> {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}
