use crate::Drawable;
use anyhow::Result;
use image::{Rgba, RgbaImage};

use crate::{DrawContext, Polygon, Style, TextLoc};

impl Drawable for [Polygon] {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}

impl Drawable for Polygon {
    fn get_local_style(&self) -> Option<&Style> {
        self.style()
    }

    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        ctx.polygon_style
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
        let mut overlay = canvas.clone();

        // filled
        if style.draw_fill() {
            let polygon_i32 = self
                .polygon()
                .exterior()
                .points()
                .take(if self.is_closed() {
                    self.count() - 1
                } else {
                    self.count()
                })
                .map(|p| imageproc::point::Point::new(p.x() as i32, p.y() as i32))
                .collect::<Vec<_>>();

            imageproc::drawing::draw_polygon_mut(
                &mut overlay,
                &polygon_i32,
                Rgba(style.color().fill.unwrap().into()),
            );
        }
        image::imageops::overlay(canvas, &overlay, 0, 0);

        // contour
        if style.draw_outline() {
            let polygon_f32 = self
                .polygon()
                .exterior()
                .points()
                .take(if self.is_closed() {
                    self.count() - 1
                } else {
                    self.count()
                })
                .map(|p| imageproc::point::Point::new(p.x() as f32, p.y() as f32))
                .collect::<Vec<_>>();

            imageproc::drawing::draw_hollow_polygon_mut(
                canvas,
                &polygon_f32,
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
        if style.draw_text() {
            let label = self.meta().label(
                style.id(),
                style.name(),
                style.confidence(),
                style.decimal_places(),
            );

            // text loc
            let (x, y) = match style.text_loc() {
                TextLoc::Center => {
                    if let Some((x, y)) = self.centroid() {
                        (x, y)
                    } else {
                        (0.0, 0.0)
                    }
                }
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
