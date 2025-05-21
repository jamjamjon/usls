use crate::Drawable;
use anyhow::Result;
use image::{Rgba, RgbaImage};

use crate::{DrawContext, Obb, Style, TextLoc};

impl Drawable for [Obb] {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}

impl Drawable for Obb {
    fn get_local_style(&self) -> Option<&Style> {
        self.style()
    }

    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        ctx.obb_style
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
        if style.draw_outline() {
            for i in 0..self.vertices().len() {
                let p1 = self.vertices()[i];
                let p2 = self.vertices()[(i + 1) % self.vertices().len()];
                imageproc::drawing::draw_line_segment_mut(
                    canvas,
                    (p1[0], p1[1]),
                    (p2[0], p2[1]),
                    Rgba(style.color().outline.unwrap().into()),
                );
            }
        }

        if style.draw_fill() {
            let polygon = self.to_polygon();
            let polygon_i32 = polygon
                .polygon()
                .exterior()
                .points()
                .take(if polygon.is_closed() {
                    polygon.count() - 1
                } else {
                    polygon.count()
                })
                .map(|p| imageproc::point::Point::new(p.x() as i32, p.y() as i32))
                .collect::<Vec<_>>();

            imageproc::drawing::draw_polygon_mut(
                &mut overlay,
                &polygon_i32,
                Rgba(style.color().fill.unwrap().into()),
            );
            image::imageops::overlay(canvas, &overlay, 0, 0);
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
            let label = self.meta().label(
                style.id(),
                style.name(),
                style.confidence(),
                style.decimal_places(),
            );

            // text loc
            let (x, y) = match style.text_loc() {
                TextLoc::OuterTopRight => (self.top()[0], self.top()[1]),
                _ => todo!(),
            };

            // put text
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
