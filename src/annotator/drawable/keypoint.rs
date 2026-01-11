use anyhow::Result;
use image::{Rgba, RgbaImage};

use crate::{
    draw_keypoint_outline_thick, draw_keypoint_shape, draw_line_solid, draw_line_solid_thick,
    Color, DrawContext, Drawable, Keypoint, KeypointStyle,
};

impl Drawable for Keypoint {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        if self.confidence().is_none_or(|conf| conf == 0.0) {
            return Ok(());
        }

        let default_style = KeypointStyle::default();
        let style = self
            .style()
            .or(ctx.keypoint_style)
            .unwrap_or(&default_style);

        if !style.visible() {
            return Ok(());
        }

        let colors = ctx.resolve_keypoint_colors(style, self.id());
        let mode = *style.mode();

        // Glow mode: draw directly on canvas
        if matches!(mode, crate::KeypointStyleMode::Glow { .. }) {
            draw_keypoint_shape(
                canvas,
                self.x(),
                self.y(),
                style.radius(),
                Rgba(colors.fill.into()),
                mode,
                true,
            );
        } else {
            // Other modes: use overlay for proper alpha blending
            if style.draw_fill() {
                let mut overlay = RgbaImage::new(canvas.width(), canvas.height());
                draw_keypoint_shape(
                    &mut overlay,
                    self.x(),
                    self.y(),
                    style.radius(),
                    Rgba(colors.fill.into()),
                    mode,
                    true,
                );
                image::imageops::overlay(canvas, &overlay, 0, 0);
            }

            if style.draw_outline() {
                draw_keypoint_outline_thick(
                    canvas,
                    self.x(),
                    self.y(),
                    style.radius(),
                    style.thickness(),
                    Rgba(colors.outline.into()),
                    mode,
                );
            }
        }

        // Draw text
        let text_style = style.text_style();
        if style.text_visible() && text_style.should_draw() {
            let label = self.meta().label(
                text_style.id(),
                text_style.name(),
                text_style.confidence(),
                text_style.decimal_places(),
            );

            let text_mode = *text_style.mode();
            let text_thickness = text_style.thickness();

            // Calculate bounding box including outline thickness (extends outward)
            let r = style.radius() as f32;
            let outline_expansion = if style.draw_outline() {
                (style.thickness() as f32).max(1.0) - 1.0
            } else {
                0.0
            };
            let total_r = r + outline_expansion;
            let bbox = (
                self.x() - total_r,
                self.y() - total_r,
                self.x() + total_r,
                self.y() + total_r,
            );

            let font_size = text_style.font_size();
            let box_size = ctx
                .text_renderer
                .box_size_with(&label, &text_mode, font_size)?;
            let canvas_size = (canvas.width(), canvas.height());

            // Add text box thickness as offset for OuterTop positions
            let text_offset = if matches!(
                *text_style.loc(),
                crate::TextLoc::OuterTopLeft
                    | crate::TextLoc::OuterTopCenter
                    | crate::TextLoc::OuterTopRight
            ) {
                Some(text_thickness as f32)
            } else {
                None
            };
            let (x, y) =
                text_style
                    .loc()
                    .compute_anchor(bbox, box_size, canvas_size, None, text_offset);

            ctx.text_renderer.render_styled_with(
                canvas,
                &label,
                x,
                y,
                colors.text,
                colors.text_bg_fill,
                colors.text_bg_outline,
                text_mode,
                text_style.draw_fill(),
                text_style.draw_outline(),
                text_thickness,
                font_size,
            )?;
        }

        Ok(())
    }
}

impl Drawable for [Keypoint] {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let nk = self.len();
        if nk > 0 {
            let default_style = KeypointStyle::default();
            let style = ctx.keypoint_style.unwrap_or(&default_style);

            if let Some(skeleton) = style.skeleton() {
                let thickness = style.skeleton_thickness();
                for connection in skeleton.iter() {
                    let (i, ii) = connection.indices;
                    let color = connection.color.unwrap_or(Color::white());
                    if i >= nk || ii >= nk {
                        continue;
                    }
                    let kpt1 = &self[i];
                    let kpt2 = &self[ii];

                    if kpt1.confidence().is_none_or(|conf| conf == 0.0)
                        || kpt2.confidence().is_none_or(|conf| conf == 0.0)
                    {
                        continue;
                    }

                    if thickness <= 1 {
                        draw_line_solid(
                            canvas,
                            (kpt1.x(), kpt1.y()),
                            (kpt2.x(), kpt2.y()),
                            Rgba(color.into()),
                        );
                    } else {
                        draw_line_solid_thick(
                            canvas,
                            (kpt1.x(), kpt1.y()),
                            (kpt2.x(), kpt2.y()),
                            Rgba(color.into()),
                            thickness,
                        );
                    }
                }
            }
        }

        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}

impl Drawable for [Vec<Keypoint>] {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}
