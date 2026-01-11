use anyhow::Result;
use image::{Rgba, RgbaImage};

use crate::{draw_hbb, DrawContext, Drawable, Hbb, HbbStyle};

impl Drawable for [Hbb] {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}

impl Drawable for Hbb {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        // Get style: prefer instance style, fall back to global, or use default
        let default_style = HbbStyle::default();
        let style = self.style().or(ctx.hbb_style).unwrap_or(&default_style);

        if !style.visible() {
            return Ok(());
        }

        let colors = ctx.resolve_hbb_colors(style, self.id());

        // Draw fill
        if style.draw_fill() {
            let mut overlay = RgbaImage::new(canvas.width(), canvas.height());
            imageproc::drawing::draw_filled_rect_mut(
                &mut overlay,
                imageproc::rect::Rect::at(self.xmin().round() as i32, self.ymin().round() as i32)
                    .of_size(
                        (self.width().round() as u32).max(1),
                        (self.height().round() as u32).max(1),
                    ),
                Rgba(colors.fill.into()),
            );
            image::imageops::overlay(canvas, &overlay, 0, 0);
        }

        // Draw outline
        if style.draw_outline() {
            let short_side = self.width().min(self.height());
            let max_thickness = (short_side * style.thickness_max_ratio()).max(1.0) as usize;
            let thickness = style.thickness().min(max_thickness).max(1);
            let color = Rgba(colors.outline.into());
            let mode = *style.mode();
            let direction = *style.thickness_direction();

            let x1 = self.xmin().round();
            let y1 = self.ymin().round();
            let x2 = self.xmax().round();
            let y2 = self.ymax().round();
            draw_hbb(canvas, x1, y1, x2, y2, color, mode, thickness, direction);
        }

        // keypoints
        if let Some(keypoints) = self.keypoints() {
            keypoints.to_vec().draw(ctx, canvas)?;
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

            // Use box_size (includes padding) for positioning
            let font_size = text_style.font_size();
            let box_size = ctx
                .text_renderer
                .box_size_with(&label, &text_mode, font_size)?;
            let canvas_size = (canvas.width(), canvas.height());

            // Adjust bbox to match the visual bounds after thickness expansion
            let hbb_t = style.thickness() as f32;
            let expansion = match style.thickness_direction() {
                crate::ThicknessDirection::Outward => {
                    if hbb_t > 0.0 {
                        hbb_t - 1.0
                    } else {
                        0.0
                    }
                }
                crate::ThicknessDirection::Inward => 0.0,
                crate::ThicknessDirection::Centered => {
                    if hbb_t > 1.0 {
                        (hbb_t - 1.0) / 2.0
                    } else {
                        0.0
                    }
                }
            };
            let bbox = (
                self.xmin() - expansion,
                self.ymin() - expansion,
                self.xmax() + expansion,
                self.ymax() + expansion,
            );

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
