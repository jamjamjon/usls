use anyhow::Result;
use image::RgbaImage;

use crate::{DrawContext, Drawable, Prob, ProbStyle, TextLoc};

impl Drawable for [Prob] {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let default_style = ProbStyle::default();
        let style = ctx.prob_style.unwrap_or(&default_style);

        let mut renderer = ProbTextRenderer::new(canvas);
        for prob in self.iter() {
            renderer.render_text(ctx, canvas, prob, style)?;
        }

        Ok(())
    }
}

impl Drawable for Prob {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let default_style = ProbStyle::default();
        let style = self.style().or(ctx.prob_style).unwrap_or(&default_style);

        let mut renderer = ProbTextRenderer::new(canvas);
        renderer.render_text(ctx, canvas, self, style)
    }
}

struct ProbTextRenderer {
    height_acc: [f32; 2], // [top_left, top_right]
    canvas_dims: (f32, f32),
}

impl ProbTextRenderer {
    fn new(canvas: &RgbaImage) -> Self {
        Self {
            height_acc: [0.0, 0.0],
            canvas_dims: (canvas.width() as f32, canvas.height() as f32),
        }
    }

    fn render_text(
        &mut self,
        ctx: &DrawContext,
        canvas: &mut RgbaImage,
        prob: &Prob,
        style: &ProbStyle,
    ) -> Result<()> {
        let text_style = style.text_style();
        if !text_style.should_draw() {
            return Ok(());
        }

        let colors = ctx.resolve_prob_colors(style, prob.id());
        let label = prob.meta().label(
            text_style.id(),
            text_style.name(),
            text_style.confidence(),
            text_style.decimal_places(),
        );

        let rx = style.text_x_pos();
        let ry = style.text_y_pos();

        // Use box_size for proper positioning with TextStyleMode
        let text_mode = *text_style.mode();
        let font_size = text_style.font_size();
        let (bw, bh) = ctx
            .text_renderer
            .box_size_with(&label, &text_mode, font_size)?;
        let (bw, bh) = (bw as f32, bh as f32);

        // Calculate total height including outline thickness (drawn outside the box)
        let outline_extra = if text_style.draw_outline() {
            text_style.thickness() as f32 * 2.0 // outline is on both top and bottom
        } else {
            0.0
        };
        let total_h = bh + outline_extra;

        let (x, y) = match *text_style.loc() {
            TextLoc::InnerTopLeft => {
                let x = self.canvas_dims.0 * rx;
                self.height_acc[0] += total_h;
                let y = self.canvas_dims.1 * ry + self.height_acc[0];
                (x, y)
            }
            TextLoc::InnerTopRight => {
                let x = self.canvas_dims.0 * (1.0 - rx) - bw;
                self.height_acc[1] += total_h;
                let y = self.canvas_dims.1 * ry + self.height_acc[1];
                (x, y)
            }
            TextLoc::InnerTopCenter => {
                let x = (self.canvas_dims.0 - bw) / 2.0;
                self.height_acc[0] += total_h;
                let y = self.canvas_dims.1 * ry + self.height_acc[0];
                (x, y)
            }
            TextLoc::InnerBottomLeft => {
                let x = self.canvas_dims.0 * rx;
                self.height_acc[0] += total_h;
                let y = self.canvas_dims.1 * (1.0 - ry) - self.height_acc[0];
                (x, y)
            }
            TextLoc::InnerBottomRight => {
                let x = self.canvas_dims.0 * (1.0 - rx) - bw;
                self.height_acc[1] += total_h;
                let y = self.canvas_dims.1 * (1.0 - ry) - self.height_acc[1];
                (x, y)
            }
            TextLoc::InnerBottomCenter => {
                let x = (self.canvas_dims.0 - bw) / 2.0;
                self.height_acc[0] += total_h;
                let y = self.canvas_dims.1 * (1.0 - ry) - self.height_acc[0];
                (x, y)
            }
            TextLoc::Center => {
                let x = (self.canvas_dims.0 - bw) / 2.0;
                let y = (self.canvas_dims.1 - bh) / 2.0;
                (x, y)
            }
            // Center Left/Right positions
            TextLoc::InnerCenterLeft => {
                let x = self.canvas_dims.0 * rx;
                let y = (self.canvas_dims.1 - bh) / 2.0;
                (x, y)
            }
            TextLoc::InnerCenterRight => {
                let x = self.canvas_dims.0 * (1.0 - rx) - bw;
                let y = (self.canvas_dims.1 - bh) / 2.0;
                (x, y)
            }
            TextLoc::OuterCenterLeft => {
                let x = self.canvas_dims.0 * rx;
                let y = (self.canvas_dims.1 - bh) / 2.0;
                (x, y)
            }
            TextLoc::OuterCenterRight => {
                let x = self.canvas_dims.0 * (1.0 - rx) - bw;
                let y = (self.canvas_dims.1 - bh) / 2.0;
                (x, y)
            }
            // Outer bottom positions
            TextLoc::OuterBottomLeft => {
                let x = self.canvas_dims.0 * rx;
                self.height_acc[0] += total_h;
                let y = self.canvas_dims.1 * (1.0 - ry) - self.height_acc[0];
                (x, y)
            }
            TextLoc::OuterBottomRight => {
                let x = self.canvas_dims.0 * (1.0 - rx) - bw;
                self.height_acc[1] += total_h;
                let y = self.canvas_dims.1 * (1.0 - ry) - self.height_acc[1];
                (x, y)
            }
            TextLoc::OuterBottomCenter => {
                let x = (self.canvas_dims.0 - bw) / 2.0;
                self.height_acc[0] += total_h;
                let y = self.canvas_dims.1 * (1.0 - ry) - self.height_acc[0];
                (x, y)
            }
            // Outer positions fall back to inner equivalents for Prob
            TextLoc::OuterTopLeft => {
                let x = self.canvas_dims.0 * rx;
                self.height_acc[0] += total_h;
                let y = self.canvas_dims.1 * ry + self.height_acc[0];
                (x, y)
            }
            TextLoc::OuterTopRight => {
                let x = self.canvas_dims.0 * (1.0 - rx) - bw;
                self.height_acc[1] += total_h;
                let y = self.canvas_dims.1 * ry + self.height_acc[1];
                (x, y)
            }
            TextLoc::OuterTopCenter => {
                let x = (self.canvas_dims.0 - bw) / 2.0;
                self.height_acc[0] += total_h;
                let y = self.canvas_dims.1 * ry + self.height_acc[0];
                (x, y)
            }
        };

        let text_thickness = text_style.thickness();
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
        )
    }
}
