use crate::Drawable;
use anyhow::Result;
use image::RgbaImage;

use crate::{DrawContext, Prob, Style, TextLoc};

impl Drawable for [Prob] {
    fn draw_texts_with_style(
        &self,
        ctx: &DrawContext,
        canvas: &mut RgbaImage,
        _style: &Style,
    ) -> Result<()> {
        let mut renderer = ProbTextRenderer::new(canvas);
        for prob in self.iter() {
            let style = ctx.update_style(prob.style(), ctx.prob_style, prob.id());
            renderer.render_text(ctx, canvas, prob, &style)?;
        }

        Ok(())
    }
}

impl Drawable for Prob {
    fn get_local_style(&self) -> Option<&Style> {
        self.style()
    }

    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        ctx.prob_style
    }

    fn get_id(&self) -> Option<usize> {
        self.id()
    }

    fn draw_texts_with_style(
        &self,
        ctx: &DrawContext,
        canvas: &mut RgbaImage,
        style: &Style,
    ) -> Result<()> {
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
        style: &Style,
    ) -> Result<()> {
        if !style.draw_text() {
            return Ok(());
        }

        let label = prob.meta().label(
            style.id(),
            style.name(),
            style.confidence(),
            style.decimal_places(),
        );

        let (tw, th) = ctx.text_renderer.text_size(&label);
        let (tw, th) = (tw as f32, th as f32);
        let rx = style.text_x_pos();
        let ry = style.text_y_pos();

        let (x, y) = match style.text_loc() {
            TextLoc::InnerTopLeft => {
                let x = self.canvas_dims.0 * rx;
                self.height_acc[0] += th;
                let y = self.canvas_dims.1 * ry + self.height_acc[0];
                (x, y)
            }
            TextLoc::InnerTopRight => {
                let x = self.canvas_dims.0 * (1.0 - rx) - tw;
                self.height_acc[1] += th;
                let y = self.canvas_dims.1 * ry + self.height_acc[1];
                (x, y)
            }
            _ => unimplemented!(),
        };

        ctx.text_renderer.render(
            canvas,
            &label,
            x,
            y,
            style.color().text.unwrap(),
            style.color().text_bg.unwrap(),
        )
    }
}
