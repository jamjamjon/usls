mod hbb;
mod keypoint;
mod mask;
mod obb;
mod polygon;
mod prob;

use crate::{DrawContext, Style, Y};

/// Defines an interface for drawing objects on an image canvas
pub trait Drawable {
    fn get_local_style(&self) -> Option<&Style> {
        None
    }

    fn get_global_style<'a>(&self, _ctx: &'a DrawContext) -> Option<&'a Style> {
        None
    }

    fn get_id(&self) -> Option<usize> {
        None
    }

    #[allow(unused_variables)]
    fn draw_shapes_with_style(
        &self,
        ctx: &DrawContext,
        canvas: &mut image::RgbaImage,
        style: &Style,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn draw_texts_with_style(
        &self,
        ctx: &DrawContext,
        canvas: &mut image::RgbaImage,
        style: &Style,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn draw(&self, ctx: &DrawContext, canvas: &mut image::RgbaImage) -> anyhow::Result<()> {
        let style = ctx.update_style(
            self.get_local_style(),
            self.get_global_style(ctx),
            self.get_id(),
        );
        if style.visible() {
            self.draw_shapes_with_style(ctx, canvas, &style)?;
        }
        if style.text_visible() {
            self.draw_texts_with_style(ctx, canvas, &style)?;
        }

        Ok(())
    }
}

impl Drawable for Y {
    fn draw(&self, ctx: &DrawContext, canvas: &mut image::RgbaImage) -> anyhow::Result<()> {
        if let Some(masks) = self.masks() {
            masks.draw(ctx, canvas)?;
        }
        if let Some(polygons) = self.polygons() {
            polygons.draw(ctx, canvas)?;
        }
        if let Some(hbbs) = self.hbbs() {
            hbbs.draw(ctx, canvas)?;
        }
        if let Some(obbs) = self.obbs() {
            obbs.draw(ctx, canvas)?;
        }
        if let Some(keypoints) = self.keypoints() {
            keypoints.draw(ctx, canvas)?;
        }
        if let Some(keypointss) = self.keypointss() {
            keypointss.draw(ctx, canvas)?;
        }
        if let Some(probs) = self.probs() {
            probs.draw(ctx, canvas)?;
        }

        Ok(())
    }
}

impl<T> Drawable for Vec<T>
where
    [T]: Drawable,
{
    fn get_local_style(&self) -> Option<&Style> {
        self.as_slice().get_local_style()
    }

    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        self.as_slice().get_global_style(ctx)
    }

    fn get_id(&self) -> Option<usize> {
        self.as_slice().get_id()
    }

    fn draw_shapes_with_style(
        &self,
        ctx: &DrawContext,
        canvas: &mut image::RgbaImage,
        style: &Style,
    ) -> anyhow::Result<()> {
        self.as_slice().draw_shapes_with_style(ctx, canvas, style)
    }

    fn draw_texts_with_style(
        &self,
        ctx: &DrawContext,
        canvas: &mut image::RgbaImage,
        style: &Style,
    ) -> anyhow::Result<()> {
        self.as_slice().draw_texts_with_style(ctx, canvas, style)
    }

    fn draw(&self, ctx: &DrawContext, canvas: &mut image::RgbaImage) -> anyhow::Result<()> {
        self.as_slice().draw(ctx, canvas)
    }
}

// impl<T: Drawable> Drawable for Vec<T> {
//     fn draw_shapes(&self, ctx: &DrawContext, canvas: &mut image::RgbaImage) ->  anyhow::Result<()> {
//         self.iter().try_for_each(|x| x.draw_shapes(ctx, canvas))
//     }

//     fn draw_texts(&self, ctx: &DrawContext, canvas: &mut image::RgbaImage) ->  anyhow::Result<()> {
//         self.iter().try_for_each(|x| x.draw_texts(ctx, canvas))
//     }
// }
