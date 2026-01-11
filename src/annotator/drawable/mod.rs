mod hbb;
mod keypoint;
mod mask;
mod obb;
mod polygon;
mod prob;

use crate::{DrawContext, Y};

/// Defines an interface for drawing objects on an image canvas.
pub trait Drawable {
    fn draw(&self, ctx: &DrawContext, canvas: &mut image::RgbaImage) -> anyhow::Result<()>;
}

impl Drawable for Y {
    fn draw(&self, ctx: &DrawContext, canvas: &mut image::RgbaImage) -> anyhow::Result<()> {
        self.masks().draw(ctx, canvas)?;
        self.polygons().draw(ctx, canvas)?;
        self.hbbs().draw(ctx, canvas)?;
        self.obbs().draw(ctx, canvas)?;
        self.keypoints().draw(ctx, canvas)?;
        self.keypointss().draw(ctx, canvas)?;
        self.probs().draw(ctx, canvas)?;

        Ok(())
    }
}

impl<T> Drawable for Vec<T>
where
    [T]: Drawable,
{
    fn draw(&self, ctx: &DrawContext, canvas: &mut image::RgbaImage) -> anyhow::Result<()> {
        self.as_slice().draw(ctx, canvas)
    }
}
