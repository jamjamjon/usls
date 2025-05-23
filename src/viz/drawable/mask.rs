use crate::Drawable;
use anyhow::Result;
use image::{DynamicImage, Rgba, RgbaImage};

use crate::{Color, ColorMap256, DrawContext, Mask, Style};

fn render_mask(mask: &Mask, colormap256: Option<&ColorMap256>) -> DynamicImage {
    if let Some(colormap256) = colormap256 {
        let luma = imageproc::map::map_colors(mask.mask(), |p| {
            let idx = p[0];
            image::Rgb(colormap256.data()[idx as usize].rgb().into())
        });
        luma.into()
    } else {
        mask.mask().clone().into()
    }
}

fn apply_mask(origin: &RgbaImage, mask: &Mask, background_color: Option<Color>) -> DynamicImage {
    let bg = background_color.unwrap_or(Color::green());
    imageproc::map::map_colors2(origin, mask.mask(), |src, mask| {
        let [r, g, b, _] = src.0;
        let mask_alpha = mask.0[0];
        if mask_alpha == 0 {
            Rgba(bg.into())
        } else {
            Rgba([r, g, b, mask_alpha])
        }
    })
    .into()
}

fn best_grid(n: usize) -> (usize, usize) {
    let mut best_rows = 1;
    let mut best_cols = n;
    let mut min_diff = n as i32;
    for rows in 1..=n {
        let cols = ((n as f32) / (rows as f32)).ceil() as usize;
        if rows * cols >= n {
            let diff = (rows as i32 - cols as i32).abs();
            if diff < min_diff {
                min_diff = diff;
                best_rows = rows;
                best_cols = cols;
            }
        }
    }
    (best_rows, best_cols)
}

fn draw_masks(
    masks: &[&Mask],
    colormap256: Option<&ColorMap256>,
    canvas: &mut RgbaImage,
    mask_cutout: bool,
    mask_background_color: Option<Color>,
) -> Result<()> {
    let (w, h) = canvas.dimensions();
    let n = masks.len() + 1; // +1 for original
    let (rows, cols) = best_grid(n);
    let out_w = w * cols as u32;
    let out_h = h * rows as u32;
    let mut out = RgbaImage::new(out_w, out_h);
    image::imageops::overlay(&mut out, canvas, 0, 0);

    for (i, mask) in masks.iter().enumerate() {
        let idx = i + 1;
        let row = idx / cols;
        let col = idx % cols;
        let mut mask_img = RgbaImage::new(w, h);

        let mask_buf = mask.mask();
        let (mw, mh) = mask_buf.dimensions();
        let x = ((w as i32 - mw as i32) / 2).max(0) as u32;
        let y = ((h as i32 - mh as i32) / 2).max(0) as u32;

        let mask_dyn = if mask_cutout {
            apply_mask(canvas, mask, mask_background_color)
        } else {
            render_mask(mask, colormap256)
        };
        image::imageops::overlay(&mut mask_img, &mask_dyn, x as i64, y as i64);

        let out_x = (col as u32 * w) as i64;
        let out_y = (row as u32 * h) as i64;
        image::imageops::overlay(&mut out, &mask_img, out_x, out_y);
    }

    *canvas = out;
    Ok(())
}

impl Drawable for [Mask] {
    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        ctx.mask_style
    }

    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let mut masks_visible = Vec::with_capacity(self.len());
        for mask in self.iter() {
            let style = ctx.update_style(
                mask.get_local_style(),
                mask.get_global_style(ctx),
                mask.get_id(),
            );

            if style.draw_mask_polygon_largest() {
                if let Some(polygon) = mask.polygon() {
                    polygon.draw(ctx, canvas)?;
                }
            }

            if style.draw_mask_polygons() {
                for polygon in mask.polygons() {
                    polygon.draw(ctx, canvas)?;
                }
            }

            if style.draw_mask_hbbs() {
                if let Some(polygon) = mask.polygon() {
                    if let Some(hbb) = polygon.hbb() {
                        hbb.draw(ctx, canvas)?;
                    }
                }
            }

            if style.draw_mask_obbs() {
                if let Some(polygon) = mask.polygon() {
                    if let Some(obb) = polygon.obb() {
                        obb.draw(ctx, canvas)?;
                    }
                }
            }

            if style.visible() {
                masks_visible.push(mask);
            }
        }

        let style = ctx.update_style(
            self.get_local_style(),
            self.get_global_style(ctx),
            self.get_id(),
        );

        draw_masks(
            &masks_visible,
            style.colormap256(),
            canvas,
            style.mask_cutout(),
            style.mask_background_color().copied(),
        )
    }
}

impl Drawable for Mask {
    fn get_local_style(&self) -> Option<&Style> {
        self.style()
    }

    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        ctx.mask_style
    }

    fn get_id(&self) -> Option<usize> {
        self.id()
    }

    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let style = ctx.update_style(
            self.get_local_style(),
            self.get_global_style(ctx),
            self.get_id(),
        );

        if style.draw_mask_polygon_largest() {
            if let Some(polygon) = self.polygon() {
                polygon.draw(ctx, canvas)?;
            }
        }
        if style.draw_mask_polygons() {
            for polygon in self.polygons() {
                polygon.draw(ctx, canvas)?;
            }
        }

        if style.draw_mask_hbbs() {
            if let Some(polygon) = self.polygon() {
                if let Some(hbb) = polygon.hbb() {
                    hbb.draw(ctx, canvas)?;
                }
            }
        }

        if style.draw_mask_obbs() {
            if let Some(polygon) = self.polygon() {
                if let Some(obb) = polygon.obb() {
                    obb.draw(ctx, canvas)?;
                }
            }
        }

        if style.visible() {
            let (w, h) = canvas.dimensions();
            let mask_dyn = if style.mask_cutout() {
                apply_mask(canvas, self, style.mask_background_color().copied())
            } else {
                render_mask(self, style.colormap256())
            };

            let (mut out, mask_x, mask_y) = if w <= h {
                (RgbaImage::new(w * 2, h), w as i64, 0)
            } else {
                (RgbaImage::new(w, h * 2), 0, h as i64)
            };
            image::imageops::overlay(&mut out, canvas, 0, 0);
            image::imageops::overlay(&mut out, &mask_dyn, mask_x, mask_y);
            *canvas = out;
        }

        Ok(())
    }
}
