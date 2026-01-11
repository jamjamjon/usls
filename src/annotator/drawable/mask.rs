use anyhow::Result;
use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};

use crate::{
    apply_overlay, Color, ColorMap256, DrawContext, Drawable, GlowRadius, Mask, MaskStyle,
    MaskStyleMode, PolygonStyle,
};

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

fn apply_mask(origin: &RgbaImage, mask: &Mask, background_color: Color) -> DynamicImage {
    let bg = background_color;
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

/// Apply halo effect: grayscale background + colored glow around mask edges + original colors inside mask
fn apply_halo(
    canvas: &RgbaImage,
    masks: &[&Mask],
    glow_radius: GlowRadius,
    glow_color: Color,
) -> RgbaImage {
    let (w, h) = canvas.dimensions();

    // Create combined mask and calculate max glow radius
    let mut combined_mask = GrayImage::new(w, h);
    let mut max_glow_radius = 0f32;

    for mask in masks {
        let mask_buf = mask.mask();
        let (mw, mh) = mask_buf.dimensions();
        let offset_x = ((w as i32 - mw as i32) / 2).max(0) as u32;
        let offset_y = ((h as i32 - mh as i32) / 2).max(0) as u32;

        // Calculate glow radius for this mask
        let mask_radius = match glow_radius {
            GlowRadius::Pixels(px) => px as f32,
            GlowRadius::Percent(pct) => {
                let diagonal = ((mw * mw + mh * mh) as f32).sqrt();
                (diagonal * pct).max(3.0)
            }
        };
        max_glow_radius = max_glow_radius.max(mask_radius);

        for y in 0..mh.min(h - offset_y) {
            for x in 0..mw.min(w - offset_x) {
                if mask_buf.get_pixel(x, y).0[0] > 0 {
                    combined_mask.put_pixel(x + offset_x, y + offset_y, Luma([255]));
                }
            }
        }
    }

    // Fast distance transform
    let distance = compute_distance_transform(&combined_mask);

    // Compose final image
    let mut result = RgbaImage::new(w, h);
    let (gr, gg, gb, ga) = glow_color.rgba();

    for y in 0..h {
        for x in 0..w {
            let src = canvas.get_pixel(x, y);
            let [r, g, b, a] = src.0;

            if combined_mask.get_pixel(x, y).0[0] > 0 {
                // Inside mask: keep original color
                result.put_pixel(x, y, *src);
            } else {
                // Outside mask: grayscale with glow
                let gray = ((r as u32 + g as u32 + b as u32) / 3) as u8;
                let dist = distance[(y * w + x) as usize];

                if dist < max_glow_radius {
                    let glow_intensity = 1.0 - dist / max_glow_radius;
                    let glow_alpha = glow_intensity * (ga as f32 / 255.0);
                    let nr = (gr as f32 * glow_alpha + gray as f32 * (1.0 - glow_alpha)) as u8;
                    let ng = (gg as f32 * glow_alpha + gray as f32 * (1.0 - glow_alpha)) as u8;
                    let nb = (gb as f32 * glow_alpha + gray as f32 * (1.0 - glow_alpha)) as u8;
                    result.put_pixel(x, y, Rgba([nr, ng, nb, a]));
                } else {
                    result.put_pixel(x, y, Rgba([gray, gray, gray, a]));
                }
            }
        }
    }

    result
}

/// Fast distance transform using two-pass algorithm (O(n) per pixel)
fn compute_distance_transform(mask: &GrayImage) -> Vec<f32> {
    let (w, h) = mask.dimensions();
    let size = (w * h) as usize;
    let mut dist = vec![f32::MAX; size];

    // Initialize: 0 for mask pixels, MAX for others
    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y).0[0] > 0 {
                dist[(y * w + x) as usize] = 0.0;
            }
        }
    }

    // Forward pass (top-left to bottom-right)
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if x > 0 {
                dist[idx] = dist[idx].min(dist[idx - 1] + 1.0);
            }
            if y > 0 {
                dist[idx] = dist[idx].min(dist[((y - 1) * w + x) as usize] + 1.0);
            }
            // Diagonal
            if x > 0 && y > 0 {
                dist[idx] = dist[idx].min(dist[((y - 1) * w + x - 1) as usize] + 1.414);
            }
            if x < w - 1 && y > 0 {
                dist[idx] = dist[idx].min(dist[((y - 1) * w + x + 1) as usize] + 1.414);
            }
        }
    }

    // Backward pass (bottom-right to top-left)
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let idx = (y * w + x) as usize;
            if x < w - 1 {
                dist[idx] = dist[idx].min(dist[idx + 1] + 1.0);
            }
            if y < h - 1 {
                dist[idx] = dist[idx].min(dist[((y + 1) * w + x) as usize] + 1.0);
            }
            // Diagonal
            if x < w - 1 && y < h - 1 {
                dist[idx] = dist[idx].min(dist[((y + 1) * w + x + 1) as usize] + 1.414);
            }
            if x > 0 && y < h - 1 {
                dist[idx] = dist[idx].min(dist[((y + 1) * w + x - 1) as usize] + 1.414);
            }
        }
    }

    dist
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
    cutout: bool,
    cutout_source: Option<&RgbaImage>, // Some(original) if cutout_original, None uses canvas
    cutout_background_color: Color,
    mode: &MaskStyleMode,
) -> Result<()> {
    // Early return if no masks
    if masks.is_empty() {
        return Ok(());
    }

    // Handle Halo mode: apply halo effect directly to canvas
    if let MaskStyleMode::Halo {
        glow_radius,
        glow_color,
    } = mode
    {
        *canvas = apply_halo(canvas, masks, *glow_radius, *glow_color);
        return Ok(());
    }

    // Default Overlay mode: create grid of mask visualizations
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

        let mask_dyn = if cutout {
            // Use original canvas if cutout_original, otherwise use canvas with overlays
            let source = cutout_source.unwrap_or(canvas);
            apply_mask(source, mask, cutout_background_color)
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
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let default_style = MaskStyle::default();
        let style = ctx.mask_style.unwrap_or(&default_style);

        // Save original canvas for cutout (before any polygon overlays) if cutout_original is enabled
        let original_canvas = if style.cutout() && style.cutout_original() && style.visible() {
            Some(canvas.clone())
        } else {
            None
        };

        // Apply background overlay once before drawing any polygons from masks
        let will_draw_polygons = style.draw_polygon_largest() || style.draw_polygons();
        if will_draw_polygons && !self.is_empty() {
            let default_polygon_style = PolygonStyle::default();
            let polygon_style = ctx.polygon_style.unwrap_or(&default_polygon_style);
            if let Some(overlay_color) = polygon_style.background_overlay() {
                apply_overlay(canvas, *overlay_color);
            }
        }

        let mut masks_visible = Vec::with_capacity(self.len());
        for mask in self.iter() {
            if style.draw_polygon_largest() {
                if let Some(polygon) = mask.polygon() {
                    polygon.draw(ctx, canvas)?;
                }
            }

            if style.draw_polygons() {
                for polygon in mask.polygons() {
                    polygon.draw(ctx, canvas)?;
                }
            }

            if style.draw_hbbs() {
                if let Some(polygon) = mask.polygon() {
                    if let Some(hbb) = polygon.hbb() {
                        hbb.draw(ctx, canvas)?;
                    }
                }
            }

            if style.draw_obbs() {
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

        draw_masks(
            &masks_visible,
            style.colormap256(),
            canvas,
            style.cutout(),
            original_canvas.as_ref(),
            *style.cutout_background_color(),
            style.mode(),
        )
    }
}

impl Drawable for Mask {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let default_style = MaskStyle::default();
        let style = self.style().or(ctx.mask_style).unwrap_or(&default_style);

        // Save original canvas for cutout (before any polygon overlays) if cutout_original is enabled
        let original_canvas = if style.cutout() && style.cutout_original() && style.visible() {
            Some(canvas.clone())
        } else {
            None
        };

        // Apply background overlay once before drawing polygons
        let will_draw_polygons = style.draw_polygon_largest() || style.draw_polygons();
        if will_draw_polygons {
            let default_polygon_style = PolygonStyle::default();
            let polygon_style = ctx.polygon_style.unwrap_or(&default_polygon_style);
            if let Some(overlay_color) = polygon_style.background_overlay() {
                apply_overlay(canvas, *overlay_color);
            }
        }

        if style.draw_polygon_largest() {
            if let Some(polygon) = self.polygon() {
                polygon.draw(ctx, canvas)?;
            }
        }
        if style.draw_polygons() {
            for polygon in self.polygons() {
                polygon.draw(ctx, canvas)?;
            }
        }

        if style.draw_hbbs() {
            if let Some(polygon) = self.polygon() {
                if let Some(hbb) = polygon.hbb() {
                    hbb.draw(ctx, canvas)?;
                }
            }
        }

        if style.draw_obbs() {
            if let Some(polygon) = self.polygon() {
                if let Some(obb) = polygon.obb() {
                    obb.draw(ctx, canvas)?;
                }
            }
        }

        if style.visible() {
            // Handle Halo mode for single mask
            if let MaskStyleMode::Halo {
                glow_radius,
                glow_color,
            } = style.mode()
            {
                *canvas = apply_halo(canvas, &[self], *glow_radius, *glow_color);
                return Ok(());
            }

            // Default mode: side-by-side view
            let (w, h) = canvas.dimensions();
            let mask_dyn = if style.cutout() {
                // Use original canvas if cutout_original, otherwise use canvas with overlays
                let source = original_canvas.as_ref().unwrap_or(canvas);
                apply_mask(source, self, *style.cutout_background_color())
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
