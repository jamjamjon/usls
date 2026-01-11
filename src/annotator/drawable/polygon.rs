use anyhow::Result;
use image::{Rgba, RgbaImage};

use crate::{apply_overlay, draw_line_solid_thick, DrawContext, Drawable, Polygon, PolygonStyle};

impl Drawable for [Polygon] {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }

        // Apply background overlay once before drawing any polygons
        let default_style = PolygonStyle::default();
        let style = ctx.polygon_style.unwrap_or(&default_style);
        if let Some(overlay_color) = style.background_overlay() {
            apply_overlay(canvas, *overlay_color);
        }

        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}

impl Drawable for Polygon {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let default_style = PolygonStyle::default();
        let style = self.style().or(ctx.polygon_style).unwrap_or(&default_style);

        if !style.visible() {
            return Ok(());
        }

        let colors = ctx.resolve_polygon_colors(style, self.id());

        // Draw fill
        if style.draw_fill() {
            let mut overlay = canvas.clone();
            let exterior = self.exterior();
            let n = if self.is_closed() {
                self.count() - 1
            } else {
                self.count()
            };
            let polygon_i32 = exterior
                .iter()
                .take(n)
                .map(|p| imageproc::point::Point::new(p[0] as i32, p[1] as i32))
                .collect::<Vec<_>>();

            imageproc::drawing::draw_polygon_mut(
                &mut overlay,
                &polygon_i32,
                Rgba(colors.fill.into()),
            );
            image::imageops::overlay(canvas, &overlay, 0, 0);
        }

        // Draw outline with thickness support
        if style.draw_outline() {
            let exterior = self.exterior();
            let n = if self.is_closed() {
                self.count() - 1
            } else {
                self.count()
            };

            // Calculate effective thickness with max ratio limit
            let base_thickness = style.thickness();
            let thickness = if let Some(hbb) = self.hbb() {
                let min_dim = hbb.width().min(hbb.height());
                let max_thickness = (min_dim * style.thickness_max_ratio()) as usize;
                base_thickness.min(max_thickness.max(1))
            } else {
                base_thickness
            };
            let color = Rgba(colors.outline.into());

            if thickness <= 1 {
                // Use simple hollow polygon for thickness 1
                let polygon_f32 = exterior
                    .iter()
                    .take(n)
                    .map(|p| imageproc::point::Point::new(p[0], p[1]))
                    .collect::<Vec<_>>();
                imageproc::drawing::draw_hollow_polygon_mut(canvas, &polygon_f32, color);
            } else {
                // Draw thick outline using filled polygons for smooth lines
                let points: Vec<_> = exterior.iter().take(n).collect();

                // Draw circles at each vertex to fill corner gaps
                let radius = thickness as i32 / 2;
                for p in &points {
                    imageproc::drawing::draw_filled_circle_mut(
                        canvas,
                        (p[0] as i32, p[1] as i32),
                        radius,
                        color,
                    );
                }

                // Draw thick line segments
                for i in 0..points.len() {
                    let p1 = points[i];
                    let p2 = points[(i + 1) % points.len()];
                    draw_line_solid_thick(canvas, (p1[0], p1[1]), (p2[0], p2[1]), color, thickness);
                }
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

            let hbb = self.hbb();
            let (cx, cy) = self.centroid().unwrap_or((0.0, 0.0));
            let bbox = hbb
                .as_ref()
                .map(|b| (b.xmin(), b.ymin(), b.xmax(), b.ymax()))
                .unwrap_or((cx, cy, cx, cy));

            let font_size = text_style.font_size();
            let box_size = ctx
                .text_renderer
                .box_size_with(&label, &text_mode, font_size)?;
            let canvas_size = (canvas.width(), canvas.height());

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
