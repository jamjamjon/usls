use anyhow::Result;
use image::{Rgba, RgbaImage};

use crate::{
    draw_line_dashed, draw_line_solid_thick, DrawContext, Drawable, Obb, ObbStyle, ObbStyleMode,
};

impl Drawable for [Obb] {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        self.iter().try_for_each(|x| x.draw(ctx, canvas))
    }
}

impl Drawable for Obb {
    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let default_style = ObbStyle::default();
        let style = self.style().or(ctx.obb_style).unwrap_or(&default_style);

        if !style.visible() {
            return Ok(());
        }

        let colors = ctx.resolve_obb_colors(style, self.id());

        // Draw outline
        if style.draw_outline() {
            let color = Rgba(colors.outline.into());
            let mode = *style.mode();
            let vertices = self.vertices();
            let thickness = style.thickness();

            match mode {
                ObbStyleMode::Solid => {
                    for i in 0..vertices.len() {
                        let p1 = vertices[i];
                        let p2 = vertices[(i + 1) % vertices.len()];
                        draw_line_solid_thick(
                            canvas,
                            (p1[0], p1[1]),
                            (p2[0], p2[1]),
                            color,
                            thickness,
                        );
                    }
                }
                ObbStyleMode::Dashed { length, gap } => {
                    for i in 0..vertices.len() {
                        let p1 = vertices[i];
                        let p2 = vertices[(i + 1) % vertices.len()];
                        draw_line_dashed(
                            canvas,
                            (p1[0], p1[1]),
                            (p2[0], p2[1]),
                            color,
                            length,
                            gap,
                            thickness,
                        );
                    }
                }
                ObbStyleMode::Corners {
                    ratio_long,
                    ratio_short,
                } => {
                    // Calculate edge lengths
                    let mut edge_lengths = Vec::with_capacity(4);
                    for i in 0..vertices.len() {
                        let p1 = vertices[i];
                        let p2 = vertices[(i + 1) % vertices.len()];
                        let dx = p2[0] - p1[0];
                        let dy = p2[1] - p1[1];
                        edge_lengths.push((dx * dx + dy * dy).sqrt());
                    }

                    // Determine which edges are long/short
                    let avg_len =
                        (edge_lengths[0] + edge_lengths[1] + edge_lengths[2] + edge_lengths[3])
                            / 4.0;

                    // Draw corner brackets for each vertex
                    for i in 0..vertices.len() {
                        let p_curr = vertices[i];
                        let p_next = vertices[(i + 1) % vertices.len()];
                        let p_prev = vertices[(i + vertices.len() - 1) % vertices.len()];

                        // Edge to next vertex
                        let edge_len_next = edge_lengths[i];
                        let ratio_next = if edge_len_next > avg_len {
                            ratio_long
                        } else {
                            ratio_short
                        };
                        let corner_len_next = (edge_len_next * ratio_next).max(1.0);
                        let dx_next = p_next[0] - p_curr[0];
                        let dy_next = p_next[1] - p_curr[1];
                        let t_next = corner_len_next / edge_len_next;
                        let end_next = [p_curr[0] + dx_next * t_next, p_curr[1] + dy_next * t_next];
                        draw_line_solid_thick(
                            canvas,
                            (p_curr[0], p_curr[1]),
                            (end_next[0], end_next[1]),
                            color,
                            thickness,
                        );

                        // Edge to previous vertex
                        let edge_len_prev = edge_lengths[(i + vertices.len() - 1) % vertices.len()];
                        let ratio_prev = if edge_len_prev > avg_len {
                            ratio_long
                        } else {
                            ratio_short
                        };
                        let corner_len_prev = (edge_len_prev * ratio_prev).max(1.0);
                        let dx_prev = p_prev[0] - p_curr[0];
                        let dy_prev = p_prev[1] - p_curr[1];
                        let t_prev = corner_len_prev / edge_len_prev;
                        let end_prev = [p_curr[0] + dx_prev * t_prev, p_curr[1] + dy_prev * t_prev];
                        draw_line_solid_thick(
                            canvas,
                            (p_curr[0], p_curr[1]),
                            (end_prev[0], end_prev[1]),
                            color,
                            thickness,
                        );
                    }
                }
                ObbStyleMode::Rounded { ratio } => {
                    // Calculate edge lengths and determine radius
                    let mut edge_lengths = Vec::with_capacity(4);
                    for i in 0..vertices.len() {
                        let p1 = vertices[i];
                        let p2 = vertices[(i + 1) % vertices.len()];
                        let dx = p2[0] - p1[0];
                        let dy = p2[1] - p1[1];
                        edge_lengths.push((dx * dx + dy * dy).sqrt());
                    }
                    let min_edge = edge_lengths.iter().cloned().fold(f32::MAX, f32::min);
                    let radius = (min_edge * ratio).max(1.0);

                    // Calculate unit vectors for each edge (from vertex i to vertex i+1)
                    let mut unit_vecs: Vec<[f32; 2]> = Vec::with_capacity(4);
                    for i in 0..vertices.len() {
                        let p1 = vertices[i];
                        let p2 = vertices[(i + 1) % vertices.len()];
                        let len = edge_lengths[i];
                        unit_vecs.push([(p2[0] - p1[0]) / len, (p2[1] - p1[1]) / len]);
                    }

                    // Draw shortened edges (leave room for arcs at corners)
                    for i in 0..vertices.len() {
                        let p1 = vertices[i];
                        let p2 = vertices[(i + 1) % vertices.len()];
                        let uv = unit_vecs[i];

                        let start = [p1[0] + uv[0] * radius, p1[1] + uv[1] * radius];
                        let end = [p2[0] - uv[0] * radius, p2[1] - uv[1] * radius];

                        draw_line_solid_thick(
                            canvas,
                            (start[0], start[1]),
                            (end[0], end[1]),
                            color,
                            thickness,
                        );
                    }

                    // Draw corner arcs (OBB corners are ~90 degrees)
                    for i in 0..vertices.len() {
                        let p_curr = vertices[i];
                        let prev_edge_idx = (i + vertices.len() - 1) % vertices.len();

                        // Direction vectors
                        let uv_in = unit_vecs[prev_edge_idx]; // incoming edge direction
                        let uv_out = unit_vecs[i]; // outgoing edge direction

                        // Arc endpoints on the original edge lines
                        let arc_start =
                            [p_curr[0] - uv_in[0] * radius, p_curr[1] - uv_in[1] * radius];
                        let arc_end = [
                            p_curr[0] + uv_out[0] * radius,
                            p_curr[1] + uv_out[1] * radius,
                        ];

                        // For 90-degree corners, the center is at arc_start + perpendicular * radius
                        // Check winding direction to determine which perpendicular to use
                        let cross = uv_in[0] * uv_out[1] - uv_in[1] * uv_out[0];
                        let perp_in = if cross > 0.0 {
                            [-uv_in[1], uv_in[0]]
                        } else {
                            [uv_in[1], -uv_in[0]]
                        };

                        let center = [
                            arc_start[0] + perp_in[0] * radius,
                            arc_start[1] + perp_in[1] * radius,
                        ];

                        // Calculate angles from center
                        let start_angle =
                            (arc_start[1] - center[1]).atan2(arc_start[0] - center[0]);
                        let end_angle = (arc_end[1] - center[1]).atan2(arc_end[0] - center[0]);

                        // Normalize angle difference
                        let mut angle_diff = end_angle - start_angle;
                        if cross > 0.0 {
                            // Counter-clockwise
                            if angle_diff < 0.0 {
                                angle_diff += 2.0 * std::f32::consts::PI;
                            }
                        } else {
                            // Clockwise
                            if angle_diff > 0.0 {
                                angle_diff -= 2.0 * std::f32::consts::PI;
                            }
                        }

                        let steps = (radius.abs() as i32).max(12);
                        for step in 0..steps {
                            let t1 = step as f32 / steps as f32;
                            let t2 = (step + 1) as f32 / steps as f32;
                            let angle1 = start_angle + angle_diff * t1;
                            let angle2 = start_angle + angle_diff * t2;
                            let x1 = center[0] + radius * angle1.cos();
                            let y1 = center[1] + radius * angle1.sin();
                            let x2 = center[0] + radius * angle2.cos();
                            let y2 = center[1] + radius * angle2.sin();
                            draw_line_solid_thick(canvas, (x1, y1), (x2, y2), color, thickness);
                        }
                    }
                }
            }
        }

        // Draw fill
        if style.draw_fill() {
            let mut overlay = canvas.clone();
            let polygon = self.to_polygon();
            let exterior = polygon.exterior();
            let n = if polygon.is_closed() {
                polygon.count() - 1
            } else {
                polygon.count()
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

            let top = self.top();
            let bottom = self.bottom();
            let left = self.left();
            let right = self.right();
            let bbox = (left[0], top[1], right[0], bottom[1]);

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
                let obb_t = style.thickness();
                let obb_offset = if obb_t > 0 { obb_t - 1 } else { 0 };
                Some((obb_offset + text_thickness) as f32)
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
