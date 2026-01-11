use image::{Rgba, RgbaImage};
use std::f32::consts::PI;

use crate::{Color, HbbStyleMode, KeypointStyleMode, ThicknessDirection};

/// Apply a semi-transparent color overlay to the entire image.
/// Useful for making annotations stand out against the background.
pub fn apply_overlay(canvas: &mut RgbaImage, color: Color) {
    let [cr, cg, cb, ca] = <[u8; 4]>::from(color);
    if ca == 0 {
        return;
    }
    let alpha = ca as f32 / 255.0;
    let inv_alpha = 1.0 - alpha;

    for pixel in canvas.pixels_mut() {
        let [sr, sg, sb, sa] = pixel.0;
        pixel.0 = [
            (cr as f32 * alpha + sr as f32 * inv_alpha) as u8,
            (cg as f32 * alpha + sg as f32 * inv_alpha) as u8,
            (cb as f32 * alpha + sb as f32 * inv_alpha) as u8,
            (ca as f32 + sa as f32 * inv_alpha).min(255.0) as u8,
        ];
    }
}

/// Draw a solid line segment.
#[inline]
pub fn draw_line_solid(
    canvas: &mut RgbaImage,
    start: (f32, f32),
    end: (f32, f32),
    color: Rgba<u8>,
) {
    imageproc::drawing::draw_line_segment_mut(canvas, start, end, color);
}

/// Draw a solid line segment with thickness.
/// Uses filled polygon for smooth thick lines.
pub fn draw_line_solid_thick(
    canvas: &mut RgbaImage,
    start: (f32, f32),
    end: (f32, f32),
    color: Rgba<u8>,
    thickness: usize,
) {
    if thickness <= 1 {
        draw_line_solid(canvas, start, end, color);
        return;
    }

    let (x1, y1) = start;
    let (x2, y2) = end;
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();

    // For very short lines (< 1 pixel), draw a filled circle instead
    if len < 1.0 {
        let radius = (thickness as i32 / 2).max(1);
        imageproc::drawing::draw_filled_circle_mut(canvas, (x1 as i32, y1 as i32), radius, color);
        return;
    }

    // Perpendicular unit vector
    let nx = -dy / len;
    let ny = dx / len;

    // Half thickness
    let half_t = thickness as f32 / 2.0;

    // Create a rectangle (4 corners) for the thick line
    let corners = [
        imageproc::point::Point::new((x1 + nx * half_t) as i32, (y1 + ny * half_t) as i32),
        imageproc::point::Point::new((x1 - nx * half_t) as i32, (y1 - ny * half_t) as i32),
        imageproc::point::Point::new((x2 - nx * half_t) as i32, (y2 - ny * half_t) as i32),
        imageproc::point::Point::new((x2 + nx * half_t) as i32, (y2 + ny * half_t) as i32),
    ];

    // Check for degenerate polygon (first == last point)
    if corners[0] == corners[3] || corners[1] == corners[2] {
        // Line too short for polygon, draw circle at midpoint
        let mx = ((x1 + x2) / 2.0) as i32;
        let my = ((y1 + y2) / 2.0) as i32;
        let radius = (thickness as i32 / 2).max(1);
        imageproc::drawing::draw_filled_circle_mut(canvas, (mx, my), radius, color);
        return;
    }

    imageproc::drawing::draw_polygon_mut(canvas, &corners, color);
}

/// Draw a dashed line segment with thickness support.
pub fn draw_line_dashed(
    canvas: &mut RgbaImage,
    start: (f32, f32),
    end: (f32, f32),
    color: Rgba<u8>,
    dash_length: usize,
    gap_length: usize,
    thickness: usize,
) {
    let (x1, y1) = start;
    let (x2, y2) = end;
    let dx = x2 - x1;
    let dy = y2 - y1;
    let length = (dx * dx + dy * dy).sqrt();

    if length == 0.0 {
        return;
    }

    let unit_x = dx / length;
    let unit_y = dy / length;

    let mut pos = 0.0;
    let mut drawing = true;
    let dash_len = dash_length as f32;
    let gap_len = gap_length as f32;

    while pos < length {
        let segment_len = if drawing { dash_len } else { gap_len };
        let end_pos = (pos + segment_len).min(length);

        if drawing {
            let seg_start = (x1 + pos * unit_x, y1 + pos * unit_y);
            let seg_end = (x1 + end_pos * unit_x, y1 + end_pos * unit_y);
            draw_line_solid_thick(canvas, seg_start, seg_end, color, thickness);
        }

        pos = end_pos;
        drawing = !drawing;
    }
}

/// Draw a rectangle based on HbbStyleMode with specified thickness and direction.
#[allow(clippy::too_many_arguments)]
pub fn draw_hbb(
    canvas: &mut RgbaImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Rgba<u8>,
    mode: HbbStyleMode,
    thickness: usize,
    direction: ThicknessDirection,
) {
    // Helper to compute offset for each layer based on direction
    let compute_offset = |i: usize| -> (f32, f32) {
        let t = thickness as f32;
        match direction {
            ThicknessDirection::Outward => {
                // Expand outward: layer 0 at original, layer n-1 at max outward
                let out = i as f32;
                (-out, out)
            }
            ThicknessDirection::Inward => {
                // Expand inward: layer 0 at original, layer n-1 at max inward
                let in_ = i as f32;
                (in_, -in_)
            }
            ThicknessDirection::Centered => {
                // Centered: half outward, half inward
                let half = t / 2.0;
                let out = (i as f32 - half).max(-half);
                (-out, out)
            }
        }
    };

    match mode {
        HbbStyleMode::Solid => {
            for i in 0..thickness {
                let (off_min, off_max) = compute_offset(i);
                draw_rect_solid(
                    canvas,
                    x1 + off_min,
                    y1 + off_min,
                    x2 + off_max,
                    y2 + off_max,
                    color,
                );
            }
        }
        HbbStyleMode::Dashed { length, gap } => {
            for i in 0..thickness {
                let (off_min, off_max) = compute_offset(i);
                draw_rect_dashed(
                    canvas,
                    x1 + off_min,
                    y1 + off_min,
                    x2 + off_max,
                    y2 + off_max,
                    color,
                    length,
                    gap,
                );
            }
        }
        HbbStyleMode::Corners {
            ratio_long,
            ratio_short,
        } => {
            let width = x2 - x1;
            let height = y2 - y1;
            let (corner_len_h, corner_len_v) = if width >= height {
                let len_h = (width * ratio_long.min(0.5)).max(1.0) as usize;
                let len_v = (height * ratio_short.min(0.5)).max(1.0) as usize;
                (len_h, len_v)
            } else {
                let len_h = (width * ratio_short.min(0.5)).max(1.0) as usize;
                let len_v = (height * ratio_long.min(0.5)).max(1.0) as usize;
                (len_h, len_v)
            };
            // Corners use filled rectangles, direction affects position
            let (off_min, off_max) = match direction {
                ThicknessDirection::Outward => (0.0, 0.0),
                ThicknessDirection::Inward => (0.0, 0.0),
                ThicknessDirection::Centered => (0.0, 0.0),
            };
            draw_rect_corners(
                canvas,
                x1 + off_min,
                y1 + off_min,
                x2 + off_max,
                y2 + off_max,
                color,
                corner_len_h,
                corner_len_v,
                thickness,
                direction,
            );
        }
        HbbStyleMode::Rounded { ratio } => {
            let width = x2 - x1;
            let height = y2 - y1;
            let short_side = width.min(height);
            let radius = (short_side * ratio.min(0.5)).max(1.0) as usize;
            for i in 0..thickness {
                let (off_min, off_max) = compute_offset(i);
                draw_rect_rounded(
                    canvas,
                    x1 + off_min,
                    y1 + off_min,
                    x2 + off_max,
                    y2 + off_max,
                    color,
                    radius,
                );
            }
        }
    }
}

/// Draw a solid rectangle outline.
fn draw_rect_solid(canvas: &mut RgbaImage, x1: f32, y1: f32, x2: f32, y2: f32, color: Rgba<u8>) {
    draw_line_solid(canvas, (x1, y1), (x2, y1), color);
    draw_line_solid(canvas, (x1, y2), (x2, y2), color);
    draw_line_solid(canvas, (x1, y1), (x1, y2), color);
    draw_line_solid(canvas, (x2, y1), (x2, y2), color);
}

/// Draw a dashed rectangle outline (single pixel thickness for offset-based drawing).
#[allow(clippy::too_many_arguments)]
fn draw_rect_dashed(
    canvas: &mut RgbaImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Rgba<u8>,
    dash_length: usize,
    gap_length: usize,
) {
    // Use thickness=1 since HBB handles thickness by drawing multiple offset rectangles
    draw_line_dashed(
        canvas,
        (x1, y1),
        (x2, y1),
        color,
        dash_length,
        gap_length,
        1,
    );
    draw_line_dashed(
        canvas,
        (x1, y2),
        (x2, y2),
        color,
        dash_length,
        gap_length,
        1,
    );
    draw_line_dashed(
        canvas,
        (x1, y1),
        (x1, y2),
        color,
        dash_length,
        gap_length,
        1,
    );
    draw_line_dashed(
        canvas,
        (x2, y1),
        (x2, y2),
        color,
        dash_length,
        gap_length,
        1,
    );
}

/// Draw corner brackets at four corners of a rectangle with thickness.
///
/// ```text
///  ┌‾       ‾┐
///
///
///  └_       _┘
/// ```
#[allow(clippy::too_many_arguments)]
pub fn draw_rect_corners(
    canvas: &mut RgbaImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Rgba<u8>,
    corner_len_h: usize,
    corner_len_v: usize,
    thickness: usize,
    direction: ThicknessDirection,
) {
    let width = x2 - x1;
    let height = y2 - y1;

    // Adjust corner lengths if rectangle is too small
    let len_h = (corner_len_h as f32).min(width / 2.5);
    let len_v = (corner_len_v as f32).min(height / 2.5);

    let t = thickness as f32;

    // Compute offsets based on direction
    let (t_out, t_in) = match direction {
        ThicknessDirection::Outward => (t, 0.0),
        ThicknessDirection::Inward => (0.0, t),
        ThicknessDirection::Centered => (t / 2.0, t / 2.0),
    };

    // Top-left corner ┌
    draw_filled_rect(canvas, x1 - t_out, y1 - t_out, x1 + len_h, y1 + t_in, color);
    draw_filled_rect(canvas, x1 - t_out, y1 - t_out, x1 + t_in, y1 + len_v, color);

    // Top-right corner ┐
    draw_filled_rect(canvas, x2 - len_h, y1 - t_out, x2 + t_out, y1 + t_in, color);
    draw_filled_rect(canvas, x2 - t_in, y1 - t_out, x2 + t_out, y1 + len_v, color);

    // Bottom-left corner └
    draw_filled_rect(canvas, x1 - t_out, y2 - t_in, x1 + len_h, y2 + t_out, color);
    draw_filled_rect(canvas, x1 - t_out, y2 - len_v, x1 + t_in, y2 + t_out, color);

    // Bottom-right corner ┘
    draw_filled_rect(canvas, x2 - len_h, y2 - t_in, x2 + t_out, y2 + t_out, color);
    draw_filled_rect(canvas, x2 - t_in, y2 - len_v, x2 + t_out, y2 + t_out, color);
}

/// Draw a filled rectangle (helper for corners)
fn draw_filled_rect(canvas: &mut RgbaImage, x1: f32, y1: f32, x2: f32, y2: f32, color: Rgba<u8>) {
    let x_start = x1.min(x2).max(0.0) as u32;
    let x_end = x1.max(x2).min(canvas.width() as f32) as u32;
    let y_start = y1.min(y2).max(0.0) as u32;
    let y_end = y1.max(y2).min(canvas.height() as f32) as u32;

    for y in y_start..y_end {
        for x in x_start..x_end {
            canvas.put_pixel(x, y, color);
        }
    }
}

/// Draw a rounded rectangle outline.
fn draw_rect_rounded(
    canvas: &mut RgbaImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Rgba<u8>,
    radius: usize,
) {
    let r = radius as f32;
    let width = x2 - x1;
    let height = y2 - y1;

    // Clamp radius to half of the smaller dimension
    let r = r.min(width / 2.0).min(height / 2.0);

    // Draw the four straight edges (excluding corners)
    // Top edge
    draw_line_solid(canvas, (x1 + r, y1), (x2 - r, y1), color);
    // Bottom edge
    draw_line_solid(canvas, (x1 + r, y2), (x2 - r, y2), color);
    // Left edge
    draw_line_solid(canvas, (x1, y1 + r), (x1, y2 - r), color);
    // Right edge
    draw_line_solid(canvas, (x2, y1 + r), (x2, y2 - r), color);

    // Draw the four corner arcs with smooth curves
    // More steps = smoother arcs
    let steps = (r * 4.0).max(16.0) as usize;

    // Helper to draw arc from start_angle to end_angle
    let draw_arc = |canvas: &mut RgbaImage, cx: f32, cy: f32, start_angle: f32, end_angle: f32| {
        let delta = end_angle - start_angle;
        for i in 0..steps {
            let angle1 = start_angle + (i as f32 / steps as f32) * delta;
            let angle2 = start_angle + ((i + 1) as f32 / steps as f32) * delta;
            let p1 = (cx + r * angle1.cos(), cy + r * angle1.sin());
            let p2 = (cx + r * angle2.cos(), cy + r * angle2.sin());
            draw_line_solid(canvas, p1, p2, color);
        }
    };

    // Top-left corner (PI to 3*PI/2, i.e., 180° to 270°)
    draw_arc(canvas, x1 + r, y1 + r, PI, PI * 1.5);
    // Top-right corner (3*PI/2 to 2*PI, i.e., 270° to 360°)
    draw_arc(canvas, x2 - r, y1 + r, PI * 1.5, PI * 2.0);
    // Bottom-right corner (0 to PI/2, i.e., 0° to 90°)
    draw_arc(canvas, x2 - r, y2 - r, 0.0, PI * 0.5);
    // Bottom-left corner (PI/2 to PI, i.e., 90° to 180°)
    draw_arc(canvas, x1 + r, y2 - r, PI * 0.5, PI);
}

/// Draw a keypoint shape at the given position.
pub fn draw_keypoint_shape(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: usize,
    color: Rgba<u8>,
    mode: KeypointStyleMode,
    fill: bool,
) {
    match mode {
        KeypointStyleMode::Circle => {
            if fill {
                imageproc::drawing::draw_filled_circle_mut(
                    canvas,
                    (cx as i32, cy as i32),
                    radius as i32,
                    color,
                );
            } else {
                imageproc::drawing::draw_hollow_circle_mut(
                    canvas,
                    (cx as i32, cy as i32),
                    radius as i32,
                    color,
                );
            }
        }
        KeypointStyleMode::Star {
            points,
            inner_ratio,
        } => {
            draw_star(
                canvas,
                cx,
                cy,
                radius as f32,
                inner_ratio,
                points,
                color,
                fill,
            );
        }
        KeypointStyleMode::Square => {
            let r = radius as f32;
            if fill {
                imageproc::drawing::draw_filled_rect_mut(
                    canvas,
                    imageproc::rect::Rect::at((cx - r) as i32, (cy - r) as i32)
                        .of_size((r * 2.0) as u32, (r * 2.0) as u32),
                    color,
                );
            } else {
                draw_rect_solid(canvas, cx - r, cy - r, cx + r, cy + r, color);
            }
        }
        KeypointStyleMode::Cross { thickness } => {
            draw_cross(canvas, cx, cy, radius as f32, thickness, color);
        }
        KeypointStyleMode::Diamond => {
            draw_diamond(canvas, cx, cy, radius as f32, color, fill);
        }
        KeypointStyleMode::Triangle => {
            draw_triangle(canvas, cx, cy, radius as f32, color, fill);
        }
        KeypointStyleMode::X { thickness } => {
            draw_x(canvas, cx, cy, radius as f32, thickness, color);
        }
        KeypointStyleMode::RoundedSquare { corner_ratio } => {
            draw_rounded_square(canvas, cx, cy, radius as f32, corner_ratio, color, fill);
        }
        KeypointStyleMode::Glow { glow_multiplier } => {
            draw_keypoint_glow(canvas, cx, cy, radius as f32, glow_multiplier, color);
        }
    }
}

/// Draw a thick keypoint outline at the given position.
/// Thickness extends outward from the radius boundary.
#[allow(clippy::too_many_arguments)]
pub fn draw_keypoint_outline_thick(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: usize,
    thickness: usize,
    color: Rgba<u8>,
    mode: KeypointStyleMode,
) {
    if thickness == 0 {
        return;
    }

    // For thickness=1, use simple outline
    if thickness == 1 {
        draw_keypoint_shape(canvas, cx, cy, radius, color, mode, false);
        return;
    }

    match mode {
        KeypointStyleMode::Circle => {
            // Draw smooth thick circle outline using distance-based filling
            draw_circle_outline_smooth(canvas, cx, cy, radius as f32, thickness as f32, color);
        }
        KeypointStyleMode::Star {
            points,
            inner_ratio,
        } => {
            // Draw thick star outline
            draw_star_thick(
                canvas,
                cx,
                cy,
                radius as f32,
                inner_ratio,
                points,
                color,
                thickness,
            );
        }
        KeypointStyleMode::Square => {
            // Draw thick square outline (outward from radius)
            let r = radius as f32;
            for i in 0..thickness {
                let offset = i as f32;
                draw_rect_solid(
                    canvas,
                    cx - r - offset,
                    cy - r - offset,
                    cx + r + offset,
                    cy + r + offset,
                    color,
                );
            }
        }
        KeypointStyleMode::Cross { thickness: cross_t } => {
            // Cross already has internal thickness, just use it
            draw_cross(canvas, cx, cy, radius as f32, cross_t, color);
        }
        KeypointStyleMode::Diamond => {
            draw_diamond_thick(canvas, cx, cy, radius as f32, color, thickness);
        }
        KeypointStyleMode::Triangle => {
            draw_triangle_thick(canvas, cx, cy, radius as f32, color, thickness);
        }
        KeypointStyleMode::X { thickness: x_t } => {
            // X already has internal thickness, just use it
            draw_x(canvas, cx, cy, radius as f32, x_t, color);
        }
        KeypointStyleMode::RoundedSquare { corner_ratio } => {
            draw_rounded_square_thick(
                canvas,
                cx,
                cy,
                radius as f32,
                corner_ratio,
                color,
                thickness,
            );
        }
        KeypointStyleMode::Glow { glow_multiplier } => {
            // Glow doesn't have outline concept
            draw_keypoint_glow(canvas, cx, cy, radius as f32, glow_multiplier, color);
        }
    }
}

/// Draw a smooth thick circle outline using distance-based anti-aliased filling.
/// This creates a ring (annulus) from inner_radius to outer_radius with smooth edges.
fn draw_circle_outline_smooth(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    inner_radius: f32,
    thickness: f32,
    color: Rgba<u8>,
) {
    let outer_radius = inner_radius + thickness;
    let (width, height) = canvas.dimensions();

    // Bounding box for the ring
    let x_min = ((cx - outer_radius - 1.0).max(0.0)) as u32;
    let x_max = ((cx + outer_radius + 2.0).min(width as f32)) as u32;
    let y_min = ((cy - outer_radius - 1.0).max(0.0)) as u32;
    let y_max = ((cy + outer_radius + 2.0).min(height as f32)) as u32;

    for py in y_min..y_max {
        for px in x_min..x_max {
            let dx = px as f32 - cx;
            let dy = py as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();

            // Check if pixel is within the ring
            if dist >= inner_radius - 0.5 && dist <= outer_radius + 0.5 {
                // Calculate anti-aliasing alpha based on distance to edges
                let alpha = if dist < inner_radius {
                    // Inner edge anti-aliasing
                    1.0 - (inner_radius - dist)
                } else if dist > outer_radius {
                    // Outer edge anti-aliasing
                    1.0 - (dist - outer_radius)
                } else {
                    1.0
                };

                if alpha > 0.0 {
                    let alpha = alpha.clamp(0.0, 1.0);
                    let existing = canvas.get_pixel(px, py);

                    // Blend with existing pixel
                    let blend_alpha = (color.0[3] as f32 / 255.0) * alpha;
                    let inv_alpha = 1.0 - blend_alpha;

                    let r =
                        (color.0[0] as f32 * blend_alpha + existing.0[0] as f32 * inv_alpha) as u8;
                    let g =
                        (color.0[1] as f32 * blend_alpha + existing.0[1] as f32 * inv_alpha) as u8;
                    let b =
                        (color.0[2] as f32 * blend_alpha + existing.0[2] as f32 * inv_alpha) as u8;
                    let a =
                        ((blend_alpha + existing.0[3] as f32 / 255.0 * inv_alpha) * 255.0) as u8;

                    canvas.put_pixel(px, py, Rgba([r, g, b, a]));
                }
            }
        }
    }
}

/// Draw a star shape.
#[allow(clippy::too_many_arguments)]
fn draw_star(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    outer_radius: f32,
    inner_ratio: f32,
    points: usize,
    color: Rgba<u8>,
    fill: bool,
) {
    let inner_radius = outer_radius * inner_ratio;
    let mut vertices = Vec::with_capacity(points * 2);

    for i in 0..(points * 2) {
        let angle = (i as f32) * PI / (points as f32) - PI / 2.0;
        let r = if i % 2 == 0 {
            outer_radius
        } else {
            inner_radius
        };
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        vertices.push(imageproc::point::Point::new(x as i32, y as i32));
    }

    if fill {
        imageproc::drawing::draw_polygon_mut(canvas, &vertices, color);
    } else {
        // Draw outline
        for i in 0..vertices.len() {
            let p1 = &vertices[i];
            let p2 = &vertices[(i + 1) % vertices.len()];
            draw_line_solid(
                canvas,
                (p1.x as f32, p1.y as f32),
                (p2.x as f32, p2.y as f32),
                color,
            );
        }
    }
}

/// Draw a cross/plus sign.
fn draw_cross(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    thickness: usize,
    color: Rgba<u8>,
) {
    let half_thick = (thickness as f32) / 2.0;

    // Horizontal line
    for t in 0..thickness {
        let offset = t as f32 - half_thick;
        draw_line_solid(
            canvas,
            (cx - radius, cy + offset),
            (cx + radius, cy + offset),
            color,
        );
    }

    // Vertical line
    for t in 0..thickness {
        let offset = t as f32 - half_thick;
        draw_line_solid(
            canvas,
            (cx + offset, cy - radius),
            (cx + offset, cy + radius),
            color,
        );
    }
}

/// Draw a diamond shape.
fn draw_diamond(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    color: Rgba<u8>,
    fill: bool,
) {
    let vertices = [
        imageproc::point::Point::new(cx as i32, (cy - radius) as i32), // top
        imageproc::point::Point::new((cx + radius) as i32, cy as i32), // right
        imageproc::point::Point::new(cx as i32, (cy + radius) as i32), // bottom
        imageproc::point::Point::new((cx - radius) as i32, cy as i32), // left
    ];

    if fill {
        imageproc::drawing::draw_polygon_mut(canvas, &vertices, color);
    } else {
        for i in 0..4 {
            let p1 = &vertices[i];
            let p2 = &vertices[(i + 1) % 4];
            draw_line_solid(
                canvas,
                (p1.x as f32, p1.y as f32),
                (p2.x as f32, p2.y as f32),
                color,
            );
        }
    }
}

/// Draw a triangle (pointing up).
fn draw_triangle(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    color: Rgba<u8>,
    fill: bool,
) {
    // Equilateral triangle pointing up
    let angle_offset = -PI / 2.0; // Start from top
    let vertices: Vec<_> = (0..3)
        .map(|i| {
            let angle = angle_offset + (i as f32) * 2.0 * PI / 3.0;
            imageproc::point::Point::new(
                (cx + radius * angle.cos()) as i32,
                (cy + radius * angle.sin()) as i32,
            )
        })
        .collect();

    if fill {
        imageproc::drawing::draw_polygon_mut(canvas, &vertices, color);
    } else {
        for i in 0..3 {
            let p1 = &vertices[i];
            let p2 = &vertices[(i + 1) % 3];
            draw_line_solid(
                canvas,
                (p1.x as f32, p1.y as f32),
                (p2.x as f32, p2.y as f32),
                color,
            );
        }
    }
}

/// Draw a smooth thick polygon outline with proper corner joins.
/// Uses distance-based anti-aliased filling for smooth edges.
fn draw_polygon_outline_smooth(
    canvas: &mut RgbaImage,
    vertices: &[(f32, f32)],
    thickness: f32,
    color: Rgba<u8>,
) {
    if vertices.is_empty() {
        return;
    }

    let half_t = thickness / 2.0;
    let (width, height) = canvas.dimensions();

    // Calculate bounding box
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for &(x, y) in vertices {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    let x_min = ((min_x - half_t - 1.0).max(0.0)) as u32;
    let x_max = ((max_x + half_t + 2.0).min(width as f32)) as u32;
    let y_min = ((min_y - half_t - 1.0).max(0.0)) as u32;
    let y_max = ((max_y + half_t + 2.0).min(height as f32)) as u32;

    let n = vertices.len();

    for py in y_min..y_max {
        for px in x_min..x_max {
            let point = (px as f32, py as f32);

            // Find minimum distance to any edge or vertex
            let mut min_dist = f32::MAX;

            // Check distance to each edge
            for i in 0..n {
                let p1 = vertices[i];
                let p2 = vertices[(i + 1) % n];
                let dist = point_to_segment_distance(point, p1, p2);
                min_dist = min_dist.min(dist);
            }

            // Check if pixel should be drawn (within thickness band)
            if min_dist <= half_t + 0.5 {
                // Calculate anti-aliasing alpha
                let alpha = if min_dist > half_t {
                    1.0 - (min_dist - half_t)
                } else {
                    1.0
                };

                if alpha > 0.0 {
                    let alpha = alpha.clamp(0.0, 1.0);
                    let existing = canvas.get_pixel(px, py);

                    // Blend with existing pixel
                    let blend_alpha = (color.0[3] as f32 / 255.0) * alpha;
                    let inv_alpha = 1.0 - blend_alpha;

                    let r =
                        (color.0[0] as f32 * blend_alpha + existing.0[0] as f32 * inv_alpha) as u8;
                    let g =
                        (color.0[1] as f32 * blend_alpha + existing.0[1] as f32 * inv_alpha) as u8;
                    let b =
                        (color.0[2] as f32 * blend_alpha + existing.0[2] as f32 * inv_alpha) as u8;
                    let a =
                        ((blend_alpha + existing.0[3] as f32 / 255.0 * inv_alpha) * 255.0) as u8;

                    canvas.put_pixel(px, py, Rgba([r, g, b, a]));
                }
            }
        }
    }
}

/// Calculate the distance from a point to a line segment.
fn point_to_segment_distance(point: (f32, f32), p1: (f32, f32), p2: (f32, f32)) -> f32 {
    let (px, py) = point;
    let (x1, y1) = p1;
    let (x2, y2) = p2;

    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;

    if len_sq < 0.0001 {
        // Segment is essentially a point
        return ((px - x1).powi(2) + (py - y1).powi(2)).sqrt();
    }

    // Project point onto the line, clamping to segment
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);

    let closest_x = x1 + t * dx;
    let closest_y = y1 + t * dy;

    ((px - closest_x).powi(2) + (py - closest_y).powi(2)).sqrt()
}

/// Draw a thick star outline.
#[allow(clippy::too_many_arguments)]
fn draw_star_thick(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    outer_radius: f32,
    inner_ratio: f32,
    points: usize,
    color: Rgba<u8>,
    thickness: usize,
) {
    let inner_radius = outer_radius * inner_ratio;
    let mut vertices = Vec::with_capacity(points * 2);

    for i in 0..(points * 2) {
        let angle = (i as f32) * PI / (points as f32) - PI / 2.0;
        let r = if i % 2 == 0 {
            outer_radius
        } else {
            inner_radius
        };
        vertices.push((cx + r * angle.cos(), cy + r * angle.sin()));
    }

    draw_polygon_outline_smooth(canvas, &vertices, thickness as f32, color);
}

/// Draw a thick diamond outline.
fn draw_diamond_thick(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    color: Rgba<u8>,
    thickness: usize,
) {
    let vertices = [
        (cx, cy - radius), // top
        (cx + radius, cy), // right
        (cx, cy + radius), // bottom
        (cx - radius, cy), // left
    ];

    draw_polygon_outline_smooth(canvas, &vertices, thickness as f32, color);
}

/// Draw a thick triangle outline.
fn draw_triangle_thick(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    color: Rgba<u8>,
    thickness: usize,
) {
    let angle_offset = -PI / 2.0;
    let vertices: Vec<_> = (0..3)
        .map(|i| {
            let angle = angle_offset + (i as f32) * 2.0 * PI / 3.0;
            (cx + radius * angle.cos(), cy + radius * angle.sin())
        })
        .collect();

    draw_polygon_outline_smooth(canvas, &vertices, thickness as f32, color);
}

/// Draw a thick rounded square outline using smooth distance-based drawing.
#[allow(clippy::too_many_arguments)]
fn draw_rounded_square_thick(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    corner_ratio: f32,
    color: Rgba<u8>,
    thickness: usize,
) {
    let half_t = thickness as f32 / 2.0;
    let (width, height) = canvas.dimensions();

    // Outer boundary includes thickness extending outward
    let outer_r = radius + half_t;

    let x_min = ((cx - outer_r - 1.0).max(0.0)) as u32;
    let x_max = ((cx + outer_r + 2.0).min(width as f32)) as u32;
    let y_min = ((cy - outer_r - 1.0).max(0.0)) as u32;
    let y_max = ((cy + outer_r + 2.0).min(height as f32)) as u32;

    for py in y_min..y_max {
        for px in x_min..x_max {
            let point_x = px as f32;
            let point_y = py as f32;

            // Calculate distance to rounded square boundary
            let dist = distance_to_rounded_rect(
                point_x,
                point_y,
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                (radius * 2.0 * corner_ratio.clamp(0.0, 0.5)).max(1.0),
            );

            // Check if pixel is within the outline band
            if dist.abs() <= half_t + 0.5 {
                let alpha = if dist.abs() > half_t {
                    1.0 - (dist.abs() - half_t)
                } else {
                    1.0
                };

                if alpha > 0.0 {
                    let alpha = alpha.clamp(0.0, 1.0);
                    let existing = canvas.get_pixel(px, py);

                    let blend_alpha = (color.0[3] as f32 / 255.0) * alpha;
                    let inv_alpha = 1.0 - blend_alpha;

                    let r =
                        (color.0[0] as f32 * blend_alpha + existing.0[0] as f32 * inv_alpha) as u8;
                    let g =
                        (color.0[1] as f32 * blend_alpha + existing.0[1] as f32 * inv_alpha) as u8;
                    let b =
                        (color.0[2] as f32 * blend_alpha + existing.0[2] as f32 * inv_alpha) as u8;
                    let a =
                        ((blend_alpha + existing.0[3] as f32 / 255.0 * inv_alpha) * 255.0) as u8;

                    canvas.put_pixel(px, py, Rgba([r, g, b, a]));
                }
            }
        }
    }
}

/// Calculate signed distance to a rounded rectangle boundary.
/// Returns negative if inside, positive if outside.
fn distance_to_rounded_rect(
    px: f32,
    py: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    corner_radius: f32,
) -> f32 {
    let cx = (x1 + x2) / 2.0;
    let cy = (y1 + y2) / 2.0;
    let half_w = (x2 - x1) / 2.0;
    let half_h = (y2 - y1) / 2.0;

    // Transform to first quadrant
    let qx = (px - cx).abs();
    let qy = (py - cy).abs();

    // Distance calculation for rounded rectangle
    let corner_r = corner_radius.min(half_w).min(half_h);

    if qx <= half_w - corner_r && qy <= half_h {
        // In horizontal band
        qy - half_h
    } else if qy <= half_h - corner_r && qx <= half_w {
        // In vertical band
        qx - half_w
    } else {
        // In corner region
        let corner_cx = half_w - corner_r;
        let corner_cy = half_h - corner_r;
        let dx = qx - corner_cx;
        let dy = qy - corner_cy;
        if dx > 0.0 && dy > 0.0 {
            (dx * dx + dy * dy).sqrt() - corner_r
        } else if dx > 0.0 {
            qx - half_w
        } else {
            qy - half_h
        }
    }
}

/// Draw an X shape (diagonal cross).
fn draw_x(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    thickness: usize,
    color: Rgba<u8>,
) {
    let r = radius * 0.85; // Slightly smaller for visual balance

    // Use thick line drawing for proper X shape without edge gaps
    draw_line_solid_thick(canvas, (cx - r, cy - r), (cx + r, cy + r), color, thickness);
    draw_line_solid_thick(canvas, (cx + r, cy - r), (cx - r, cy + r), color, thickness);
}

/// Draw a rounded square.
fn draw_rounded_square(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    corner_ratio: f32,
    color: Rgba<u8>,
    fill: bool,
) {
    let side = radius * 2.0;
    let corner_r = (side * corner_ratio.clamp(0.0, 0.5)).max(1.0);
    let x1 = cx - radius;
    let y1 = cy - radius;
    let x2 = cx + radius;
    let y2 = cy + radius;

    if fill {
        // Fill the main rectangle area (excluding corners)
        // Center horizontal strip
        for py in (y1 + corner_r) as i32..(y2 - corner_r) as i32 {
            for px in x1 as i32..x2 as i32 {
                if px >= 0
                    && py >= 0
                    && (px as u32) < canvas.width()
                    && (py as u32) < canvas.height()
                {
                    canvas.put_pixel(px as u32, py as u32, color);
                }
            }
        }
        // Top and bottom strips
        for py in y1 as i32..(y1 + corner_r) as i32 {
            for px in (x1 + corner_r) as i32..(x2 - corner_r) as i32 {
                if px >= 0
                    && py >= 0
                    && (px as u32) < canvas.width()
                    && (py as u32) < canvas.height()
                {
                    canvas.put_pixel(px as u32, py as u32, color);
                }
            }
        }
        for py in (y2 - corner_r) as i32..y2 as i32 {
            for px in (x1 + corner_r) as i32..(x2 - corner_r) as i32 {
                if px >= 0
                    && py >= 0
                    && (px as u32) < canvas.width()
                    && (py as u32) < canvas.height()
                {
                    canvas.put_pixel(px as u32, py as u32, color);
                }
            }
        }
        // Fill corner circles
        let corners = [
            (x1 + corner_r, y1 + corner_r),
            (x2 - corner_r, y1 + corner_r),
            (x1 + corner_r, y2 - corner_r),
            (x2 - corner_r, y2 - corner_r),
        ];
        for (ccx, ccy) in corners {
            imageproc::drawing::draw_filled_circle_mut(
                canvas,
                (ccx as i32, ccy as i32),
                corner_r as i32,
                color,
            );
        }
    } else {
        // Draw outline only
        // Straight edges
        draw_line_solid(canvas, (x1 + corner_r, y1), (x2 - corner_r, y1), color);
        draw_line_solid(canvas, (x1 + corner_r, y2), (x2 - corner_r, y2), color);
        draw_line_solid(canvas, (x1, y1 + corner_r), (x1, y2 - corner_r), color);
        draw_line_solid(canvas, (x2, y1 + corner_r), (x2, y2 - corner_r), color);

        // Corner arcs
        let steps = (corner_r * 4.0).max(8.0) as usize;
        let draw_arc = |canvas: &mut RgbaImage, acx: f32, acy: f32, start: f32, end: f32| {
            let delta = end - start;
            for i in 0..steps {
                let a1 = start + (i as f32 / steps as f32) * delta;
                let a2 = start + ((i + 1) as f32 / steps as f32) * delta;
                draw_line_solid(
                    canvas,
                    (acx + corner_r * a1.cos(), acy + corner_r * a1.sin()),
                    (acx + corner_r * a2.cos(), acy + corner_r * a2.sin()),
                    color,
                );
            }
        };
        draw_arc(canvas, x1 + corner_r, y1 + corner_r, PI, PI * 1.5);
        draw_arc(canvas, x2 - corner_r, y1 + corner_r, PI * 1.5, PI * 2.0);
        draw_arc(canvas, x2 - corner_r, y2 - corner_r, 0.0, PI * 0.5);
        draw_arc(canvas, x1 + corner_r, y2 - corner_r, PI * 0.5, PI);
    }
}

/// Apply halo effect(Supervision style) to an image based on a binary mask.
///
/// This creates the Supervision-style halo effect:
/// 1. Background becomes grayscale
/// 2. Masked regions keep original color
/// 3. A colored glow radiates from mask edges
pub fn apply_mask_halo(
    canvas: &mut RgbaImage,
    mask: &image::GrayImage,
    glow_radius: usize,
    glow_color: [u8; 4],
) {
    let (width, height) = canvas.dimensions();
    let (mask_w, mask_h) = mask.dimensions();

    // Calculate offset to center mask on canvas
    let offset_x = ((width as i32 - mask_w as i32) / 2).max(0) as u32;
    let offset_y = ((height as i32 - mask_h as i32) / 2).max(0) as u32;

    // First pass: compute distance from mask edge for glow
    // Create a distance field from the mask
    let mut distance_field = vec![f32::MAX; (width * height) as usize];

    // Simple distance transform approximation
    for my in 0..mask_h {
        for mx in 0..mask_w {
            let mask_val = mask.get_pixel(mx, my).0[0];
            if mask_val > 0 {
                let px = mx + offset_x;
                let py = my + offset_y;
                if px < width && py < height {
                    // This pixel is inside the mask
                    distance_field[(py * width + px) as usize] = 0.0;
                }
            }
        }
    }

    // Propagate distances (simplified distance transform)
    let radius = glow_radius as i32;
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if distance_field[idx] == 0.0 {
                // Inside mask - spread distance to neighbors
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                            let dist = ((dx * dx + dy * dy) as f32).sqrt();
                            let nidx = (ny as u32 * width + nx as u32) as usize;
                            if dist < distance_field[nidx] {
                                distance_field[nidx] = dist;
                            }
                        }
                    }
                }
            }
        }
    }

    // Second pass: apply effects
    let [gr, gg, gb, ga] = glow_color;
    let glow_r = glow_radius as f32;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let dist = distance_field[idx];
            let pixel = canvas.get_pixel(x, y);
            let [r, g, b, a] = pixel.0;

            // Check if inside mask
            let mx = x.saturating_sub(offset_x);
            let my = y.saturating_sub(offset_y);
            let in_mask = mx < mask_w && my < mask_h && mask.get_pixel(mx, my).0[0] > 0;

            if in_mask {
                // Keep original color for masked pixels
                // No change needed
            } else if dist < glow_r {
                // In glow zone - blend glow color with grayscale background
                let gray = ((r as u32 + g as u32 + b as u32) / 3) as u8;

                // Glow intensity decreases with distance
                let glow_factor = 1.0 - (dist / glow_r);
                let glow_alpha = (ga as f32 * glow_factor) as u8;

                // Blend glow color with grayscale
                let blend = |base: u8, overlay: u8, alpha: u8| -> u8 {
                    let af = alpha as f32 / 255.0;
                    (base as f32 * (1.0 - af) + overlay as f32 * af) as u8
                };

                let new_r = blend(gray, gr, glow_alpha);
                let new_g = blend(gray, gg, glow_alpha);
                let new_b = blend(gray, gb, glow_alpha);

                canvas.put_pixel(x, y, Rgba([new_r, new_g, new_b, a]));
            } else {
                // Outside glow zone - grayscale only
                let gray = ((r as u32 + g as u32 + b as u32) / 3) as u8;
                canvas.put_pixel(x, y, Rgba([gray, gray, gray, a]));
            }
        }
    }
}

/// Draw a keypoint with radial glow effect - color fades from center outward.
fn draw_keypoint_glow(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    glow_multiplier: f32,
    color: Rgba<u8>,
) {
    let (w, h) = canvas.dimensions();
    let glow_radius = radius * glow_multiplier;
    let [cr, cg, cb, ca] = color.0;

    // Calculate bounding box for the glow
    let x_min = ((cx - glow_radius).floor() as i32).max(0) as u32;
    let x_max = ((cx + glow_radius).ceil() as i32).min(w as i32) as u32;
    let y_min = ((cy - glow_radius).floor() as i32).max(0) as u32;
    let y_max = ((cy + glow_radius).ceil() as i32).min(h as i32) as u32;

    // Draw radial gradient glow from center
    for y in y_min..y_max {
        for x in x_min..x_max {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist <= glow_radius {
                let src = canvas.get_pixel(x, y);
                let [sr, sg, sb, sa] = src.0;

                // Glow intensity with quadratic falloff for stronger center
                let t = dist / glow_radius;
                let intensity = (1.0 - t * t).max(0.0); // Quadratic: more concentrated at center
                let alpha = intensity * (ca as f32 / 255.0);

                // Blend: glow color over source
                let nr = (cr as f32 * alpha + sr as f32 * (1.0 - alpha)) as u8;
                let ng = (cg as f32 * alpha + sg as f32 * (1.0 - alpha)) as u8;
                let nb = (cb as f32 * alpha + sb as f32 * (1.0 - alpha)) as u8;

                canvas.put_pixel(x, y, Rgba([nr, ng, nb, sa]));
            }
        }
    }
}
