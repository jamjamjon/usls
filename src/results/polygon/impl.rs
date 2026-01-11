use aksr::Builder;

use crate::{Hbb, InstanceMeta, Mask, Obb, PolygonStyle};

/// Polygon with metadata.
#[derive(Builder, Clone, Default)]
pub struct Polygon {
    coords: Vec<[f32; 2]>, // NOT automatically closed
    meta: InstanceMeta,
    style: Option<PolygonStyle>,
}

impl std::fmt::Debug for Polygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Polygon");
        f.field("n_points", &self.count());
        if let Some(id) = &self.meta.id() {
            f.field("id", id);
        }
        if let Some(name) = &self.meta.name() {
            f.field("name", name);
        }
        if let Some(confidence) = &self.meta.confidence() {
            f.field("confidence", confidence);
        }
        if let Some(track_id) = &self.meta.track_id() {
            f.field("track_id", track_id);
        }
        f.finish()
    }
}

impl PartialEq for Polygon {
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

impl TryFrom<Vec<[f32; 2]>> for Polygon {
    type Error = anyhow::Error;

    fn try_from(coords: Vec<[f32; 2]>) -> Result<Self, Self::Error> {
        if coords.len() < 3 {
            return Err(anyhow::anyhow!(
                "Polygon must have at least 3 points, got {}",
                coords.len()
            ));
        }
        Ok(Self {
            coords,
            ..Default::default()
        })
    }
}

impl TryFrom<&[[f32; 2]]> for Polygon {
    type Error = anyhow::Error;

    fn try_from(coords: &[[f32; 2]]) -> Result<Self, Self::Error> {
        if coords.len() < 3 {
            return Err(anyhow::anyhow!(
                "Polygon must have at least 3 points, got {}",
                coords.len()
            ));
        }
        Ok(Self {
            coords: coords.to_vec(),
            ..Default::default()
        })
    }
}

impl<const N: usize> TryFrom<[[f32; 2]; N]> for Polygon {
    type Error = anyhow::Error;

    fn try_from(coords: [[f32; 2]; N]) -> Result<Self, Self::Error> {
        if N < 3 {
            return Err(anyhow::anyhow!(
                "Polygon must have at least 3 points, got {}",
                N
            ));
        }
        Ok(Self {
            coords: coords.to_vec(),
            ..Default::default()
        })
    }
}

impl<const N: usize> TryFrom<&[[f32; 2]; N]> for Polygon {
    type Error = anyhow::Error;

    fn try_from(coords: &[[f32; 2]; N]) -> Result<Self, Self::Error> {
        if N < 3 {
            return Err(anyhow::anyhow!(
                "Polygon must have at least 3 points, got {}",
                N
            ));
        }
        Ok(Self {
            coords: coords.to_vec(),
            ..Default::default()
        })
    }
}

impl TryFrom<Vec<Vec<f32>>> for Polygon {
    type Error = anyhow::Error;

    fn try_from(coords: Vec<Vec<f32>>) -> Result<Self, Self::Error> {
        if coords.len() < 3 {
            return Err(anyhow::anyhow!(
                "Polygon must have at least 3 points, got {}",
                coords.len()
            ));
        }

        let coords: Vec<[f32; 2]> = coords
            .iter()
            .map(|p| {
                if p.len() >= 2 {
                    Ok([p[0], p[1]])
                } else {
                    Err(anyhow::anyhow!("Point must have at least 2 coordinates"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            coords,
            ..Default::default()
        })
    }
}

impl Polygon {
    impl_meta_methods!();

    pub fn is_closed(&self) -> bool {
        if self.coords.len() < 2 {
            return false;
        }
        let first = self.coords[0];
        let last = self.coords[self.coords.len() - 1];
        first[0] == last[0] && first[1] == last[1]
    }

    /// Get the number of points
    pub fn count(&self) -> usize {
        self.coords.len()
    }

    /// Calculate perimeter (Euclidean length)
    pub fn perimeter(&self) -> f64 {
        let mut length = 0.0;
        for i in 0..self.coords.len().saturating_sub(1) {
            let dx = self.coords[i + 1][0] - self.coords[i][0];
            let dy = self.coords[i + 1][1] - self.coords[i][1];
            length += (dx * dx + dy * dy).sqrt();
        }
        length as f64
    }

    /// Calculate unsigned area using the Shoelace formula
    /// Handles both closed and unclosed polygons
    pub fn area(&self) -> f64 {
        if self.coords.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = self.coords.len();

        // Calculate for all edges
        for i in 0..n - 1 {
            area += self.coords[i][0] * self.coords[i + 1][1];
            area -= self.coords[i + 1][0] * self.coords[i][1];
        }

        // Close the polygon if not already closed
        if !self.is_closed() {
            area += self.coords[n - 1][0] * self.coords[0][1];
            area -= self.coords[0][0] * self.coords[n - 1][1];
        }

        (area.abs() / 2.0) as f64
    }

    /// Calculate centroid
    /// Handles both closed and unclosed polygons
    pub fn centroid(&self) -> Option<(f32, f32)> {
        if self.coords.len() < 3 {
            return None;
        }

        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut area = 0.0;

        let n = self.coords.len();
        let is_closed = self.is_closed();
        let limit = if is_closed { n - 1 } else { n };

        for i in 0..limit {
            let j = (i + 1) % n;
            let cross =
                self.coords[i][0] * self.coords[j][1] - self.coords[j][0] * self.coords[i][1];
            cx += (self.coords[i][0] + self.coords[j][0]) * cross;
            cy += (self.coords[i][1] + self.coords[j][1]) * cross;
            area += cross;
        }

        area /= 2.0;
        if area.abs() < 1e-10 {
            return None;
        }

        cx /= 6.0 * area;
        cy /= 6.0 * area;

        Some((cx, cy))
    }

    pub fn intersect(&self, other: &Self) -> f32 {
        self.intersection_poly(other).area() as f32
    }

    pub fn union(&self, other: &Self) -> f32 {
        self.union_poly(other).area() as f32
    }

    pub fn points(&self) -> Vec<[f32; 2]> {
        self.coords.to_vec()
    }

    /// Get exterior coordinates (for compatibility)
    #[allow(dead_code)]
    pub fn exterior(&self) -> &[[f32; 2]] {
        &self.coords
    }

    pub fn mask(&self) -> Mask {
        todo!()
    }

    pub fn hbb(&self) -> Option<Hbb> {
        self.bounding_box().map(|bbox| {
            let mut hbb = Hbb::default().with_xyxy(bbox[0], bbox[1], bbox[2], bbox[3]);
            if let Some(id) = self.id() {
                hbb = hbb.with_id(id);
            }
            if let Some(name) = self.name() {
                hbb = hbb.with_name(name);
            }
            if let Some(confidence) = self.confidence() {
                hbb = hbb.with_confidence(confidence);
            }

            hbb
        })
    }

    pub fn obb(&self) -> Option<Obb> {
        self.minimum_rotated_rect().map(|rect_poly| {
            let xy4 = rect_poly.coords.to_vec();
            let mut obb = Obb::from(xy4);

            if let Some(id) = self.id() {
                obb = obb.with_id(id);
            }
            if let Some(name) = self.name() {
                obb = obb.with_name(name);
            }
            if let Some(confidence) = self.confidence() {
                obb = obb.with_confidence(confidence);
            }

            obb
        })
    }

    pub fn convex_hull(mut self) -> Self {
        self.coords = self.convex_hull_coords();
        self
    }

    pub fn simplify(mut self, eps: f64) -> Self {
        self.coords = self.simplify_coords(eps as f32);
        self
    }

    pub fn resample(mut self, num_samples: usize) -> Self {
        let points = &self.coords;
        let mut new_points = Vec::new();
        let is_closed = self.is_closed();
        let num_edges = if is_closed {
            points.len() - 1
        } else {
            points.len()
        };

        for i in 0..num_edges {
            let start_point = points[i];
            let end_point = points[(i + 1) % points.len()];
            new_points.push(start_point);
            let dx = end_point[0] - start_point[0];
            let dy = end_point[1] - start_point[1];
            for j in 1..num_samples {
                let t = (j as f32) / (num_samples as f32);
                let new_x = start_point[0] + t * dx;
                let new_y = start_point[1] + t * dy;
                new_points.push([new_x, new_y]);
            }
        }
        self.coords = new_points;
        self
    }

    pub fn unclip(mut self, delta: f64, width: f64, height: f64) -> Self {
        let points = &self.coords;

        // Determine actual number of unique points (if closed, last point is duplicate)
        let is_closed = self.is_closed();
        let num_points = if is_closed {
            points.len() - 1
        } else {
            points.len()
        };

        let mut new_points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let prev_idx = if i == 0 { num_points - 1 } else { i - 1 };
            let next_idx = (i + 1) % num_points;

            let edge_vector_x = points[next_idx][0] - points[prev_idx][0];
            let edge_vector_y = points[next_idx][1] - points[prev_idx][1];

            let normal_vector_x = -edge_vector_y;
            let normal_vector_y = edge_vector_x;

            let normal_length = (normal_vector_x.powi(2) + normal_vector_y.powi(2)).sqrt();
            if normal_length.abs() < 1e-6 {
                new_points.push(points[i]);
            } else {
                let normalized_normal_x = normal_vector_x / normal_length;
                let normalized_normal_y = normal_vector_y / normal_length;

                let delta_f = delta as f32;
                let width_f = width as f32;
                let height_f = height as f32;
                let new_x = points[i][0] + normalized_normal_x * delta_f;
                let new_y = points[i][1] + normalized_normal_y * delta_f;
                let new_x = new_x.max(0.0).min(width_f);
                let new_y = new_y.max(0.0).min(height_f);
                new_points.push([new_x, new_y]);
            }
        }

        self.coords = new_points;
        self
    }

    pub fn verify(mut self) -> Self {
        // Remove duplicates and redundant points
        let mut points = self.coords.clone();
        Self::remove_duplicates(&mut points);
        self.coords = points;
        self
    }

    fn remove_duplicates(xs: &mut Vec<[f32; 2]>) {
        // Step 1: Remove elements from the end if they match the first element
        if let Some(first) = xs.first() {
            let p_1st_x = first[0] as i32;
            let p_1st_y = first[1] as i32;
            while xs.len() > 1 {
                if let Some(last) = xs.last() {
                    if last[0] as i32 == p_1st_x && last[1] as i32 == p_1st_y {
                        xs.pop();
                    } else {
                        break;
                    }
                }
            }
        }

        // Step 2: Remove duplicates
        let mut seen = std::collections::HashSet::new();
        xs.retain(|point| seen.insert((point[0] as i32, point[1] as i32)));
    }

    /// Get bounding rectangle [min_x, min_y, max_x, max_y]
    fn bounding_box(&self) -> Option<[f32; 4]> {
        if self.coords.is_empty() {
            return None;
        }

        let mut min_x = self.coords[0][0];
        let mut min_y = self.coords[0][1];
        let mut max_x = self.coords[0][0];
        let mut max_y = self.coords[0][1];

        for coord in self.coords.iter().skip(1) {
            min_x = min_x.min(coord[0]);
            min_y = min_y.min(coord[1]);
            max_x = max_x.max(coord[0]);
            max_y = max_y.max(coord[1]);
        }

        Some([min_x, min_y, max_x, max_y])
    }

    /// Convex hull using Andrew's monotone chain algorithm
    fn convex_hull_coords(&self) -> Vec<[f32; 2]> {
        let mut points: Vec<[f32; 2]> = self.coords.clone();

        if points.len() < 3 {
            return points;
        }

        // Remove duplicate last point if closed
        if points.len() >= 2 {
            let n = points.len();
            if points[0][0] == points[n - 1][0] && points[0][1] == points[n - 1][1] {
                points.pop();
            }
        }

        if points.len() < 3 {
            return points;
        }

        // Sort points lexicographically (first by x, then by y)
        points.sort_by(|a, b| {
            a[0].partial_cmp(&b[0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Build lower hull
        let mut lower: Vec<[f32; 2]> = Vec::new();
        for point in &points {
            while lower.len() >= 2 {
                let p2 = lower[lower.len() - 1];
                let p1 = lower[lower.len() - 2];
                let cross =
                    (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0]);
                if cross <= 0.0 {
                    lower.pop();
                } else {
                    break;
                }
            }
            lower.push(*point);
        }

        // Build upper hull
        let mut upper: Vec<[f32; 2]> = Vec::new();
        for point in points.iter().rev() {
            while upper.len() >= 2 {
                let p2 = upper[upper.len() - 1];
                let p1 = upper[upper.len() - 2];
                let cross =
                    (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0]);
                if cross <= 0.0 {
                    upper.pop();
                } else {
                    break;
                }
            }
            upper.push(*point);
        }

        // Remove last point of each half because it's repeated
        lower.pop();
        upper.pop();

        // Concatenate lower and upper hull
        lower.extend(upper);

        lower
    }

    /// Simplify using Ramer-Douglas-Peucker algorithm
    fn simplify_coords(&self, epsilon: f32) -> Vec<[f32; 2]> {
        if self.coords.len() <= 2 {
            return self.coords.clone();
        }

        let mut points = self.coords.clone();

        // Remove duplicate last point if closed
        let is_closed = points.len() >= 2
            && points[0][0] == points[points.len() - 1][0]
            && points[0][1] == points[points.len() - 1][1];
        if is_closed {
            points.pop();
        }

        Self::rdp_simplify(&points, epsilon)
    }

    fn rdp_simplify(points: &[[f32; 2]], epsilon: f32) -> Vec<[f32; 2]> {
        if points.len() <= 2 {
            return points.to_vec();
        }

        let mut dmax = 0.0;
        let mut index = 0;

        for i in 1..points.len() - 1 {
            let d = Self::perpendicular_distance(&points[i], &points[0], &points[points.len() - 1]);
            if d > dmax {
                index = i;
                dmax = d;
            }
        }

        if dmax > epsilon {
            let mut left = Self::rdp_simplify(&points[0..=index], epsilon);
            let right = Self::rdp_simplify(&points[index..], epsilon);
            left.pop(); // Remove duplicate point
            left.extend(right);
            left
        } else {
            vec![points[0], points[points.len() - 1]]
        }
    }

    fn perpendicular_distance(point: &[f32; 2], line_start: &[f32; 2], line_end: &[f32; 2]) -> f32 {
        let dx = line_end[0] - line_start[0];
        let dy = line_end[1] - line_start[1];
        let mag = (dx * dx + dy * dy).sqrt();

        if mag < 1e-10 {
            let dx = point[0] - line_start[0];
            let dy = point[1] - line_start[1];
            return (dx * dx + dy * dy).sqrt();
        }

        let u = ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy) / (mag * mag);
        let closest_x = line_start[0] + u * dx;
        let closest_y = line_start[1] + u * dy;

        let dx = point[0] - closest_x;
        let dy = point[1] - closest_y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Minimum rotated rectangle (OBB) - returns 4 corner points
    fn minimum_rotated_rect(&self) -> Option<Polygon> {
        // First get convex hull
        let hull_coords = self.convex_hull_coords();

        if hull_coords.len() < 3 {
            return None;
        }

        let mut min_area = f32::MAX;
        let mut best_rect = None;

        // Try each edge of the convex hull
        for i in 0..hull_coords.len() - 1 {
            let edge_x = hull_coords[i + 1][0] - hull_coords[i][0];
            let edge_y = hull_coords[i + 1][1] - hull_coords[i][1];
            let angle = edge_y.atan2(edge_x);

            // Rotate all points
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;

            for coord in hull_coords.iter() {
                let rx = coord[0] * cos_a + coord[1] * sin_a;
                let ry = -coord[0] * sin_a + coord[1] * cos_a;
                min_x = min_x.min(rx);
                max_x = max_x.max(rx);
                min_y = min_y.min(ry);
                max_y = max_y.max(ry);
            }

            let area = (max_x - min_x) * (max_y - min_y);
            if area < min_area {
                min_area = area;

                // Create rectangle corners in rotated space
                let corners = [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                ];

                // Rotate back
                let rect_coords: Vec<[f32; 2]> = corners
                    .iter()
                    .map(|(rx, ry)| {
                        let x = rx * cos_a - ry * sin_a;
                        let y = rx * sin_a + ry * cos_a;
                        [x, y]
                    })
                    .collect();

                best_rect = Some(Polygon {
                    coords: rect_coords,
                    ..Default::default()
                });
            }
        }

        best_rect
    }

    /// Calculate intersection area with another polygon (simplified Sutherland-Hodgman)
    /// Handles both closed and unclosed polygons
    fn intersection_poly(&self, other: &Polygon) -> Polygon {
        let mut output: Vec<[f32; 2]> = self.coords.clone();

        let clip_n = other.coords.len();
        if clip_n < 3 {
            return Polygon::default();
        }

        // For each edge in the clipping polygon
        // Handle both closed and unclosed polygons
        let num_edges = if other.is_closed() {
            clip_n - 1
        } else {
            clip_n
        };

        for i in 0..num_edges {
            let edge_start = other.coords[i];
            let edge_end = other.coords[(i + 1) % clip_n];
            let input = output;
            output = Vec::new();

            if input.is_empty() {
                break;
            }

            let input_n = input.len();
            for j in 0..input_n {
                let current = input[j];
                let next = input[(j + 1) % input_n];

                let current_inside = Self::is_inside(&current, &edge_start, &edge_end);
                let next_inside = Self::is_inside(&next, &edge_start, &edge_end);

                if next_inside {
                    if !current_inside {
                        if let Some(intersection) =
                            Self::line_intersection(&current, &next, &edge_start, &edge_end)
                        {
                            output.push(intersection);
                        }
                    }
                    output.push(next);
                } else if current_inside {
                    if let Some(intersection) =
                        Self::line_intersection(&current, &next, &edge_start, &edge_end)
                    {
                        output.push(intersection);
                    }
                }
            }
        }

        if output.is_empty() {
            Polygon::default()
        } else {
            Polygon {
                coords: output,
                ..Default::default()
            }
        }
    }

    fn is_inside(point: &[f32; 2], edge_start: &[f32; 2], edge_end: &[f32; 2]) -> bool {
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
            >= 0.0
    }

    fn line_intersection(
        p1: &[f32; 2],
        p2: &[f32; 2],
        p3: &[f32; 2],
        p4: &[f32; 2],
    ) -> Option<[f32; 2]> {
        let x1 = p1[0];
        let y1 = p1[1];
        let x2 = p2[0];
        let y2 = p2[1];
        let x3 = p3[0];
        let y3 = p3[1];
        let x4 = p4[0];
        let y4 = p4[1];

        let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if denom.abs() < 1e-10 {
            return None;
        }

        let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;

        Some([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    }

    /// Union operation (simplified - returns bounding box approximation)
    fn union_poly(&self, other: &Polygon) -> Polygon {
        // Simplified: combine bounding boxes for approximation
        // For exact union would need Weiler-Atherton algorithm
        if let (Some(bbox1), Some(bbox2)) = (self.bounding_box(), other.bounding_box()) {
            let min_x = bbox1[0].min(bbox2[0]);
            let min_y = bbox1[1].min(bbox2[1]);
            let max_x = bbox1[2].max(bbox2[2]);
            let max_y = bbox1[3].max(bbox2[3]);

            let coords = vec![
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ];
            return Polygon {
                coords,
                ..Default::default()
            };
        }

        // Fallback: return the larger polygon
        if self.area() >= other.area() {
            self.clone()
        } else {
            other.clone()
        }
    }
}
