use aksr::Builder;
use geo::{
    coord, point, polygon, Area, BooleanOps, Centroid, ConvexHull, Euclidean, Length, LineString,
    Point, Simplify,
};

use crate::{Hbb, InstanceMeta, Mask, Obb, Style};

/// Polygon.
#[derive(Builder, Clone)]
pub struct Polygon {
    polygon: geo::Polygon, // TODO: Vec<[f32; 2]>
    meta: InstanceMeta,
    style: Option<Style>,
}

impl Default for Polygon {
    fn default() -> Self {
        Self {
            polygon: polygon![],
            meta: InstanceMeta::default(),
            style: None,
        }
    }
}

impl std::fmt::Debug for Polygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Polygon")
            .field("n_points", &self.count())
            .field("id", &self.meta.id())
            .field("name", &self.meta.name())
            .field("confidence", &self.meta.confidence())
            .finish()
    }
}

impl PartialEq for Polygon {
    fn eq(&self, other: &Self) -> bool {
        self.polygon == other.polygon
    }
}

impl Polygon {
    impl_meta_methods!();
    pub fn with_points_imageproc(mut self, points: &[imageproc::point::Point<i32>]) -> Self {
        // exterior
        let v = points
            .iter()
            .map(|p| coord! { x: p.x as f64, y: p.y as f64})
            .collect::<Vec<_>>();
        self.polygon = geo::Polygon::new(LineString::from(v), vec![]);
        self
    }

    pub fn from_xys(xys: &[[f32; 2]]) -> Self {
        // exterior
        let v = xys
            .iter()
            .map(|p| coord! { x: p[0] as f64, y: p[1] as f64})
            .collect::<Vec<_>>();
        let polygon = geo::Polygon::new(LineString::from(v), vec![]);

        Self {
            polygon,
            ..Default::default()
        }
    }

    pub fn with_points(mut self, points: &[Vec<f32>]) -> Self {
        // exterior
        let v = points
            .iter()
            .map(|p| coord! { x: p[0] as f64, y: p[1] as f64})
            .collect::<Vec<_>>();
        self.polygon = geo::Polygon::new(LineString::from(v), vec![]);
        self
    }

    pub fn is_closed(&self) -> bool {
        self.polygon.exterior().is_closed()
    }

    pub fn count(&self) -> usize {
        self.polygon.exterior().points().len()
    }

    pub fn perimeter(&self) -> f64 {
        Euclidean.length(self.polygon.exterior())
    }

    pub fn area(&self) -> f64 {
        self.polygon.unsigned_area()
    }

    pub fn centroid(&self) -> Option<(f32, f32)> {
        self.polygon
            .centroid()
            .map(|x| (x.x() as f32, x.y() as f32))
    }

    pub fn intersect(&self, other: &Self) -> f32 {
        self.polygon.intersection(&other.polygon).unsigned_area() as f32
    }

    pub fn union(&self, other: &Self) -> f32 {
        self.polygon.union(&other.polygon).unsigned_area() as f32
    }

    pub fn points(&self) -> Vec<[f32; 2]> {
        self.polygon
            .exterior()
            .coords()
            .map(|c| [c.x as f32, c.y as f32])
            .collect::<Vec<_>>()
    }

    pub fn mask(&self) -> Mask {
        todo!()
    }

    pub fn hbb(&self) -> Option<Hbb> {
        use geo::BoundingRect;
        self.polygon.bounding_rect().map(|x| {
            let mut hbb = Hbb::default().with_xyxy(
                x.min().x as f32,
                x.min().y as f32,
                x.max().x as f32,
                x.max().y as f32,
            );
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
        use geo::MinimumRotatedRect;
        MinimumRotatedRect::minimum_rotated_rect(&self.polygon).map(|x| {
            let xy4 = x
                .exterior()
                .coords()
                .map(|c| [c.x as f32, c.y as f32])
                .collect::<Vec<_>>();

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
        self.polygon = self.polygon.convex_hull();
        self
    }

    pub fn simplify(mut self, eps: f64) -> Self {
        self.polygon = self.polygon.simplify(&eps);
        self
    }

    pub fn resample(mut self, num_samples: usize) -> Self {
        let points = self.polygon.exterior().to_owned().into_points();
        let mut new_points = Vec::new();
        for i in 0..points.len() {
            let start_point: Point = points[i];
            let end_point = points[(i + 1) % points.len()];
            new_points.push(start_point);
            let dx = end_point.x() - start_point.x();
            let dy = end_point.y() - start_point.y();
            for j in 1..num_samples {
                let t = (j as f64) / (num_samples as f64);
                let new_x = start_point.x() + t * dx;
                let new_y = start_point.y() + t * dy;
                new_points.push(point! { x: new_x, y: new_y });
            }
        }
        self.polygon = geo::Polygon::new(LineString::from(new_points), vec![]);
        self
    }

    pub fn unclip(mut self, delta: f64, width: f64, height: f64) -> Self {
        let points = self.polygon.exterior().to_owned().into_points();
        let num_points = points.len();
        let mut new_points = Vec::with_capacity(points.len());
        for i in 0..num_points {
            let prev_idx = if i == 0 { num_points - 1 } else { i - 1 };
            let next_idx = (i + 1) % num_points;

            let edge_vector = point! {
                x: points[next_idx].x() - points[prev_idx].x(),
                y: points[next_idx].y() - points[prev_idx].y(),
            };
            let normal_vector = point! {
                x: -edge_vector.y(),
                y: edge_vector.x(),
            };

            let normal_length = (normal_vector.x().powi(2) + normal_vector.y().powi(2)).sqrt();
            if normal_length.abs() < 1e-6 {
                new_points.push(points[i]);
            } else {
                let normalized_normal = point! {
                    x: normal_vector.x() / normal_length,
                    y: normal_vector.y() / normal_length,
                };

                let new_x = points[i].x() + normalized_normal.x() * delta;
                let new_y = points[i].y() + normalized_normal.y() * delta;
                let new_x = new_x.max(0.0).min(width);
                let new_y = new_y.max(0.0).min(height);
                new_points.push(point! {
                    x: new_x,
                    y: new_y,
                });
            }
        }

        self.polygon = geo::Polygon::new(LineString::from(new_points), vec![]);
        self
    }

    pub fn verify(mut self) -> Self {
        // Remove duplicates and redundant points
        let mut points = self.polygon.exterior().points().collect::<Vec<_>>();
        Self::remove_duplicates(&mut points);
        self.polygon = geo::Polygon::new(LineString::from(points), vec![]);
        self
    }

    fn remove_duplicates(xs: &mut Vec<Point>) {
        // Step 1: Remove elements from the end if they match the first element
        if let Some(first) = xs.first() {
            let p_1st_x = first.x() as i32;
            let p_1st_y = first.y() as i32;
            while xs.len() > 1 {
                if let Some(last) = xs.last() {
                    if last.x() as i32 == p_1st_x && last.y() as i32 == p_1st_y {
                        xs.pop();
                    } else {
                        break;
                    }
                }
            }
        }

        // Step 2: Remove duplicates
        let mut seen = std::collections::HashSet::new();
        xs.retain(|point| seen.insert((point.x() as i32, point.y() as i32)));
    }
}
