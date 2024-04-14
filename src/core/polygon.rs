use crate::{Point, Rect};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Polygon {
    pub points: Vec<Point>,
}

impl From<Vec<Point>> for Polygon {
    fn from(points: Vec<Point>) -> Self {
        Self { points }
    }
}

impl Polygon {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_contour(contour: &imageproc::contours::Contour<i32>) -> Self {
        let points = contour
            .points
            .iter()
            .map(|p| Point::new(p.x as f32, p.y as f32))
            .collect::<Vec<_>>();
        Self { points }
    }

    pub fn to_imageproc_points(&self) -> Vec<imageproc::point::Point<i32>> {
        self.points
            .iter()
            .map(|p| imageproc::point::Point::new(p.x as i32, p.y as i32))
            .collect::<Vec<_>>()
    }

    pub fn from_imageproc_points(points: &[imageproc::point::Point<i32>]) -> Self {
        let points = points
            .iter()
            .map(|p| Point::new(p.x as f32, p.y as f32))
            .collect::<Vec<_>>();
        Self { points }
    }

    pub fn with_points(mut self, points: &[Point]) {
        self.points = points.to_vec();
    }

    pub fn area(&self) -> f32 {
        // make sure points are already sorted
        let mut area = 0.0;
        let n = self.points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.points[i].x * self.points[j].y;
            area -= self.points[j].x * self.points[i].y;
        }
        area.abs() / 2.0
    }

    pub fn center(&self) -> Point {
        let rect = self.find_min_rect();
        rect.center()
    }

    pub fn find_min_rect(&self) -> Rect {
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for point in self.points.iter() {
            if point.x <= min_x {
                min_x = point.x
            }
            if point.x > max_x {
                max_x = point.x
            }
            if point.y <= min_y {
                min_y = point.y
            }
            if point.y > max_y {
                max_y = point.y
            }
        }
        ((min_x - 1.0, min_y - 1.0), (max_x + 1.0, max_y + 1.0)).into()
    }

    pub fn perimeter(&self) -> f32 {
        let mut perimeter = 0.0;
        let n = self.points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            perimeter += self.points[i].distance_from(&self.points[j]);
        }
        perimeter
    }

    pub fn offset(&self, delta: f32, width: f32, height: f32) -> Self {
        let num_points = self.points.len();
        let mut new_points = Vec::with_capacity(self.points.len());
        for i in 0..num_points {
            let prev_idx = if i == 0 { num_points - 1 } else { i - 1 };
            let next_idx = (i + 1) % num_points;

            let edge_vector = Point {
                x: self.points[next_idx].x - self.points[prev_idx].x,
                y: self.points[next_idx].y - self.points[prev_idx].y,
            };
            let normal_vector = Point {
                x: -edge_vector.y,
                y: edge_vector.x,
            };

            let normal_length = (normal_vector.x.powi(2) + normal_vector.y.powi(2)).sqrt();
            if normal_length.abs() < 1e-6 {
                new_points.push(self.points[i]);
            } else {
                let normalized_normal = Point {
                    x: normal_vector.x / normal_length,
                    y: normal_vector.y / normal_length,
                };

                let new_x = self.points[i].x + normalized_normal.x * delta;
                let new_y = self.points[i].y + normalized_normal.y * delta;
                let new_x = new_x.max(0.0).min(width);
                let new_y = new_y.max(0.0).min(height);
                new_points.push(Point { x: new_x, y: new_y });
            }
        }
        Self { points: new_points }
    }

    pub fn resample(&self, num_samples: usize) -> Polygon {
        let mut points = Vec::new();
        for i in 0..self.points.len() {
            let start_point = self.points[i];
            let end_point = self.points[(i + 1) % self.points.len()];
            points.push(start_point);
            let dx = end_point.x - start_point.x;
            let dy = end_point.y - start_point.y;
            for j in 1..num_samples {
                let t = (j as f32) / (num_samples as f32);
                let new_x = start_point.x + t * dx;
                let new_y = start_point.y + t * dy;
                points.push(Point { x: new_x, y: new_y });
            }
        }
        Self { points }
    }

    pub fn simplify(&self, epsilon: f32) -> Self {
        let mask = self.rdp_iter(epsilon);
        let points = self
            .points
            .iter()
            .enumerate()
            .filter_map(|(i, &point)| if mask[i] { Some(point) } else { None })
            .collect();
        Self { points }
    }

    #[allow(clippy::needless_range_loop)]
    fn rdp_iter(&self, epsilon: f32) -> Vec<bool> {
        let mut stk = Vec::new();
        let mut indices = vec![true; self.points.len()];
        stk.push((0, self.points.len() - 1));
        while let Some((start_index, last_index)) = stk.pop() {
            let mut dmax = 0.0;
            let mut index = start_index;
            for i in (start_index + 1)..last_index {
                let d = self.points[i]
                    .perpendicular_distance(&self.points[start_index], &self.points[last_index]);
                if d > dmax {
                    index = i;
                    dmax = d;
                }
            }

            if dmax > epsilon {
                stk.push((start_index, index));
                stk.push((index, last_index));
            } else {
                for j in (start_index + 1)..last_index {
                    indices[j] = false;
                }
            }
        }

        indices
    }

    pub fn convex_hull(&self) -> Self {
        let mut points = self.points.clone();
        points.sort_by(|a, b| {
            a.x.partial_cmp(&b.x)
                .unwrap()
                .then(a.y.partial_cmp(&b.y).unwrap())
        });
        let mut hull: Vec<Point> = Vec::new();

        // Lower hull
        for &point in &points {
            while hull.len() >= 2 {
                let last = hull.len() - 1;
                let second_last = hull.len() - 2;
                let vec_a = hull[last] - hull[second_last];
                let vec_b = point - hull[second_last];

                if vec_a.cross(&vec_b) <= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(point);
        }

        // Upper hull
        let lower_hull_size = hull.len();
        for &point in points.iter().rev().skip(1) {
            while hull.len() > lower_hull_size {
                let last = hull.len() - 1;
                let second_last = hull.len() - 2;
                let vec_a: Point = hull[last] - hull[second_last];
                let vec_b = point - hull[second_last];

                if vec_a.cross(&vec_b) <= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(point);
        }

        // Remove duplicate points
        hull.dedup();
        if hull.len() > 1 && hull.first() == hull.last() {
            hull.pop();
        }

        Self { points: hull }
    }
}
