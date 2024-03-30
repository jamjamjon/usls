use crate::{Point, Rect, RotatedRect};

#[derive(Default, Debug, PartialOrd, PartialEq, Clone)]
pub struct Polygon {
    points: Vec<Point>,
}

impl Polygon {
    pub fn new(points: &[Point]) -> Self {
        // TODO: refactor
        Self {
            points: points.to_vec(),
        }
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
        ((min_x, min_y), (max_x, max_y)).into()
    }

    pub fn find_min_rotated_rect() -> RotatedRect {
        todo!()
    }

    pub fn expand(&mut self) -> Self {
        todo!()
    }
}
