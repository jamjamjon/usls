use std::ops::{Add, Div, Mul, Sub};

#[derive(Default, Debug, PartialOrd, PartialEq, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Add<f32> for Point {
    type Output = Self;

    fn add(self, other: f32) -> Self::Output {
        Self {
            x: self.x + other,
            y: self.y + other,
        }
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Sub<f32> for Point {
    type Output = Self;

    fn sub(self, other: f32) -> Self::Output {
        Self {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl Mul<f32> for Point {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        Self {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl Mul for Point {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }
}

impl Div for Point {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
        }
    }
}

impl Div<f32> for Point {
    type Output = Self;

    fn div(self, other: f32) -> Self::Output {
        Self {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl From<(f32, f32)> for Point {
    fn from((x, y): (f32, f32)) -> Self {
        Self { x, y }
    }
}

impl From<Point> for (f32, f32) {
    fn from(Point { x, y }: Point) -> Self {
        (x, y)
    }
}

impl From<[f32; 2]> for Point {
    fn from([x, y]: [f32; 2]) -> Self {
        Self { x, y }
    }
}

impl From<Point> for [f32; 2] {
    fn from(Point { x, y }: Point) -> Self {
        [x, y]
    }
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn coord(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    pub fn is_origin(&self) -> bool {
        self.x == 0.0_f32 && self.y == 0.0_f32
    }

    pub fn distance_from(&self, other: &Point) -> f32 {
        ((self.x - other.x).powf(2.0) + (self.y - other.y).powf(2.0)).sqrt()
    }

    pub fn distance_from_origin(&self) -> f32 {
        (self.x.powf(2.0) + self.y.powf(2.0)).sqrt()
    }

    pub fn sum(&self) -> f32 {
        self.x + self.y
    }

    pub fn perpendicular_distance(&self, start: &Point, end: &Point) -> f32 {
        let numerator = ((end.y - start.y) * self.x - (end.x - start.x) * self.y + end.x * start.y
            - end.y * start.x)
            .abs();
        let denominator = ((end.y - start.y).powi(2) + (end.x - start.x).powi(2)).sqrt();
        numerator / denominator
    }

    pub fn cross(&self, other: &Point) -> f32 {
        self.x * other.y - self.y * other.x
    }
}

#[cfg(test)]
mod tests_points {
    use super::Point;

    #[test]
    fn new() {
        let origin1 = Point::from((0.0f32, 0.0f32));
        let origin2 = Point::from([0.0f32, 0.0f32]);
        let origin3 = (0.0f32, 0.0f32).into();
        let origin4 = [0.0f32, 0.0f32].into();
        let origin5 = Point::new(1.0f32, 2.0f32);
        let origin6 = Point {
            x: 1.0f32,
            y: 2.0f32,
        };
        assert_eq!(origin1, origin2);
        assert_eq!(origin2, origin3);
        assert_eq!(origin3, origin4);
        assert_eq!(origin5, origin6);
        assert!(origin1.is_origin());
        assert!(origin2.is_origin());
        assert!(origin3.is_origin());
        assert!(origin4.is_origin());
        assert!(!origin5.is_origin());
        assert!(!origin6.is_origin());
    }

    #[test]
    fn into_tuple_array() {
        let point = Point::from((1.0, 2.0));
        let tuple: (f32, f32) = point.into();
        let array: [f32; 2] = point.into();
        assert_eq!(tuple, (1.0, 2.0));
        assert_eq!(array, [1.0, 2.0]);
    }
}
