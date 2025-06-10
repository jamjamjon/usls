use aksr::Builder;
use std::ops::{Add, Div, Mul, Sub};

use crate::{InstanceMeta, Style};

/// Represents a keypoint in a 2D space with optional metadata.
#[derive(Builder, Default, Clone)]
pub struct Keypoint {
    x: f32,
    y: f32,
    meta: InstanceMeta,
    style: Option<Style>,
}

// #[derive(Default, Builder, Clone)]
// pub struct Keypoints {
//     keypoints: Vec<Keypoint>,
//     skeletons: Option<Vec<(usize, usize)>>,
// }

impl std::fmt::Debug for Keypoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keypoint")
            .field("xy", &[self.x, self.y])
            .field("id", &self.meta.id())
            .field("name", &self.meta.name())
            .field("confidence", &self.meta.confidence())
            .finish()
    }
}

impl PartialEq for Keypoint {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl From<(f32, f32)> for Keypoint {
    fn from((x, y): (f32, f32)) -> Self {
        Self {
            x,
            y,
            ..Default::default()
        }
    }
}

impl From<[f32; 2]> for Keypoint {
    fn from([x, y]: [f32; 2]) -> Self {
        Self {
            x,
            y,
            ..Default::default()
        }
    }
}

impl From<Keypoint> for (f32, f32) {
    fn from(Keypoint { x, y, .. }: Keypoint) -> Self {
        (x, y)
    }
}

impl From<Keypoint> for [f32; 2] {
    fn from(Keypoint { x, y, .. }: Keypoint) -> Self {
        [x, y]
    }
}

impl Keypoint {
    impl_meta_methods!();

    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            ..Default::default()
        }
    }

    pub fn with_xy(mut self, x: f32, y: f32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    pub fn xy(&self) -> (f32, f32) {
        (self.x, self.y)
    }

    pub fn is_origin(&self) -> bool {
        self.x == 0.0_f32 && self.y == 0.0_f32
    }

    pub fn distance_from(&self, other: &Self) -> f32 {
        ((self.x - other.x).powf(2.0) + (self.y - other.y).powf(2.0)).sqrt()
    }

    pub fn distance_from_origin(&self) -> f32 {
        (self.x.powf(2.0) + self.y.powf(2.0)).sqrt()
    }

    pub fn sum(&self) -> f32 {
        self.x + self.y
    }

    pub fn rotate(&self, angle_rad: f32) -> Self {
        let sn = angle_rad.sin();
        let cs = angle_rad.cos();
        (cs * self.x - sn * self.y, sn * self.x + cs * self.y).into()
    }

    /// Calculates the perpendicular distance from the current keypoint (`self`)
    /// to a line segment defined by two other keypoints (`start` and `end`).
    pub fn perpendicular_distance(&self, start: &Self, end: &Self) -> f32 {
        let numerator = ((end.y - start.y) * self.x - (end.x - start.x) * self.y + end.x * start.y
            - end.y * start.x)
            .abs();
        let denominator = ((end.y - start.y).powi(2) + (end.x - start.x).powi(2)).sqrt();
        numerator / denominator
    }

    pub fn cross(&self, other: &Keypoint) -> f32 {
        self.x * other.y - self.y * other.x
    }
}

impl Add for Keypoint {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            ..Default::default()
        }
    }
}

impl Add<f32> for Keypoint {
    type Output = Self;

    fn add(self, other: f32) -> Self::Output {
        Self {
            x: self.x + other,
            y: self.y + other,
            ..Default::default()
        }
    }
}

impl Sub for Keypoint {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            ..Default::default()
        }
    }
}

impl Sub<f32> for Keypoint {
    type Output = Self;

    fn sub(self, other: f32) -> Self::Output {
        Self {
            x: self.x - other,
            y: self.y - other,
            ..Default::default()
        }
    }
}

impl Mul<f32> for Keypoint {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        Self {
            x: self.x * other,
            y: self.y * other,
            ..Default::default()
        }
    }
}

impl Mul for Keypoint {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            ..Default::default()
        }
    }
}

impl Div for Keypoint {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
            ..Default::default()
        }
    }
}

impl Div<f32> for Keypoint {
    type Output = Self;

    fn div(self, other: f32) -> Self::Output {
        Self {
            x: self.x / other,
            y: self.y / other,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests_keypoint {
    use super::Keypoint;

    #[test]
    fn new() {
        let kpt1 = Keypoint::from((0., 0.));
        let kpt2 = Keypoint::from([0., 0.]);
        let kpt3 = (0., 0.).into();
        let kpt4 = [0., 0.].into();
        let kpt5: Keypoint = [5.5, 6.6].into();
        let kpt6 = Keypoint::default().with_xy(5.5, 6.6);
        assert_eq!(kpt1, kpt2);
        assert_eq!(kpt2, kpt3);
        assert_eq!(kpt3, kpt4);
        assert_eq!(kpt6, kpt5);
        assert_eq!(kpt5.x(), 5.5);
        assert_eq!(kpt5.y(), 6.6);
    }

    #[test]
    fn into_tuple() {
        let kpt = Keypoint::from((1., 2.));
        let tuple: (f32, f32) = kpt.into();
        assert_eq!(tuple, (1., 2.));
    }

    #[test]
    fn into_array() {
        let kpt = Keypoint::from((1., 2.));
        let array: [f32; 2] = kpt.into();
        assert_eq!(array, [1., 2.]);
    }

    #[test]
    fn op_div() {
        assert_eq!(Keypoint::from([10., 10.]) / 2., Keypoint::from([5., 5.]));
        assert_eq!(
            Keypoint::from([10., 10.]) / Keypoint::from([2., 5.]),
            Keypoint::from([5., 2.])
        );
    }

    #[test]
    fn op_mul() {
        assert_eq!(Keypoint::from([10., 10.]) * 2., Keypoint::from([20., 20.]));
        assert_eq!(
            Keypoint::from([10., 10.]) * Keypoint::from([2., 5.]),
            Keypoint::from([20., 50.])
        );
    }

    #[test]
    fn op_add() {
        assert_eq!(Keypoint::from([10., 10.]) + 2., Keypoint::from([12., 12.]));
        assert_eq!(
            Keypoint::from([10., 10.]) + Keypoint::from([2., 5.]),
            Keypoint::from([12., 15.])
        );
    }

    #[test]
    fn op_minus() {
        assert_eq!(Keypoint::from([10., 10.]) - 2., Keypoint::from([8., 8.]));
        assert_eq!(
            Keypoint::from([10., 10.]) - Keypoint::from([2., 5.]),
            Keypoint::from([8., 5.])
        );
    }

    #[test]
    fn fns() {
        assert!(Keypoint::from([0., 0.]).is_origin());
        assert!(!Keypoint::from([0., 0.1]).is_origin());
        let kpt1 = Keypoint::from((0., 0.));
        let kpt2 = Keypoint::from((5., 0.));
        assert_eq!(kpt1.distance_from(&kpt2), 5.);
    }
}
