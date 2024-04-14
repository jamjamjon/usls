use crate::Point;

#[derive(PartialEq, Clone, Default)]
pub struct Keypoint {
    pub point: Point,
    confidence: f32,
}

impl std::fmt::Debug for Keypoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keypoint")
            .field("x", &self.point.x)
            .field("y", &self.point.y)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl Keypoint {
    pub fn new(point: Point, confidence: f32) -> Self {
        Self { point, confidence }
    }

    pub fn x(&self) -> f32 {
        self.point.x
    }

    pub fn y(&self) -> f32 {
        self.point.y
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }
}
