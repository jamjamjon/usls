use crate::Point;

#[derive(PartialEq, Clone)]
pub struct Keypoint {
    pub point: Point,
    confidence: f32,
    id: isize,
    name: Option<String>,
}

impl Default for Keypoint {
    fn default() -> Self {
        Self {
            id: -1,
            confidence: 0.0,
            point: Point::default(),
            name: None,
        }
    }
}

impl std::fmt::Debug for Keypoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keypoint")
            .field("x", &self.point.x)
            .field("y", &self.point.y)
            .field("confidence", &self.confidence)
            .field("id", &self.id)
            .field("name", &self.name)
            .finish()
    }
}

impl Keypoint {
    pub fn new(point: Point, confidence: f32, id: isize, name: Option<String>) -> Self {
        Self {
            point,
            confidence,
            id,
            name,
        }
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

    pub fn id(&self) -> isize {
        self.id
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
}
