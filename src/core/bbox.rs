use crate::Rect;

#[derive(Clone, PartialEq, Default)]
pub struct Bbox {
    rect: Rect,
    id: usize,
    confidence: f32,
    name: Option<String>,
}

impl std::fmt::Debug for Bbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bbox")
            .field("xmin", &self.rect.xmin())
            .field("ymin", &self.rect.ymin())
            .field("xmax", &self.rect.xmax())
            .field("ymax", &self.rect.ymax())
            .field("id", &self.id)
            .field("name", &self.name)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl Bbox {
    pub fn new(rect: Rect, id: usize, confidence: f32, name: Option<String>) -> Self {
        Self {
            rect,
            id,
            confidence,
            name,
        }
    }

    pub fn width(&self) -> f32 {
        self.rect.width()
    }

    pub fn height(&self) -> f32 {
        self.rect.height()
    }

    pub fn xmin(&self) -> f32 {
        self.rect.xmin()
    }

    pub fn ymin(&self) -> f32 {
        self.rect.ymin()
    }

    pub fn xmax(&self) -> f32 {
        self.rect.xmax()
    }

    pub fn ymax(&self) -> f32 {
        self.rect.ymax()
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    pub fn area(&self) -> f32 {
        self.rect.area()
    }

    pub fn iou(&self, other: &Bbox) -> f32 {
        self.rect.intersect(&other.rect) / self.rect.union(&other.rect)
    }
}
