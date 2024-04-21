/// Bounding Box 2D
#[derive(Clone, PartialEq, PartialOrd)]
pub struct Bbox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    id: isize,
    confidence: f32,
    name: Option<String>,
    id_born: isize,
}

impl Default for Bbox {
    fn default() -> Self {
        Self {
            x: 0.,
            y: 0.,
            w: 0.,
            h: 0.,
            id: -1,
            confidence: 0.,
            name: None,
            id_born: -1,
        }
    }
}

impl std::fmt::Debug for Bbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bbox")
            .field("xyxy", &[self.x, self.y, self.xmax(), self.ymax()])
            .field("id", &self.id)
            .field("id_born", &self.id_born)
            .field("name", &self.name)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl From<(f32, f32, f32, f32)> for Bbox {
    fn from((x, y, w, h): (f32, f32, f32, f32)) -> Self {
        Self {
            x,
            y,
            w,
            h,
            ..Default::default()
        }
    }
}

impl From<[f32; 4]> for Bbox {
    fn from([x, y, w, h]: [f32; 4]) -> Self {
        Self {
            x,
            y,
            w,
            h,
            ..Default::default()
        }
    }
}

impl From<(f32, f32, f32, f32, isize, f32)> for Bbox {
    fn from((x, y, w, h, id, confidence): (f32, f32, f32, f32, isize, f32)) -> Self {
        Self {
            x,
            y,
            w,
            h,
            id,
            confidence,
            ..Default::default()
        }
    }
}

impl Bbox {
    pub fn with_xyxy(mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        self.x = x1;
        self.y = y1;
        self.w = x2 - x1;
        self.h = y2 - y1;
        self
    }

    pub fn with_xywh(mut self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.x = x;
        self.y = y;
        self.w = w;
        self.h = h;
        self
    }

    pub fn with_id(mut self, x: isize) -> Self {
        self.id = x;
        self
    }

    pub fn with_id_born(mut self, x: isize) -> Self {
        self.id_born = x;
        self
    }

    pub fn with_confidence(mut self, x: f32) -> Self {
        self.confidence = x;
        self
    }

    pub fn with_name(mut self, x: Option<String>) -> Self {
        self.name = x;
        self
    }

    pub fn width(&self) -> f32 {
        self.w
    }

    pub fn height(&self) -> f32 {
        self.h
    }

    pub fn xmin(&self) -> f32 {
        self.x
    }

    pub fn ymin(&self) -> f32 {
        self.y
    }

    pub fn xmax(&self) -> f32 {
        self.x + self.w
    }

    pub fn ymax(&self) -> f32 {
        self.y + self.h
    }

    pub fn cx(&self) -> f32 {
        self.x + self.w / 2.
    }

    pub fn cy(&self) -> f32 {
        self.y + self.h / 2.
    }

    pub fn id(&self) -> isize {
        self.id
    }

    pub fn id_born(&self) -> isize {
        self.id_born
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    pub fn area(&self) -> f32 {
        self.h * self.w
    }

    pub fn perimeter(&self) -> f32 {
        (self.h + self.w) * 2.0
    }

    pub fn is_squre(&self) -> bool {
        self.w == self.h
    }

    pub fn intersect(&self, other: &Bbox) -> f32 {
        let l = self.xmin().max(other.xmin());
        let r = (self.xmin() + self.width()).min(other.xmin() + other.width());
        let t = self.ymin().max(other.ymin());
        let b = (self.ymin() + self.height()).min(other.ymin() + other.height());
        (r - l).max(0.) * (b - t).max(0.)
    }

    pub fn union(&self, other: &Bbox) -> f32 {
        self.area() + other.area() - self.intersect(other)
    }

    pub fn iou(&self, other: &Bbox) -> f32 {
        self.intersect(other) / self.union(other)
    }

    pub fn contains(&self, other: &Bbox) -> bool {
        self.xmin() <= other.xmin()
            && self.xmax() >= other.xmax()
            && self.ymin() <= other.ymin()
            && self.ymax() >= other.ymax()
    }
}

#[cfg(test)]
mod tests_bbox {
    use super::Bbox;

    #[test]
    fn new() {
        let bbox1 = Bbox::from((0., 0., 5., 5.));
        let bbox2 = Bbox::from([0., 0., 5., 5.]);
        assert_eq!(bbox1, bbox2);

        let bbox1: Bbox = [0., 0., 5., 5.].into();
        let bbox2: Bbox = (0., 0., 5., 5.).into();
        assert_eq!(bbox1, bbox2);

        let bbox1: Bbox = (1., 1., 5., 5., 99, 0.99).into();
        let bbox2 = Bbox::default()
            .with_xyxy(1., 1., 6., 6.)
            .with_id(99)
            .with_confidence(0.99);
        assert_eq!(bbox1, bbox2);

        let bbox1: Bbox = (1., 1., 5., 5., 1, 1.).into();
        let bbox2 = Bbox::default()
            .with_xywh(1., 1., 5., 5.)
            .with_id(1)
            .with_confidence(1.);
        assert_eq!(bbox1, bbox2);
    }

    #[test]
    fn funcs() {
        let bbox1 = Bbox::default().with_xyxy(0., 0., 5., 5.);
        let bbox2 = Bbox::default().with_xyxy(1., 1., 6., 6.);
        assert_eq!(bbox1.intersect(&bbox2), 16.);
        assert_eq!(bbox1.area(), 25.);
        assert_eq!(bbox2.area(), 25.);
        assert_eq!(bbox2.perimeter(), 20.);
        assert!(bbox2.is_squre());
        assert_eq!(bbox1.union(&bbox2), 34.);

        let bbox3 = Bbox::default().with_xyxy(2., 2., 5., 5.);
        assert!(!bbox1.contains(&bbox2));
        assert!(bbox1.contains(&bbox3));
        assert!(bbox2.contains(&bbox3));
    }
}
