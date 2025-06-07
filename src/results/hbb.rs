use aksr::Builder;

use crate::{InstanceMeta, Keypoint, Style};

/// Horizontal bounding box with position, size, and metadata.
#[derive(Builder, Clone, Default)]
pub struct Hbb {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    meta: InstanceMeta,
    style: Option<Style>,
    keypoints: Option<Vec<Keypoint>>,
}

impl std::fmt::Debug for Hbb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hbb")
            .field("xyxy", &[self.x, self.y, self.xmax(), self.ymax()])
            .field("id", &self.meta.id())
            .field("name", &self.meta.name())
            .field("confidence", &self.meta.confidence())
            .finish()
    }
}

impl PartialEq for Hbb {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.w == other.w && self.h == other.h
    }
}

impl From<(f32, f32, f32, f32)> for Hbb {
    /// Creates a `Hbb` from a tuple of `(x, y, w, h)`.
    ///
    /// # Arguments
    ///
    /// * `(x, y, w, h)` - A tuple representing the bounding box's position and size.
    ///
    /// # Returns
    ///
    /// A `Hbb` with the specified position and size.
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

impl From<[f32; 4]> for Hbb {
    /// Creates a `Hbb` from an array of `[x, y, w, h]`.
    ///
    /// # Arguments
    ///
    /// * `[x, y, w, h]` - An array representing the bounding box's position and size.
    ///
    /// # Returns
    ///
    /// A `Hbb` with the specified position and size.
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

impl From<Hbb> for (f32, f32, f32, f32) {
    fn from(Hbb { x, y, w, h, .. }: Hbb) -> Self {
        (x, y, w, h)
    }
}

impl From<Hbb> for [f32; 4] {
    fn from(Hbb { x, y, w, h, .. }: Hbb) -> Self {
        [x, y, w, h]
    }
}

impl Hbb {
    impl_meta_methods!();

    pub fn from_xywh(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            x,
            y,
            w,
            h,
            ..Default::default()
        }
    }

    pub fn from_xyxy(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            x: x1,
            y: y1,
            w: x2 - x1,
            h: y2 - y1,
            ..Default::default()
        }
    }

    pub fn from_cxcywh(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self {
            x: cx - w / 2.0,
            y: cy - h / 2.0,
            w,
            h,
            ..Default::default()
        }
    }

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

    pub fn with_cxcywh(mut self, cx: f32, cy: f32, w: f32, h: f32) -> Self {
        self.x = cx - w / 2.0;
        self.y = cy - h / 2.0;
        self.w = w;
        self.h = h;
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

    pub fn xyxy(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.x + self.w, self.y + self.h)
    }

    pub fn xywh(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    pub fn cxywh(&self) -> (f32, f32, f32, f32) {
        (self.cx(), self.cy(), self.w, self.h)
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

    pub fn intersect(&self, other: &Hbb) -> f32 {
        let l = self.xmin().max(other.xmin());
        let r = (self.xmin() + self.width()).min(other.xmin() + other.width());
        let t = self.ymin().max(other.ymin());
        let b = (self.ymin() + self.height()).min(other.ymin() + other.height());
        (r - l).max(0.) * (b - t).max(0.)
    }

    pub fn union(&self, other: &Hbb) -> f32 {
        self.area() + other.area() - self.intersect(other)
    }

    pub fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }

    pub fn contains(&self, other: &Hbb) -> bool {
        self.xmin() <= other.xmin()
            && self.xmax() >= other.xmax()
            && self.ymin() <= other.ymin()
            && self.ymax() >= other.ymax()
    }

    pub fn to_json() {
        // Display?
        todo!()
    }
}

#[cfg(test)]
mod tests_bbox {
    use super::Hbb;

    #[test]
    fn new() {
        let bbox1 = Hbb::from((0., 0., 5., 5.));
        let bbox2: Hbb = [0., 0., 5., 5.].into();
        assert_eq!(bbox1, bbox2);
    }

    #[test]
    fn funcs() {
        let bbox1 = Hbb::from_xyxy(0., 0., 5., 5.);
        let bbox2 = Hbb::from_xyxy(1., 1., 6., 6.);
        assert_eq!(bbox1.intersect(&bbox2), 16.);
        assert_eq!(bbox1.area(), 25.);
        assert_eq!(bbox2.area(), 25.);
        assert_eq!(bbox2.perimeter(), 20.);
        assert!(bbox2.is_squre());
        assert_eq!(bbox1.union(&bbox2), 34.);

        let bbox3 = Hbb::from_xyxy(2., 2., 5., 5.);
        assert!(!bbox1.contains(&bbox2));
        assert!(bbox1.contains(&bbox3));
        assert!(bbox2.contains(&bbox3));
    }
}
