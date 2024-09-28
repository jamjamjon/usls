use crate::Nms;

/// Bounding Box 2D.
///
/// This struct represents a 2D bounding box with properties such as position, size,
/// class ID, confidence score, optional name, and an ID representing the born state.
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

impl Nms for Bbox {
    /// Returns the confidence score of the bounding box.
    fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Computes the intersection over union (IoU) between this bounding box and another.
    fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }
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
            .field("class_id", &self.id)
            .field("name", &self.name)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl From<(f32, f32, f32, f32)> for Bbox {
    /// Creates a `Bbox` from a tuple of `(x, y, w, h)`.
    ///
    /// # Arguments
    ///
    /// * `(x, y, w, h)` - A tuple representing the bounding box's position and size.
    ///
    /// # Returns
    ///
    /// A `Bbox` with the specified position and size.
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
    /// Creates a `Bbox` from an array of `[x, y, w, h]`.
    ///
    /// # Arguments
    ///
    /// * `[x, y, w, h]` - An array representing the bounding box's position and size.
    ///
    /// # Returns
    ///
    /// A `Bbox` with the specified position and size.
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
    /// Creates a `Bbox` from a tuple of `(x, y, w, h, id, confidence)`.
    ///
    /// # Arguments
    ///
    /// * `(x, y, w, h, id, confidence)` - A tuple representing the bounding box's position, size, class ID, and confidence score.
    ///
    /// # Returns
    ///
    /// A `Bbox` with the specified position, size, class ID, and confidence score.
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
    /// Sets the bounding box's coordinates using `(x1, y1, x2, y2)` and calculates width and height.
    ///
    /// # Arguments
    ///
    /// * `x1` - The x-coordinate of the top-left corner.
    /// * `y1` - The y-coordinate of the top-left corner.
    /// * `x2` - The x-coordinate of the bottom-right corner.
    /// * `y2` - The y-coordinate of the bottom-right corner.
    ///
    /// # Returns
    ///
    /// A `Bbox` instance with updated coordinates and dimensions.
    pub fn with_xyxy(mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        self.x = x1;
        self.y = y1;
        self.w = x2 - x1;
        self.h = y2 - y1;
        self
    }

    /// Sets the bounding box's coordinates and dimensions using `(x, y, w, h)`.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the top-left corner.
    /// * `y` - The y-coordinate of the top-left corner.
    /// * `w` - The width of the bounding box.
    /// * `h` - The height of the bounding box.
    ///
    /// # Returns
    ///
    /// A `Bbox` instance with updated coordinates and dimensions.
    pub fn with_xywh(mut self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.x = x;
        self.y = y;
        self.w = w;
        self.h = h;
        self
    }

    /// Sets the class ID of the bounding box.
    ///
    /// # Arguments
    ///
    /// * `x` - The class ID to be set.
    ///
    /// # Returns
    ///
    /// A `Bbox` instance with updated class ID.
    pub fn with_id(mut self, x: isize) -> Self {
        self.id = x;
        self
    }

    /// Sets the ID representing the born state of the bounding box.
    ///
    /// # Arguments
    ///
    /// * `x` - The ID to be set.
    ///
    /// # Returns
    ///
    /// A `Bbox` instance with updated born state ID.
    pub fn with_id_born(mut self, x: isize) -> Self {
        self.id_born = x;
        self
    }

    /// Sets the confidence score of the bounding box.
    ///
    /// # Arguments
    ///
    /// * `x` - The confidence score to be set.
    ///
    /// # Returns
    ///
    /// A `Bbox` instance with updated confidence score.
    pub fn with_confidence(mut self, x: f32) -> Self {
        self.confidence = x;
        self
    }

    /// Sets the optional name of the bounding box.
    ///
    /// # Arguments
    ///
    /// * `x` - The name to be set.
    ///
    /// # Returns
    ///
    /// A `Bbox` instance with updated name.
    pub fn with_name(mut self, x: &str) -> Self {
        self.name = Some(x.to_string());
        self
    }

    /// Returns the width of the bounding box.
    pub fn width(&self) -> f32 {
        self.w
    }

    /// Returns the height of the bounding box.
    pub fn height(&self) -> f32 {
        self.h
    }

    /// Returns the minimum x-coordinate of the bounding box.
    pub fn xmin(&self) -> f32 {
        self.x
    }

    /// The minimum y-coordinate of the bounding box.
    pub fn ymin(&self) -> f32 {
        self.y
    }

    /// Returns the maximum x-coordinate of the bounding box.
    pub fn xmax(&self) -> f32 {
        self.x + self.w
    }

    /// The maximum x-coordinate of the bounding box.
    pub fn ymax(&self) -> f32 {
        self.y + self.h
    }

    /// Returns the center x-coordinate of the bounding box.
    pub fn cx(&self) -> f32 {
        self.x + self.w / 2.
    }

    /// Returns the center y-coordinate of the bounding box.
    pub fn cy(&self) -> f32 {
        self.y + self.h / 2.
    }

    /// Returns the bounding box coordinates as `(x1, y1, x2, y2)`.
    pub fn xyxy(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.x + self.w, self.y + self.h)
    }

    /// Returns the bounding box coordinates and size as `(x, y, w, h)`.
    pub fn xywh(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    /// Returns the center coordinates and size of the bounding box as `(cx, cy, w, h)`.
    pub fn cxywh(&self) -> (f32, f32, f32, f32) {
        (self.cx(), self.cy(), self.w, self.h)
    }

    /// Returns the class ID of the bounding box.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the born state ID of the bounding box.
    pub fn id_born(&self) -> isize {
        self.id_born
    }

    /// Returns the optional name associated with the bounding box, if any.
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    // /// Returns the confidence score of the bounding box.
    // pub fn confidence(&self) -> f32 {
    //     self.confidence
    // }

    /// A label string representing the bounding box, optionally including name and confidence score.
    pub fn label(&self, with_name: bool, with_conf: bool, decimal_places: usize) -> String {
        let mut label = String::new();
        if with_name {
            label.push_str(
                &self
                    .name
                    .as_ref()
                    .unwrap_or(&self.id.to_string())
                    .to_string(),
            );
        }
        if with_conf {
            if with_name {
                label.push_str(&format!(": {:.decimal_places$}", self.confidence));
            } else {
                label.push_str(&format!("{:.decimal_places$}", self.confidence));
            }
        }
        label
    }

    /// Computes the area of the bounding box.
    pub fn area(&self) -> f32 {
        self.h * self.w
    }

    /// Computes the perimeter of the bounding box.
    pub fn perimeter(&self) -> f32 {
        (self.h + self.w) * 2.0
    }

    /// Checks if the bounding box is square (i.e., width equals height).
    pub fn is_squre(&self) -> bool {
        self.w == self.h
    }

    /// Computes the intersection area between this bounding box and another.
    pub fn intersect(&self, other: &Bbox) -> f32 {
        let l = self.xmin().max(other.xmin());
        let r = (self.xmin() + self.width()).min(other.xmin() + other.width());
        let t = self.ymin().max(other.ymin());
        let b = (self.ymin() + self.height()).min(other.ymin() + other.height());
        (r - l).max(0.) * (b - t).max(0.)
    }

    /// Computes the union area between this bounding box and another.
    pub fn union(&self, other: &Bbox) -> f32 {
        self.area() + other.area() - self.intersect(other)
    }

    // /// Computes the intersection over union (IoU) between this bounding box and another.
    // pub fn iou(&self, other: &Bbox) -> f32 {
    //     self.intersect(other) / self.union(other)
    // }

    /// Checks if this bounding box completely contains another bounding box `other`.
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
