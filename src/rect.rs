use crate::Point;

#[derive(Default, PartialOrd, PartialEq, Clone, Copy)]
pub struct Rect {
    top_left: Point,
    bottom_right: Point,
}

impl std::fmt::Debug for Rect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rectangle")
            .field("xmin", &self.xmin())
            .field("ymin", &self.ymin())
            .field("xmax", &self.xmax())
            .field("ymax", &self.ymax())
            .finish()
    }
}

impl<P: Into<Point>> From<(P, P)> for Rect {
    fn from((top_left, bottom_right): (P, P)) -> Self {
        Self {
            top_left: top_left.into(),
            bottom_right: bottom_right.into(),
        }
    }
}

impl<P: Into<Point>> From<[P; 2]> for Rect {
    fn from([top_left, bottom_right]: [P; 2]) -> Self {
        Self {
            top_left: top_left.into(),
            bottom_right: bottom_right.into(),
        }
    }
}

impl Rect {
    pub fn new(top_left: Point, bottom_right: Point) -> Self {
        Self {
            top_left,
            bottom_right,
        }
    }

    pub fn from_xywh(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            top_left: Point::new(x, y),
            bottom_right: Point::new(x + w, y + h),
        }
    }

    pub fn from_xyxy(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            top_left: Point::new(x1, y1),
            bottom_right: Point::new(x2, y2),
        }
    }

    pub fn from_cxywh(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self {
            top_left: Point::new(cx - w / 2.0, cy - h / 2.0),
            bottom_right: Point::new(cx + w / 2.0, cy + h / 2.0),
        }
    }

    pub fn width(&self) -> f32 {
        (self.bottom_right - self.top_left).x
    }

    pub fn height(&self) -> f32 {
        (self.bottom_right - self.top_left).y
    }

    pub fn xmin(&self) -> f32 {
        self.top_left.x
    }

    pub fn ymin(&self) -> f32 {
        self.top_left.y
    }

    pub fn xmax(&self) -> f32 {
        self.bottom_right.x
    }

    pub fn ymax(&self) -> f32 {
        self.bottom_right.y
    }

    pub fn cx(&self) -> f32 {
        self.bottom_right.x - self.top_left.x
    }

    pub fn cy(&self) -> f32 {
        self.bottom_right.y - self.top_left.y
    }

    pub fn tl(&self) -> Point {
        self.top_left
    }

    pub fn br(&self) -> Point {
        self.bottom_right
    }

    pub fn tr(&self) -> Point {
        Point::new(self.bottom_right.x, self.top_left.y)
    }

    pub fn bl(&self) -> Point {
        Point::new(self.top_left.x, self.bottom_right.y)
    }

    pub fn center(&self) -> Point {
        (self.bottom_right + self.top_left) / 2.0
    }

    pub fn area(&self) -> f32 {
        self.height() * self.width()
    }

    pub fn perimeter(&self) -> f32 {
        (self.height() + self.width()) * 2.0
    }

    pub fn is_empty(&self) -> bool {
        self.area() == 0.0
    }

    pub fn is_squre(&self) -> bool {
        self.width() == self.height()
    }

    pub fn intersect(&self, other: &Rect) -> f32 {
        let l = self.xmin().max(other.xmin());
        let r = (self.xmin() + self.width()).min(other.xmin() + other.width());
        let t = self.ymin().max(other.ymin());
        let b = (self.ymin() + self.height()).min(other.ymin() + other.height());
        (r - l).max(0.) * (b - t).max(0.)
    }

    pub fn union(&self, other: &Rect) -> f32 {
        self.area() + other.area() - self.intersect(other)
    }

    pub fn iou(&self, other: &Rect) -> f32 {
        self.intersect(other) / self.union(other)
    }

    pub fn contains(&self, other: &Rect) -> bool {
        self.xmin() <= other.xmin()
            && self.xmax() >= other.xmax()
            && self.ymin() <= other.ymin()
            && self.ymax() >= other.ymax()
    }

    pub fn expand(&mut self, x: f32, y: f32, max_x: f32, max_y: f32) -> Self {
        Self::from_xyxy(
            (self.xmin() - x).max(0.0f32).min(max_x),
            (self.ymin() - y).max(0.0f32).min(max_y),
            (self.xmax() + x).max(0.0f32).min(max_x),
            (self.ymax() + y).max(0.0f32).min(max_y),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Rect;
    use crate::Point;

    #[test]
    fn new() {
        let rect1 = Rect {
            top_left: Point {
                x: 0.0f32,
                y: 0.0f32,
            },
            bottom_right: Point {
                x: 5.0f32,
                y: 5.0f32,
            },
        };
        let rect2 = Rect {
            top_left: (0.0f32, 0.0f32).into(),
            bottom_right: [5.0f32, 5.0f32].into(),
        };
        let rect3 = Rect::new([0.0, 0.0].into(), [5.0, 5.0].into());
        let rect4: Rect = ((0.0, 0.0), (5.0, 5.0)).into();
        let rect5: Rect = [(0.0, 0.0), (5.0, 5.0)].into();
        let rect6: Rect = ([0.0, 0.0], [5.0, 5.0]).into();
        let rect7: Rect = Rect::from(([0.0, 0.0], [5.0, 5.0]));
        let rect8: Rect = Rect::from([[0.0, 0.0], [5.0, 5.0]]);
        let rect9: Rect = Rect::from([(0.0, 0.0), (5.0, 5.0)]);
        let rect10: Rect = Rect::from_xyxy(0.0, 0.0, 5.0, 5.0);
        let rect11: Rect = Rect::from_xywh(0.0, 0.0, 5.0, 5.0);

        assert_eq!(rect1, rect2);
        assert_eq!(rect3, rect4);
        assert_eq!(rect5, rect6);
        assert_eq!(rect7, rect8);
        assert_eq!(rect9, rect8);
        assert_eq!(rect10, rect11);
    }
}
