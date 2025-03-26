use aksr::Builder;
use geo::{coord, line_string, Area, BooleanOps, Coord, Distance, Euclidean, LineString, Polygon};

use crate::Nms;

/// Minimum Bounding Rectangle.
#[derive(Builder, Clone, PartialEq)]
pub struct Mbr {
    ls: LineString,
    id: isize,
    confidence: f32,
    name: Option<String>,
}

impl Nms for Mbr {
    /// Returns the confidence score of the bounding box.
    fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Computes the intersection over union (IoU) between this bounding box and another.
    fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }
}

impl Default for Mbr {
    fn default() -> Self {
        Self {
            ls: line_string![],
            id: -1,
            confidence: 0.,
            name: None,
        }
    }
}

impl std::fmt::Debug for Mbr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mbr")
            .field("vertices", &self.ls)
            .field("id", &self.id)
            .field("name", &self.name)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl Mbr {
    /// Build from (cx, cy, width, height, degrees)
    pub fn from_cxcywhd(cx: f64, cy: f64, w: f64, h: f64, d: f64) -> Self {
        Self::from_cxcywhr(cx, cy, w, h, d.to_radians())
    }

    /// Build from (cx, cy, width, height, radians)
    pub fn from_cxcywhr(cx: f64, cy: f64, w: f64, h: f64, r: f64) -> Self {
        // [[cos -sin], [sin cos]]
        let m = [
            [r.cos() * 0.5 * w, -r.sin() * 0.5 * h],
            [r.sin() * 0.5 * w, r.cos() * 0.5 * h],
        ];
        let c = coord! {
            x: cx,
            y: cy,
        };

        let a_ = coord! {
            x: m[0][0] + m[0][1],
            y: m[1][0] + m[1][1],
        };

        let b_ = coord! {
            x: m[0][0] - m[0][1],
            y: m[1][0] - m[1][1],
        };

        let v1 = c + a_;
        let v2 = c + b_;
        let v3 = c * 2. - v1;
        let v4 = c * 2. - v2;

        Self {
            ls: vec![v1, v2, v3, v4].into(),
            ..Default::default()
        }
    }

    pub fn from_line_string(ls: LineString) -> Self {
        Self {
            ls,
            ..Default::default()
        }
    }

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

    pub fn vertices(&self) -> Vec<Coord> {
        self.ls.0.clone()
    }

    pub fn top(&self) -> &Coord {
        self.ls
            .0
            .iter()
            .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap())
            .unwrap()
    }

    pub fn xmin(&self) -> f32 {
        self.ls
            .0
            .iter()
            .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap())
            .unwrap()
            .x as f32
    }

    pub fn ymin(&self) -> f32 {
        self.ls
            .0
            .iter()
            .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap())
            .unwrap()
            .y as f32
    }

    pub fn xmax(&self) -> f32 {
        self.ls
            .0
            .iter()
            .max_by(|a, b| a.x.partial_cmp(&b.x).unwrap())
            .unwrap()
            .x as f32
    }

    pub fn ymax(&self) -> f32 {
        self.ls
            .0
            .iter()
            .max_by(|a, b| a.y.partial_cmp(&b.y).unwrap())
            .unwrap()
            .y as f32
    }

    pub fn distance_min_max(&self) -> (f32, f32) {
        let ls = self.vertices();
        // "Please use the `Euclidean.distance` method from the `Distance` trait instead"
        let min = Euclidean.distance(ls[0], ls[1]);
        let max = Euclidean.distance(ls[1], ls[2]);
        if min < max {
            (min as f32, max as f32)
        } else {
            (max as f32, min as f32)
        }
    }

    pub fn intersect(&self, other: &Mbr) -> f32 {
        let p1 = Polygon::new(self.ls.clone(), vec![]);
        let p2 = Polygon::new(other.ls.clone(), vec![]);
        p1.intersection(&p2).unsigned_area() as f32
    }

    pub fn union(&self, other: &Mbr) -> f32 {
        let p1 = Polygon::new(self.ls.clone(), vec![]);
        let p2 = Polygon::new(other.ls.clone(), vec![]);
        p1.union(&p2).unsigned_area() as f32
    }
}

#[cfg(test)]
mod tests_mbr {
    use super::Mbr;
    use crate::Nms;
    use geo::{coord, line_string};

    #[test]
    fn from_cxcywhd1() {
        let mbr1 = Mbr::from_line_string(line_string![
         (x: 0., y: 0.),
         (x: 0., y: 2.),
         (x: 2., y: 2.),
         (x: 2., y: 0.),
        ]);
        let mbr2 = Mbr::from_cxcywhd(1., 1., 2., 2., 0.);
        assert_eq!(mbr1.iou(&mbr2), 1.0);
    }

    #[test]
    fn from_cxcywhd2() {
        let mbr1 = Mbr::from_line_string(line_string![
         (x: 2.5, y: 5.),
         (x: -2.5, y: 5.),
         (x: -2.5, y: -5.),
         (x: 2.5, y: -5.),
        ]);
        let mbr2 = Mbr::from_cxcywhd(0., 0., 10., 5., 90.);
        assert_eq!(mbr1.iou(&mbr2), 1.0);
    }

    #[test]
    fn from_cxcywhr1() {
        let mbr1 = Mbr::from_line_string(line_string![
         (x: 0., y: 0.),
         (x: 0., y: 2.),
         (x: 2., y: 2.),
         (x: 2., y: 0.),
        ]);
        let mbr2 = Mbr::from_cxcywhr(1., 1., 2., 2., 0.);
        assert_eq!(mbr1.iou(&mbr2), 1.0);
    }

    #[test]
    fn from_cxcywhr2() {
        let mbr1 = Mbr::from_line_string(line_string![
         (x: 2.5, y: 5.),
         (x: -2.5, y: 5.),
         (x: -2.5, y: -5.),
         (x: 2.5, y: -5.),
        ]);
        let mbr2 = Mbr::from_cxcywhr(0., 0., 10., 5., std::f64::consts::PI / 2.);
        assert_eq!(mbr1.iou(&mbr2), 1.0);
    }

    #[test]
    fn distance_min_max() {
        let mbr = Mbr::from_line_string(line_string![
         (x: 2., y: 0.),
         (x: 0., y: 2.),
         (x: -2., y: 0.),
         (x: 0., y: -2.),
        ]);
        assert_eq!(mbr.distance_min_max(), (2.828427, 2.828427));
    }

    #[test]
    fn vertices() {
        let mbr = Mbr::from_line_string(line_string![
         (x: 2., y: 0.),
         (x: 0., y: 2.),
         (x: -2., y: 0.),
         (x: 0., y: -2.),
        ]);
        assert_eq!(
            mbr.vertices(),
            vec![
                coord! {
                    x: 2.,
                    y: 0.,
                },
                coord! {
                    x: 0.,
                    y: 2.,
                },
                coord! {
                    x: -2.,
                    y: 0.,
                },
                coord! {
                    x: 0.,
                    y: -2.,
                },
            ]
        );
    }

    #[test]
    fn intersect() {
        let mbr1 = Mbr::from_cxcywhr(0., 0., 2.828427, 2.828427, 45.);
        let mbr2 = Mbr::from_cxcywhr(1., 1., 2., 2., 0.);
        assert_eq!(mbr1.intersect(&mbr2).round(), 2.);
    }

    #[test]
    fn union() {
        let mbr1 = Mbr::from_line_string(line_string![
         (x: 2., y: 0.),
         (x: 0., y: 2.),
         (x: -2., y: 0.),
         (x: 0., y: -2.),
        ]);
        let mbr2 = Mbr::from_line_string(line_string![
         (x: 0., y: 0.),
         (x: 2., y: 0.),
         (x: 2., y: 2.),
         (x: 0., y: 2.),
        ]);
        assert_eq!(mbr1.union(&mbr2), 10.);
    }

    #[test]
    fn iou() {
        let mbr1 = Mbr::from_line_string(line_string![
         (x: 2., y: 0.),
         (x: 0., y: 2.),
         (x: -2., y: 0.),
         (x: 0., y: -2.),
        ]);
        let mbr2 = Mbr::from_line_string(line_string![
         (x: 0., y: 0.),
         (x: 2., y: 0.),
         (x: 2., y: 2.),
         (x: 0., y: 2.),
        ]);
        assert_eq!(mbr1.iou(&mbr2), 0.2);
    }
}
