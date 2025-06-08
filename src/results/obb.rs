use aksr::Builder;

use crate::{Hbb, InstanceMeta, Keypoint, Polygon, Style};

/// Oriented bounding box with four vertices and metadata.
#[derive(Builder, Default, Clone, PartialEq)]
pub struct Obb {
    vertices: [[f32; 2]; 4], // ordered
    meta: InstanceMeta,
    style: Option<Style>,
    keypoints: Option<Vec<Keypoint>>,
}

impl std::fmt::Debug for Obb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Obb")
            .field("vertices", &self.vertices)
            .field("id", &self.meta.id())
            .field("name", &self.meta.name())
            .field("confidence", &self.meta.confidence())
            .finish()
    }
}

impl From<[[f32; 2]; 4]> for Obb {
    fn from(vertices: [[f32; 2]; 4]) -> Self {
        Self {
            vertices,
            ..Default::default()
        }
    }
}

impl From<Vec<[f32; 2]>> for Obb {
    fn from(vertices: Vec<[f32; 2]>) -> Self {
        // Self::from(vertices[..4])
        let vertices = [vertices[0], vertices[1], vertices[2], vertices[3]];
        Self {
            vertices,
            ..Default::default()
        }
    }
}

impl From<Obb> for [[f32; 2]; 4] {
    fn from(obb: Obb) -> Self {
        obb.vertices
    }
}

impl Obb {
    impl_meta_methods!();
    /// Build from (cx, cy, width, height, degrees)
    pub fn from_cxcywhd(cx: f32, cy: f32, w: f32, h: f32, d: f32) -> Self {
        Self::from_cxcywhr(cx, cy, w, h, d.to_radians())
    }

    /// Build from (cx, cy, width, height, radians)
    pub fn from_cxcywhr(cx: f32, cy: f32, w: f32, h: f32, r: f32) -> Self {
        // [[cos -sin], [sin cos]]
        let m = [
            [r.cos() * 0.5 * w, -r.sin() * 0.5 * h],
            [r.sin() * 0.5 * w, r.cos() * 0.5 * h],
        ];
        let c = [cx, cy];
        let a_ = [m[0][0] + m[0][1], m[1][0] + m[1][1]];
        let b_ = [m[0][0] - m[0][1], m[1][0] - m[1][1]];

        let v1 = [c[0] + a_[0], c[1] + a_[1]];
        let v2 = [c[0] + b_[0], c[1] + b_[1]];
        let v3 = [c[0] * 2. - v1[0], c[1] * 2. - v1[1]];
        let v4 = [c[0] * 2. - v2[0], c[1] * 2. - v2[1]];

        Self {
            vertices: [v1, v2, v3, v4],
            ..Default::default()
        }
    }

    pub fn top(&self) -> [f32; 2] {
        let mut top = self.vertices[0];
        for v in &self.vertices {
            if v[1] < top[1] {
                top = *v;
            }
        }
        top
    }

    pub fn bottom(&self) -> [f32; 2] {
        let mut bottom = self.vertices[0];
        for v in &self.vertices {
            if v[1] > bottom[1] {
                bottom = *v;
            }
        }
        bottom
    }
    pub fn left(&self) -> [f32; 2] {
        let mut left = self.vertices[0];
        for v in &self.vertices {
            if v[0] < left[0] {
                left = *v;
            }
        }
        left
    }
    pub fn right(&self) -> [f32; 2] {
        let mut right = self.vertices[0];
        for v in &self.vertices {
            if v[0] > right[0] {
                right = *v;
            }
        }
        right
    }

    pub fn to_polygon(&self) -> Polygon {
        Polygon::from_xys(&self.vertices)
    }

    pub fn area(&self) -> f32 {
        self.to_polygon().area() as f32
    }

    pub fn intersect(&self, other: &Self) -> f32 {
        let pa = self.to_polygon();
        let pb = other.to_polygon();
        pa.intersect(&pb)
    }

    pub fn union(&self, other: &Self) -> f32 {
        let pa = self.to_polygon();
        let pb = other.to_polygon();
        pa.union(&pb)
    }

    pub fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }

    pub fn hbb() -> Hbb {
        todo!()
    }
}

#[cfg(test)]
mod tests_mbr {
    // use crate::Nms;
    use super::Obb;

    #[test]
    fn iou1() {
        let a = Obb::from([[0., 0.], [0., 2.], [2., 2.], [2., 0.]]);
        let b = Obb::from_cxcywhd(1., 1., 2., 2., 0.);

        assert_eq!(a.iou(&b), 1.0);
    }

    #[test]
    fn iou2() {
        let a = Obb::from([[2.5, 5.], [-2.5, 5.], [-2.5, -5.], [2.5, -5.]]);
        let b = Obb::from_cxcywhd(0., 0., 10., 5., 90.);
        assert_eq!(a.iou(&b), 1.0);
    }

    #[test]
    fn intersect() {
        let a = Obb::from_cxcywhr(0., 0., 2.828427, 2.828427, 45.);
        let b = Obb::from_cxcywhr(1., 1., 2., 2., 0.);
        assert_eq!(a.intersect(&b).round(), 2.);
    }

    #[test]
    fn union() {
        let a = Obb::from([[2., 0.], [0., 2.], [-2., 0.], [0., -2.]]);
        let b = Obb::from([[0., 0.], [2., 0.], [2., 2.], [0., 2.]]);
        assert_eq!(a.union(&b), 10.);
    }

    #[test]
    fn iou() {
        let a = Obb::from([[2., 0.], [0., 2.], [-2., 0.], [0., -2.]]);
        let b = Obb::from([[0., 0.], [2., 0.], [2., 2.], [0., 2.]]);
        assert_eq!(a.iou(&b), 0.2);
    }
}
