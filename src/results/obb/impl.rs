use aksr::Builder;

use crate::{InstanceMeta, Keypoint, ObbStyle, Polygon};

/// Oriented bounding box with four vertices and metadata.
#[derive(Builder, Default, Clone, PartialEq)]
pub struct Obb {
    vertices: [[f32; 2]; 4], // CCW ordered
    meta: InstanceMeta,
    style: Option<ObbStyle>,
    keypoints: Option<Vec<Keypoint>>,
}

impl std::fmt::Debug for Obb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Obb");
        f.field("vertices", &self.vertices);
        if let Some(id) = &self.meta.id() {
            f.field("id", id);
        }
        if let Some(name) = &self.meta.name() {
            f.field("name", name);
        }
        if let Some(confidence) = &self.meta.confidence() {
            f.field("confidence", confidence);
        }
        if let Some(track_id) = &self.meta.track_id() {
            f.field("track_id", track_id);
        }
        if let Some(keypoints) = &self.keypoints {
            f.field("keypoints", keypoints);
        }
        f.finish()
    }
}

impl From<[[f32; 2]; 4]> for Obb {
    fn from(mut vertices: [[f32; 2]; 4]) -> Self {
        Self::normalize_vertices(&mut vertices);
        Self {
            vertices,
            ..Default::default()
        }
    }
}

impl From<Vec<[f32; 2]>> for Obb {
    fn from(vertices: Vec<[f32; 2]>) -> Self {
        let mut vertices = [vertices[0], vertices[1], vertices[2], vertices[3]];
        Self::normalize_vertices(&mut vertices);
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

        // Use From trait to ensure normalization
        Self::from([v1, v2, v3, v4])
    }

    /// Get the 4 vertices as slice (CCW ordered, starting from bottom-left)
    pub fn coords(&self) -> &[[f32; 2]; 4] {
        &self.vertices
    }

    /// Check if this OBB is axis-aligned (i.e., it's actually an Hbb)
    /// Returns true if all edges are parallel to x or y axis
    #[inline]
    pub fn is_hbb(&self) -> bool {
        let v = &self.vertices;

        // Check if all edges are horizontal or vertical
        // Edge 0-1: either dx=0 or dy=0
        // Edge 1-2: either dx=0 or dy=0
        // etc.

        let epsilon = 1e-6;

        for i in 0..4 {
            let j = (i + 1) % 4;
            let dx = (v[j][0] - v[i][0]).abs();
            let dy = (v[j][1] - v[i][1]).abs();

            // Each edge must be either horizontal (dy≈0) or vertical (dx≈0)
            if dx > epsilon && dy > epsilon {
                return false;
            }
        }

        true
    }

    /// Get [top-left, top-right, bottom-right, bottom-left] corners for Hbb
    /// Or [top, left, bottom, right] extreme points for rotated OBB
    ///
    /// For axis-aligned rectangles (Hbb), returns 4 distinct corner points.
    /// For rotated OBB, returns extreme points (some may coincide at corners).
    #[inline]
    pub fn tlbr(&self) -> [[f32; 2]; 4] {
        if self.is_hbb() {
            self.hbb_corners()
        } else {
            self.extreme_points()
        }
    }

    /// Get 4 corners for axis-aligned rectangle (Hbb mode)
    /// Returns [top-left, top-right, bottom-right, bottom-left]
    #[inline]
    fn hbb_corners(&self) -> [[f32; 2]; 4] {
        let v = &self.vertices;

        // Find min/max coordinates
        let mut x_min = v[0][0];
        let mut x_max = v[0][0];
        let mut y_min = v[0][1];
        let mut y_max = v[0][1];

        for vertex in v.iter().skip(1) {
            if vertex[0] < x_min {
                x_min = vertex[0];
            }
            if vertex[0] > x_max {
                x_max = vertex[0];
            }
            if vertex[1] < y_min {
                y_min = vertex[1];
            }
            if vertex[1] > y_max {
                y_max = vertex[1];
            }
        }

        // Return 4 distinct corners
        [
            [x_min, y_min], // top-left
            [x_max, y_min], // top-right
            [x_max, y_max], // bottom-right
            [x_min, y_max], // bottom-left
        ]
    }

    /// Get extreme points in one pass (for rotated OBB)
    /// Returns [top, left, bottom, right] where each is the vertex with extreme coordinate
    /// Note: For Hbb, some extreme points may coincide at corners
    #[inline]
    fn extreme_points(&self) -> [[f32; 2]; 4] {
        let v = &self.vertices;

        // Initialize with first vertex
        let mut top = v[0];
        let mut left = v[0];
        let mut bottom = v[0];
        let mut right = v[0];

        // Unrolled comparison for vertices 1, 2, 3
        // Vertex 1
        if v[1][1] < top[1] {
            top = v[1];
        }
        if v[1][0] < left[0] {
            left = v[1];
        }
        if v[1][1] > bottom[1] {
            bottom = v[1];
        }
        if v[1][0] > right[0] {
            right = v[1];
        }

        // Vertex 2
        if v[2][1] < top[1] {
            top = v[2];
        }
        if v[2][0] < left[0] {
            left = v[2];
        }
        if v[2][1] > bottom[1] {
            bottom = v[2];
        }
        if v[2][0] > right[0] {
            right = v[2];
        }

        // Vertex 3
        if v[3][1] < top[1] {
            top = v[3];
        }
        if v[3][0] < left[0] {
            left = v[3];
        }
        if v[3][1] > bottom[1] {
            bottom = v[3];
        }
        if v[3][0] > right[0] {
            right = v[3];
        }

        [top, left, bottom, right]
    }

    /// Get the topmost point (minimum y coordinate)
    #[inline]
    pub fn top(&self) -> [f32; 2] {
        self.tlbr()[0]
    }

    /// Get the leftmost point (minimum x coordinate)
    #[inline]
    pub fn left(&self) -> [f32; 2] {
        self.tlbr()[1]
    }

    /// Get the bottommost point (maximum y coordinate)
    #[inline]
    pub fn bottom(&self) -> [f32; 2] {
        self.tlbr()[2]
    }

    /// Get the rightmost point (maximum x coordinate)
    #[inline]
    pub fn right(&self) -> [f32; 2] {
        self.tlbr()[3]
    }

    /// Convert to Polygon with metadata
    pub fn to_polygon(&self) -> Polygon {
        let mut polygon = Polygon::try_from(&self.vertices).expect("OBB always has 4 vertices");
        if let Some(id) = self.id() {
            polygon = polygon.with_id(id);
        }
        if let Some(name) = self.name() {
            polygon = polygon.with_name(name);
        }
        if let Some(confidence) = self.confidence() {
            polygon = polygon.with_confidence(confidence);
        }

        polygon
    }

    /// Calculate area directly using Shoelace formula
    #[inline]
    pub fn area(&self) -> f32 {
        let v = &self.vertices;
        // Shoelace formula unrolled for 4 vertices
        // Cross products: (x0*y1 - x1*y0) + (x1*y2 - x2*y1) + (x2*y3 - x3*y2) + (x3*y0 - x0*y3)
        let area = v[0][0] * v[1][1] - v[1][0] * v[0][1] + v[1][0] * v[2][1] - v[2][0] * v[1][1]
            + v[2][0] * v[3][1]
            - v[3][0] * v[2][1]
            + v[3][0] * v[0][1]
            - v[0][0] * v[3][1];
        area.abs() * 0.5
    }

    /// Calculate intersection area with another OBB using optimized Sutherland-Hodgman
    pub fn intersect(&self, other: &Self) -> f32 {
        let mut output: [[f32; 2]; 8] = [[0.0, 0.0]; 8];
        let mut output_count = 4;
        output[..4].copy_from_slice(&self.vertices);

        // Clip against each edge of the clipping polygon (other)
        for i in 0..4 {
            if output_count == 0 {
                return 0.0;
            }

            let edge_start = other.vertices[i];
            let edge_end = other.vertices[(i + 1) % 4];

            // Input becomes previous output
            let mut input: [[f32; 2]; 8] = [[0.0, 0.0]; 8];
            let input_count = output_count;
            input[..input_count].copy_from_slice(&output[..input_count]);
            output_count = 0;

            for j in 0..input_count {
                let current = input[j];
                let next = input[(j + 1) % input_count];

                let current_inside = Self::is_inside(&current, &edge_start, &edge_end);
                let next_inside = Self::is_inside(&next, &edge_start, &edge_end);

                if next_inside {
                    if !current_inside {
                        if let Some(intersection) =
                            Self::line_intersection(&current, &next, &edge_start, &edge_end)
                        {
                            if output_count < 8 {
                                output[output_count] = intersection;
                                output_count += 1;
                            }
                        }
                    }
                    if output_count < 8 {
                        output[output_count] = next;
                        output_count += 1;
                    }
                } else if current_inside {
                    if let Some(intersection) =
                        Self::line_intersection(&current, &next, &edge_start, &edge_end)
                    {
                        if output_count < 8 {
                            output[output_count] = intersection;
                            output_count += 1;
                        }
                    }
                }
            }
        }

        if output_count < 3 {
            return 0.0;
        }

        // Calculate area of intersection polygon using Shoelace formula
        let mut area = 0.0;
        for i in 0..output_count {
            let j = (i + 1) % output_count;
            area += output[i][0] * output[j][1];
            area -= output[j][0] * output[i][1];
        }
        area.abs() * 0.5
    }

    #[inline]
    fn is_inside(point: &[f32; 2], edge_start: &[f32; 2], edge_end: &[f32; 2]) -> bool {
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
            >= 0.0
    }

    /// Normalize OBB vertices to ensure consistent IoU calculation
    /// 1. Sort by polar angle from centroid (handles unordered input)
    /// 2. Ensure counter-clockwise order (for intersection algorithm)
    /// 3. Rotate to start from bottom-left vertex (for consistent comparison)
    fn normalize_vertices(vertices: &mut [[f32; 2]; 4]) {
        // Calculate centroid
        let cx = (vertices[0][0] + vertices[1][0] + vertices[2][0] + vertices[3][0]) / 4.0;
        let cy = (vertices[0][1] + vertices[1][1] + vertices[2][1] + vertices[3][1]) / 4.0;

        // Sort by polar angle to ensure ring order
        vertices.sort_by(|a, b| {
            let angle_a = (a[1] - cy).atan2(a[0] - cx);
            let angle_b = (b[1] - cy).atan2(b[0] - cx);
            angle_a
                .partial_cmp(&angle_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Ensure CCW order using signed area
        let mut area = 0.0;
        for i in 0..4 {
            let j = (i + 1) % 4;
            area += (vertices[j][0] - vertices[i][0]) * (vertices[j][1] + vertices[i][1]);
        }

        // If clockwise (area >= 0), reverse to make counter-clockwise
        if area >= 0.0 {
            vertices.reverse();
        }

        // Rotate to start from bottom-left vertex (min y, then min x)
        // This ensures consistent starting point for intersection algorithm
        let mut min_idx = 0;
        for (i, vertex) in vertices.iter().enumerate().skip(1) {
            if vertex[1] < vertices[min_idx][1]
                || (vertex[1] == vertices[min_idx][1] && vertex[0] < vertices[min_idx][0])
            {
                min_idx = i;
            }
        }

        // Rotate array to start from min_idx
        if min_idx != 0 {
            vertices.rotate_left(min_idx);
        }
    }

    #[inline]
    fn line_intersection(
        p1: &[f32; 2],
        p2: &[f32; 2],
        p3: &[f32; 2],
        p4: &[f32; 2],
    ) -> Option<[f32; 2]> {
        let x1 = p1[0];
        let y1 = p1[1];
        let x2 = p2[0];
        let y2 = p2[1];
        let x3 = p3[0];
        let y3 = p3[1];
        let x4 = p4[0];
        let y4 = p4[1];

        let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if denom.abs() < 1e-10 {
            return None;
        }

        let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;

        Some([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    }

    pub fn union(&self, other: &Self) -> f32 {
        self.area() + other.area() - self.intersect(other)
    }

    pub fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }
}

#[cfg(test)]
mod tests_mbr {
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
        let iou_result = a.iou(&b);
        assert_eq!(iou_result, 0.2);
    }

    #[test]
    fn test_is_hbb() {
        // Axis-aligned rectangle (Hbb)
        let hbb = Obb::from([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        assert!(
            hbb.is_hbb(),
            "Axis-aligned rectangle should be detected as Hbb"
        );

        // Rotated rectangle (OBB)
        let obb = Obb::from_cxcywhr(5.0, 5.0, 6.0, 4.0, std::f32::consts::FRAC_PI_4); // 45 degrees
        assert!(!obb.is_hbb(), "Rotated rectangle should NOT be Hbb");
    }

    #[test]
    fn test_tlbr_hbb() {
        // Test tlbr() for axis-aligned rectangle (Hbb mode)
        let hbb = Obb::from([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);

        let tlbr = hbb.tlbr();

        // Should return 4 DISTINCT corners
        assert_eq!(tlbr[0], [0.0, 0.0], "top-left");
        assert_eq!(tlbr[1], [10.0, 0.0], "top-right");
        assert_eq!(tlbr[2], [10.0, 10.0], "bottom-right");
        assert_eq!(tlbr[3], [0.0, 10.0], "bottom-left");
    }

    #[test]
    fn test_tlbr_obb() {
        // Test tlbr() for rotated OBB (extreme points mode)
        let obb = Obb::from_cxcywhr(5.0, 5.0, 6.0, 4.0, std::f32::consts::FRAC_PI_4); // 45 degrees
        let tlbr = obb.tlbr();

        // Verify extreme points ordering
        assert!(tlbr[0][1] < tlbr[2][1], "Top y < Bottom y");
        assert!(tlbr[1][0] < tlbr[3][0], "Left x < Right x");
    }

    #[test]
    fn test_unordered_vertices() {
        let unordered = Obb::from([[0., 0.], [2., 2.], [0., 2.], [2., 0.]]); // Random order!
        let ordered = Obb::from([[0., 0.], [2., 0.], [2., 2.], [0., 2.]]); // Correct order

        // Should have same area
        assert_eq!(
            unordered.area(),
            ordered.area(),
            "Unordered and ordered should have same area"
        );

        let iou = unordered.iou(&ordered);
        assert!(
            iou > 0.99,
            "Should be nearly identical after normalization, got IoU={iou}"
        );
    }

    #[test]
    fn test_real_duplicate_obbs() {
        let obb1 = Obb::from([
            [542.4359, 179.76318],
            [624.21497, 273.6377],
            [729.36743, 182.0339],
            [647.5884, 88.15939],
        ]);
        let obb2 = Obb::from([
            [538.0109, 181.63753],
            [619.72876, 276.03265],
            [729.5933, 180.92296],
            [647.8755, 86.52783],
        ]);
        let iou = obb1.iou(&obb2);
        assert!(iou > 0.7, "Similar OBBs should have high IoU, got {iou}");
    }
}
