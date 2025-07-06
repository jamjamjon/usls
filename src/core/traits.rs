use crate::{Hbb, Obb};

/// Trait for models that can provide performance summaries.
pub trait ModelSummary {
    /// Print performance summary for the model.
    fn summary(&mut self);
}

/// Trait for objects that have a confidence score.
pub trait HasScore {
    /// Returns the confidence score.
    fn score(&self) -> f32;
}

/// Trait for objects that can calculate Intersection over Union (IoU).
pub trait HasIoU {
    /// Calculates IoU with another object.
    fn iou(&self, other: &Self) -> f32;
}

/// Trait for Non-Maximum Suppression operations.
pub trait NmsOps {
    /// Applies NMS in-place with the given IoU threshold.
    fn apply_nms_inplace(&mut self, iou_threshold: f32);
    /// Applies NMS and returns the filtered result.
    fn apply_nms(self, iou_threshold: f32) -> Self;
}

impl<T> NmsOps for Vec<T>
where
    T: HasScore + HasIoU,
{
    fn apply_nms_inplace(&mut self, iou_threshold: f32) {
        self.sort_by(|a, b| b.score().total_cmp(&a.score()));

        let mut current_index = 0;
        for index in 0..self.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                if self[prev_index].iou(&self[index]) > iou_threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                self.swap(current_index, index);
                current_index += 1;
            }
        }

        self.truncate(current_index);
    }

    fn apply_nms(mut self, iou_threshold: f32) -> Self {
        self.apply_nms_inplace(iou_threshold);
        self
    }
}

impl HasScore for Hbb {
    fn score(&self) -> f32 {
        self.confidence().unwrap_or(0.)
    }
}

impl HasIoU for Hbb {
    fn iou(&self, other: &Self) -> f32 {
        self.iou(other)
    }
}

impl HasScore for Obb {
    fn score(&self) -> f32 {
        self.confidence().unwrap_or(0.)
    }
}

impl HasIoU for Obb {
    fn iou(&self, other: &Self) -> f32 {
        self.iou(other)
    }
}

/// Trait for geometric regions with area and intersection calculations.
pub trait Region {
    /// Calculates the area of the region.
    fn area(&self) -> f32;
    /// Calculates the perimeter of the region.
    fn perimeter(&self) -> f32;
    /// Calculates the intersection area with another region.
    fn intersect(&self, other: &Self) -> f32;
    /// Calculates the union area with another region.
    fn union(&self, other: &Self) -> f32;
    /// Calculates Intersection over Union (IoU) with another region.
    fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }
}
