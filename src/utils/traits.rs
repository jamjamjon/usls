use crate::{Hbb, Obb};

pub trait HasScore {
    fn score(&self) -> f32;
}

pub trait HasIoU {
    fn iou(&self, other: &Self) -> f32;
}

pub trait NmsOps {
    fn apply_nms_inplace(&mut self, iou_threshold: f32);
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

pub trait Region {
    fn area(&self) -> f32;
    fn perimeter(&self) -> f32;
    fn intersect(&self, other: &Self) -> f32;
    fn union(&self, other: &Self) -> f32;
    fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }
}
