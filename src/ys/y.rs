use aksr::Builder;

use crate::{Bbox, Embedding, Keypoint, Mask, Mbr, Nms, Polygon, Prob};

/// Container for inference results for each image.
///
/// This struct holds various possible outputs from an image inference process,
/// including probabilities, bounding boxes, keypoints, minimum bounding rectangles,
/// polygons, masks, text annotations, and embeddings.
///
/// # Fields
///
/// * `probs` - Optionally contains the probability scores for the detected objects.
/// * `bboxes` - Optionally contains a vector of bounding boxes.
/// * `keypoints` - Optionally contains a nested vector of keypoints.
/// * `mbrs` - Optionally contains a vector of minimum bounding rectangles.
/// * `polygons` - Optionally contains a vector of polygons.
/// * `texts` - Optionally contains a vector of text annotations.
/// * `masks` - Optionally contains a vector of masks.
/// * `embedding` - Optionally contains the embedding representation.
#[derive(Builder, Clone, PartialEq, Default)]
pub struct Y {
    probs: Option<Prob>,
    bboxes: Option<Vec<Bbox>>,
    keypoints: Option<Vec<Vec<Keypoint>>>,
    mbrs: Option<Vec<Mbr>>,
    polygons: Option<Vec<Polygon>>,
    texts: Option<Vec<String>>,
    masks: Option<Vec<Mask>>,
    embedding: Option<Embedding>,
}

impl std::fmt::Debug for Y {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Y");
        if let Some(x) = &self.texts {
            if !x.is_empty() {
                f.field("Texts", &x);
            }
        }
        if let Some(x) = &self.probs {
            f.field("Probabilities", &x);
        }
        if let Some(x) = &self.bboxes {
            if !x.is_empty() {
                f.field("BoundingBoxes", &x);
            }
        }
        if let Some(x) = &self.mbrs {
            if !x.is_empty() {
                f.field("MinimumBoundingRectangles", &x);
            }
        }
        if let Some(x) = &self.keypoints {
            if !x.is_empty() {
                f.field("Keypoints", &x);
            }
        }
        if let Some(x) = &self.polygons {
            if !x.is_empty() {
                f.field("Polygons", &x);
            }
        }
        if let Some(x) = &self.masks {
            if !x.is_empty() {
                f.field("Masks", &x);
            }
        }
        if let Some(x) = &self.embedding {
            f.field("Embedding", &x);
        }
        f.finish()
    }
}

impl Y {
    pub fn apply_nms(mut self, iou_threshold: f32) -> Self {
        match &mut self.bboxes {
            None => match &mut self.mbrs {
                None => self,
                Some(ref mut mbrs) => {
                    Self::nms(mbrs, iou_threshold);
                    self
                }
            },
            Some(ref mut bboxes) => {
                Self::nms(bboxes, iou_threshold);
                self
            }
        }
    }

    pub fn nms<T: Nms>(xxx: &mut Vec<T>, iou_threshold: f32) {
        xxx.sort_by(|b1, b2| {
            b2.confidence()
                .partial_cmp(&b1.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut current_index = 0;
        for index in 0..xxx.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = xxx[prev_index].iou(&xxx[index]);
                if iou > iou_threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                xxx.swap(current_index, index);
                current_index += 1;
            }
        }
        xxx.truncate(current_index);
    }
}
