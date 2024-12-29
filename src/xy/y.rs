use aksr::Builder;

use crate::{Bbox, Keypoint, Mask, Mbr, Nms, Polygon, Prob, Text, X};

/// Container for inference results for each image.
///
/// This struct holds various possible outputs from an image inference process,
/// including probabilities, bounding boxes, keypoints, minimum bounding rectangles,
/// polygons, masks, text annotations, and embeddings.
///
/// # Fields
///
/// * `texts` - Optionally contains a vector of texts.
/// * `embedding` - Optionally contains the embedding representation.
/// * `probs` - Optionally contains the probability scores for the detected objects.
/// * `bboxes` - Optionally contains a vector of bounding boxes.
/// * `keypoints` - Optionally contains a nested vector of keypoints.
/// * `mbrs` - Optionally contains a vector of minimum bounding rectangles.
/// * `polygons` - Optionally contains a vector of polygons.
/// * `masks` - Optionally contains a vector of masks.
#[derive(Builder, Clone, PartialEq, Default)]
pub struct Y {
    texts: Option<Vec<Text>>,
    embedding: Option<X>,
    probs: Option<Prob>,
    bboxes: Option<Vec<Bbox>>,
    keypoints: Option<Vec<Vec<Keypoint>>>,
    mbrs: Option<Vec<Mbr>>,
    polygons: Option<Vec<Polygon>>,
    masks: Option<Vec<Mask>>,
}

impl std::fmt::Debug for Y {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Y");
        if let Some(xs) = &self.texts {
            if !xs.is_empty() {
                f.field("Texts", &xs);
            }
        }
        if let Some(xs) = &self.probs {
            f.field("Probs", &xs);
        }
        if let Some(xs) = &self.bboxes {
            if !xs.is_empty() {
                f.field("BBoxes", &xs);
            }
        }
        if let Some(xs) = &self.mbrs {
            if !xs.is_empty() {
                f.field("OBBs", &xs);
            }
        }
        if let Some(xs) = &self.keypoints {
            if !xs.is_empty() {
                f.field("Kpts", &xs);
            }
        }
        if let Some(xs) = &self.polygons {
            if !xs.is_empty() {
                f.field("Polys", &xs);
            }
        }
        if let Some(xs) = &self.masks {
            if !xs.is_empty() {
                f.field("Masks", &xs);
            }
        }
        if let Some(x) = &self.embedding {
            f.field("Embedding", &x);
        }
        f.finish()
    }
}

impl Y {
    pub fn hbbs(&self) -> Option<&[Bbox]> {
        self.bboxes.as_deref()
    }

    pub fn obbs(&self) -> Option<&[Mbr]> {
        self.mbrs.as_deref()
    }

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
