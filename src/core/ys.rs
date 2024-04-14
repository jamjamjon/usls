use crate::{Bbox, Embedding, Keypoint, Mask};

#[derive(Clone, PartialEq, Default)]
pub struct Ys {
    // Results for each frame
    pub probs: Option<Embedding>,
    pub bboxes: Option<Vec<Bbox>>,
    pub keypoints: Option<Vec<Vec<Keypoint>>>,
    pub masks: Option<Vec<Mask>>,
}

impl std::fmt::Debug for Ys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Results")
            .field("Probabilities", &self.probs)
            .field("BoundingBoxes", &self.bboxes)
            .field("Keypoints", &self.keypoints)
            .field("Masks", &self.masks)
            .finish()
    }
}

impl Ys {
    pub fn with_probs(mut self, probs: Embedding) -> Self {
        self.probs = Some(probs);
        self
    }

    pub fn with_bboxes(mut self, bboxes: &[Bbox]) -> Self {
        self.bboxes = Some(bboxes.to_vec());
        self
    }

    pub fn with_keypoints(mut self, keypoints: &[Vec<Keypoint>]) -> Self {
        self.keypoints = Some(keypoints.to_vec());
        self
    }

    pub fn with_masks(mut self, masks: &[Mask]) -> Self {
        self.masks = Some(masks.to_vec());
        self
    }

    pub fn probs(&self) -> Option<&Embedding> {
        self.probs.as_ref()
    }

    pub fn keypoints(&self) -> Option<&Vec<Vec<Keypoint>>> {
        self.keypoints.as_ref()
    }

    pub fn masks(&self) -> Option<&Vec<Mask>> {
        self.masks.as_ref()
    }

    pub fn bboxes(&self) -> Option<&Vec<Bbox>> {
        self.bboxes.as_ref()
    }

    pub fn non_max_suppression(xs: &mut Vec<Bbox>, iou_threshold: f32) {
        xs.sort_by(|b1, b2| b2.confidence().partial_cmp(&b1.confidence()).unwrap());
        let mut current_index = 0;
        for index in 0..xs.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = xs[prev_index].iou(&xs[index]);
                if iou > iou_threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                xs.swap(current_index, index);
                current_index += 1;
            }
        }
        xs.truncate(current_index);
    }
}
