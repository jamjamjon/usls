use crate::{Bbox, Embedding, Keypoint, Mask, Mbr, Polygon, Prob};

/// Inference results container for each image.
#[derive(Clone, PartialEq, Default)]
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
        let mut f = f.debug_struct("Result");
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
    pub fn with_masks(mut self, masks: &[Mask]) -> Self {
        self.masks = Some(masks.to_vec());
        self
    }

    pub fn with_probs(mut self, probs: Prob) -> Self {
        self.probs = Some(probs);
        self
    }

    pub fn with_texts(mut self, texts: &[String]) -> Self {
        self.texts = Some(texts.to_vec());
        self
    }

    pub fn with_mbrs(mut self, mbrs: &[Mbr]) -> Self {
        self.mbrs = Some(mbrs.to_vec());
        self
    }

    pub fn with_bboxes(mut self, bboxes: &[Bbox]) -> Self {
        self.bboxes = Some(bboxes.to_vec());
        self
    }

    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_keypoints(mut self, keypoints: &[Vec<Keypoint>]) -> Self {
        self.keypoints = Some(keypoints.to_vec());
        self
    }

    pub fn with_polygons(mut self, polygons: &[Polygon]) -> Self {
        self.polygons = Some(polygons.to_vec());
        self
    }

    pub fn masks(&self) -> Option<&Vec<Mask>> {
        self.masks.as_ref()
    }

    pub fn probs(&self) -> Option<&Prob> {
        self.probs.as_ref()
    }

    pub fn keypoints(&self) -> Option<&Vec<Vec<Keypoint>>> {
        self.keypoints.as_ref()
    }

    pub fn polygons(&self) -> Option<&Vec<Polygon>> {
        self.polygons.as_ref()
    }

    pub fn bboxes(&self) -> Option<&Vec<Bbox>> {
        self.bboxes.as_ref()
    }

    pub fn mbrs(&self) -> Option<&Vec<Mbr>> {
        self.mbrs.as_ref()
    }

    pub fn texts(&self) -> Option<&Vec<String>> {
        self.texts.as_ref()
    }

    pub fn embedding(&self) -> Option<&Embedding> {
        self.embedding.as_ref()
    }

    pub fn apply_bboxes_nms(mut self, iou_threshold: f32) -> Self {
        match &mut self.bboxes {
            None => self,
            Some(ref mut bboxes) => {
                Self::nms_bboxes(bboxes, iou_threshold);
                self
            }
        }
    }

    pub fn apply_mbrs_nms(mut self, iou_threshold: f32) -> Self {
        match &mut self.mbrs {
            None => self,
            Some(ref mut mbrs) => {
                mbrs.sort_by(|b1, b2| {
                    b2.confidence()
                        .partial_cmp(&b1.confidence())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut current_index = 0;
                for index in 0..mbrs.len() {
                    let mut drop = false;
                    for prev_index in 0..current_index {
                        let iou = mbrs[prev_index].iou(&mbrs[index]);
                        if iou > iou_threshold {
                            drop = true;
                            break;
                        }
                    }
                    if !drop {
                        mbrs.swap(current_index, index);
                        current_index += 1;
                    }
                }
                mbrs.truncate(current_index);
                self
            }
        }
    }

    pub fn nms_bboxes(bboxes: &mut Vec<Bbox>, iou_threshold: f32) {
        bboxes.sort_by(|b1, b2| {
            b2.confidence()
                .partial_cmp(&b1.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut current_index = 0;
        for index in 0..bboxes.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = bboxes[prev_index].iou(&bboxes[index]);
                if iou > iou_threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes.truncate(current_index);
    }
}
