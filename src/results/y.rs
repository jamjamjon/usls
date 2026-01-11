use aksr::Builder;
use std::collections::HashMap;

use crate::{Hbb, Image, Keypoint, Mask, Obb, Polygon, Prob, Text, X};

/// Container for inference results for each image.
///
/// This struct holds various possible outputs from an image inference process,
/// including probabilities, bounding boxes, keypoints, minimum bounding rectangles,
/// polygons, masks, text annotations, and embeddings.
///
#[derive(Builder, Clone, Default)]
pub struct Y {
    pub texts: Vec<Text>,
    pub probs: Vec<Prob>,
    pub keypoints: Vec<Keypoint>,
    pub keypointss: Vec<Vec<Keypoint>>,
    pub hbbs: Vec<Hbb>,
    pub obbs: Vec<Obb>,
    pub polygons: Vec<Polygon>,
    pub masks: Vec<Mask>,
    pub images: Vec<Image>,
    pub embedding: X,
    pub extras: HashMap<String, X>,
}

impl std::fmt::Debug for Y {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Y");

        if !self.texts.is_empty() {
            s.field("Texts", &self.texts);
        }
        if !self.probs.is_empty() {
            s.field("Probs", &self.probs);
        }
        if !self.hbbs.is_empty() {
            s.field("Hbbs", &self.hbbs);
        }
        if !self.obbs.is_empty() {
            s.field("Obbs", &self.obbs);
        }
        if !self.keypoints.is_empty() {
            s.field("Kpts", &self.keypoints);
        }
        if !self.keypointss.is_empty() {
            s.field("Kptss", &self.keypointss);
        }
        if !self.polygons.is_empty() {
            s.field("Polys", &self.polygons);
        }
        if !self.masks.is_empty() {
            s.field("Masks", &self.masks);
        }
        if !self.images.is_empty() {
            s.field("Images", &self.images);
        }
        if !self.embedding.is_empty() {
            s.field("Embeddings", &self.embedding);
        }
        if !self.extras.is_empty() {
            s.field("Extras", &self.extras);
        }
        s.finish()
    }
}

impl Y {
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
            && self.probs.is_empty()
            && self.hbbs.is_empty()
            && self.obbs.is_empty()
            && self.keypoints.is_empty()
            && self.keypointss.is_empty()
            && self.polygons.is_empty()
            && self.masks.is_empty()
            && self.images.is_empty()
            && self.embedding.is_empty()
            && self.extras.is_empty()
    }
}
