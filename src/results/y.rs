use aksr::Builder;

use crate::{Hbb, Image, Keypoint, Mask, Obb, Polygon, Prob, Text};

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
}

impl std::fmt::Debug for Y {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Y");
        let fields: &[(&'static str, &dyn std::fmt::Debug)] = &[
            ("Texts", &self.texts),
            ("Probs", &self.probs),
            ("Hbbs", &self.hbbs),
            ("Obbs", &self.obbs),
            ("Kpts", &self.keypoints),
            ("Kptss", &self.keypointss),
            ("Polys", &self.polygons),
            ("Masks", &self.masks),
            ("Images", &self.images),
        ];
        for (name, value) in fields {
            f.field(name, value);
        }
        f.finish()
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
    }
}
