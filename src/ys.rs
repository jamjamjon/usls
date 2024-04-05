use crate::{Bbox, Embedding, Keypoint, Polygon};

#[derive(Clone, PartialEq, Default)]
pub struct Ys {
    // Results for each frame
    pub probs: Option<Embedding>,
    pub bboxes: Option<Vec<Bbox>>,
    pub keypoints: Option<Vec<Vec<Keypoint>>>,
    pub masks: Option<Vec<Vec<u8>>>,
    pub polygons: Option<Vec<Polygon>>,
}

impl std::fmt::Debug for Ys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Results")
            .field("Probabilities", &self.probs)
            .field("BoundingBoxes", &self.bboxes)
            .field("Keypoints", &self.keypoints)
            .field(
                "Masks",
                &format_args!("{:?}", self.masks().map(|masks| masks.len())),
            )
            .field(
                "Polygons",
                &format_args!("{:?}", self.polygons().map(|polygons| polygons.len())),
            )
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

    pub fn with_masks(mut self, masks: &[Vec<u8>]) -> Self {
        self.masks = Some(masks.to_vec());
        self
    }

    pub fn with_polygons(mut self, polygons: &[Polygon]) -> Self {
        self.polygons = Some(polygons.to_vec());
        self
    }

    pub fn probs(&self) -> Option<&Embedding> {
        self.probs.as_ref()
    }

    pub fn keypoints(&self) -> Option<&Vec<Vec<Keypoint>>> {
        self.keypoints.as_ref()
    }

    pub fn masks(&self) -> Option<&Vec<Vec<u8>>> {
        self.masks.as_ref()
    }

    pub fn polygons(&self) -> Option<&Vec<Polygon>> {
        self.polygons.as_ref()
    }

    pub fn bboxes(&self) -> Option<&Vec<Bbox>> {
        self.bboxes.as_ref()
    }
}
