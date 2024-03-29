use crate::{Bbox, Embedding, Keypoint};

#[derive(Clone, PartialEq, Default)]
pub struct Results {
    pub probs: Option<Embedding>,
    pub bboxes: Option<Vec<Bbox>>,
    pub keypoints: Option<Vec<Vec<Keypoint>>>,
    pub masks: Option<Vec<Vec<u8>>>,
}

impl std::fmt::Debug for Results {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Results")
            .field("Probabilities", &self.probs)
            .field("BoundingBoxes", &self.bboxes)
            .field("Keypoints", &self.keypoints)
            .field(
                "Masks",
                &format_args!("{:?}", self.masks().map(|masks| masks.len())),
            )
            .finish()
    }
}

impl Results {
    pub fn new(
        probs: Option<Embedding>,
        bboxes: Option<Vec<Bbox>>,
        keypoints: Option<Vec<Vec<Keypoint>>>,
        masks: Option<Vec<Vec<u8>>>,
    ) -> Self {
        Self {
            probs,
            bboxes,
            keypoints,
            masks,
        }
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

    pub fn bboxes(&self) -> Option<&Vec<Bbox>> {
        self.bboxes.as_ref()
    }

    pub fn bboxes_mut(&mut self) -> Option<&mut Vec<Bbox>> {
        self.bboxes.as_mut()
    }
}
