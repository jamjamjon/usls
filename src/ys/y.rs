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
    /// Sets the `masks` field with the provided vector of masks.
    ///
    /// # Arguments
    ///
    /// * `masks` - A slice of `Mask` to be set.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new masks set.
    pub fn with_masks(mut self, masks: &[Mask]) -> Self {
        self.masks = Some(masks.to_vec());
        self
    }

    /// Sets the `probs` field with the provided probability scores.
    ///
    /// # Arguments
    ///
    /// * `probs` - A reference to a `Prob` instance to be cloned and set in the struct.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new probabilities set.
    ///
    /// # Examples
    ///
    /// ```
    /// let probs = Prob::default();
    /// let y = Y::default().with_probs(&probs);
    /// ```
    pub fn with_probs(mut self, probs: &Prob) -> Self {
        self.probs = Some(probs.clone());
        self
    }

    /// Sets the `texts` field with the provided vector of text annotations.
    ///
    /// # Arguments
    ///
    /// * `texts` - A slice of `String` to be set.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new texts set.
    pub fn with_texts(mut self, texts: &[String]) -> Self {
        self.texts = Some(texts.to_vec());
        self
    }

    /// Sets the `mbrs` field with the provided vector of minimum bounding rectangles.
    ///
    /// # Arguments
    ///
    /// * `mbrs` - A slice of `Mbr` to be set.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new minimum bounding rectangles set.
    pub fn with_mbrs(mut self, mbrs: &[Mbr]) -> Self {
        self.mbrs = Some(mbrs.to_vec());
        self
    }

    /// Sets the `bboxes` field with the provided vector of bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `bboxes` - A slice of `Bbox` to be set.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new bounding boxes set.
    pub fn with_bboxes(mut self, bboxes: &[Bbox]) -> Self {
        self.bboxes = Some(bboxes.to_vec());
        self
    }

    /// Sets the `embedding` field with the provided embedding.
    ///
    /// # Arguments
    ///
    /// * `embedding` - A reference to an `Embedding` instance to be cloned and set in the struct.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new embedding set.
    pub fn with_embedding(mut self, embedding: &Embedding) -> Self {
        self.embedding = Some(embedding.clone());
        self
    }

    /// Sets the `keypoints` field with the provided nested vector of keypoints.
    ///
    /// # Arguments
    ///
    /// * `keypoints` - A slice of vectors of `Keypoint` to be set.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new keypoints set.
    pub fn with_keypoints(mut self, keypoints: &[Vec<Keypoint>]) -> Self {
        self.keypoints = Some(keypoints.to_vec());
        self
    }

    /// Sets the `polygons` field with the provided vector of polygons.
    ///
    /// # Arguments
    ///
    /// * `polygons` - A slice of `Polygon` to be set.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new polygons set.
    pub fn with_polygons(mut self, polygons: &[Polygon]) -> Self {
        self.polygons = Some(polygons.to_vec());
        self
    }

    /// Returns a reference to the `masks` field, if it exists.
    ///
    /// # Returns
    ///
    /// * `Option<&Vec<Mask>>` - A reference to the vector of masks, or `None` if it is not set.
    pub fn masks(&self) -> Option<&Vec<Mask>> {
        self.masks.as_ref()
    }

    /// Returns a reference to the `probs` field, if it exists.
    ///
    /// # Returns
    ///
    /// * `Option<&Prob>` - A reference to the probabilities, or `None` if it is not set.
    pub fn probs(&self) -> Option<&Prob> {
        self.probs.as_ref()
    }

    /// Returns a reference to the `keypoints` field, if it exists.
    ///
    /// # Returns
    ///
    /// * `Option<&Vec<Vec<Keypoint>>>` - A reference to the nested vector of keypoints, or `None` if it is not set.
    pub fn keypoints(&self) -> Option<&Vec<Vec<Keypoint>>> {
        self.keypoints.as_ref()
    }

    /// Returns a reference to the `polygons` field, if it exists.
    ///
    /// # Returns
    ///
    /// * `Option<&Vec<Polygon>>` - A reference to the vector of polygons, or `None` if it is not set.
    pub fn polygons(&self) -> Option<&Vec<Polygon>> {
        self.polygons.as_ref()
    }

    /// Returns a reference to the `bboxes` field, if it exists.
    ///
    /// # Returns
    ///
    /// * `Option<&Vec<Bbox>>` - A reference to the vector of bounding boxes, or `None` if it is not set.
    pub fn bboxes(&self) -> Option<&Vec<Bbox>> {
        self.bboxes.as_ref()
    }

    /// Returns a reference to the `mbrs` field, if it exists.
    ///
    /// # Returns
    ///
    /// * `Option<&Vec<Mbr>>` - A reference to the vector of minimum bounding rectangles, or `None` if it is not set.
    pub fn mbrs(&self) -> Option<&Vec<Mbr>> {
        self.mbrs.as_ref()
    }

    /// Returns a reference to the `texts` field, if it exists.
    ///
    /// # Returns
    ///
    /// * `Option<&Vec<String>>` - A reference to the vector of texts, or `None` if it is not set.
    pub fn texts(&self) -> Option<&Vec<String>> {
        self.texts.as_ref()
    }

    /// Returns a reference to the `embedding` field, if it exists.
    ///
    /// # Returns
    ///
    /// * `Option<&Embedding>` - A reference to the embedding, or `None` if it is not set.
    pub fn embedding(&self) -> Option<&Embedding> {
        self.embedding.as_ref()
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
