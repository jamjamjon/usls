use image::DynamicImage;

/// Gray-Scale Mask.
#[derive(Clone, PartialEq)]
pub struct Mask {
    mask: DynamicImage,
    mask_vec: Vec<u8>,
    id: isize,
    name: Option<String>,
    confidence: f32, // placeholder
}

impl Default for Mask {
    fn default() -> Self {
        Self {
            mask: DynamicImage::default(),
            mask_vec: vec![],
            id: -1,
            name: None,
            confidence: 0.,
        }
    }
}

impl std::fmt::Debug for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mask")
            // .field("mask", &self.mask)
            .field("id", &self.id)
            .field("name", &self.name)
            // .field("confidence", &self.confidence)
            .finish()
    }
}

impl Mask {
    pub fn with_mask(mut self, x: DynamicImage) -> Self {
        self.mask = x;
        self
    }

    pub fn with_vec(mut self, vec: &[u8]) -> Self {
        self.mask_vec = vec.to_vec();
        self
    }

    pub fn with_id(mut self, x: isize) -> Self {
        self.id = x;
        self
    }

    pub fn with_name(mut self, x: Option<String>) -> Self {
        self.name = x;
        self
    }

    pub fn mask(&self) -> &DynamicImage {
        &self.mask
    }

    pub fn vec(&self) -> Vec<u8> {
        // self.mask.to_luma8().into_raw()
        self.mask_vec.clone()
    }

    pub fn id(&self) -> isize {
        self.id
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }
}
