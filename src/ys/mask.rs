use image::GrayImage;

/// Mask: Gray Image.
#[derive(Clone, PartialEq)]
pub struct Mask {
    mask: GrayImage,
    id: isize,
    name: Option<String>,
    confidence: f32,
}

impl Default for Mask {
    fn default() -> Self {
        Self {
            mask: GrayImage::default(),
            id: -1,
            name: None,
            confidence: 0.,
        }
    }
}

impl std::fmt::Debug for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mask")
            .field("dimensions", &self.dimensions())
            .field("id", &self.id)
            .field("name", &self.name)
            .finish()
    }
}

impl Mask {
    pub fn with_mask(mut self, x: GrayImage) -> Self {
        self.mask = x;
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

    pub fn mask(&self) -> &GrayImage {
        &self.mask
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.mask.to_vec()
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

    pub fn height(&self) -> u32 {
        self.mask.height()
    }

    pub fn width(&self) -> u32 {
        self.mask.width()
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.mask.dimensions()
    }
}
