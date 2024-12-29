use aksr::Builder;
use image::GrayImage;

/// Mask: Gray Image.
#[derive(Builder, Clone, PartialEq)]
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
    pub fn to_vec(&self) -> Vec<u8> {
        self.mask.to_vec()
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
