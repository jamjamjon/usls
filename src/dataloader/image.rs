use aksr::Builder;
use anyhow::Result;
use image::{DynamicImage, GrayImage, RgbImage, RgbaImage, SubImage};
use std::path::{Path, PathBuf};

use crate::{Hub, X};

/// Image wrapper with metadata and transformation capabilities.
#[derive(Builder, Clone)]
pub struct Image {
    /// `ImageBuffer<Rgb<u8>, Vec<u8>>`
    pub image: RgbImage, // TODO
    pub source: Option<PathBuf>,
    pub timestamp: Option<f64>,
}

impl Default for Image {
    fn default() -> Self {
        Self {
            image: RgbImage::new(0, 0),
            source: None,
            timestamp: None,
        }
    }
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("Height", &self.height())
            .field("Width", &self.width())
            .field("Source", &self.source)
            .finish()
    }
}

impl std::ops::Deref for Image {
    type Target = RgbImage;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl std::ops::DerefMut for Image {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.image
    }
}

impl From<DynamicImage> for Image {
    fn from(image: DynamicImage) -> Self {
        Self {
            image: image.into_rgb8(),
            ..Default::default()
        }
    }
}

impl From<GrayImage> for Image {
    fn from(image: GrayImage) -> Self {
        Self {
            image: DynamicImage::from(image).into_rgb8(),
            ..Default::default()
        }
    }
}

impl From<RgbImage> for Image {
    fn from(image: RgbImage) -> Self {
        Self {
            image,
            ..Default::default()
        }
    }
}

impl From<RgbaImage> for Image {
    fn from(image: RgbaImage) -> Self {
        Self {
            image: DynamicImage::from(image).into_rgb8(),
            ..Default::default()
        }
    }
}

impl<I> From<SubImage<I>> for Image
where
    I: std::ops::Deref,
    I::Target: image::GenericImageView<Pixel = image::Rgb<u8>> + 'static,
{
    fn from(sub_image: SubImage<I>) -> Self {
        let image: RgbImage = sub_image.to_image();

        Self {
            image,
            ..Default::default()
        }
    }
}

impl From<Image> for DynamicImage {
    fn from(image: Image) -> Self {
        image.into_dyn()
    }
}

impl From<Image> for RgbImage {
    fn from(image: Image) -> Self {
        image.into_rgb8()
    }
}

impl From<Image> for RgbaImage {
    fn from(image: Image) -> Self {
        image.into_rgba8()
    }
}

impl Image {
    pub fn from_u8s(u8s: &[u8], width: u32, height: u32) -> Result<Self> {
        let image = RgbImage::from_raw(width, height, u8s.to_vec())
                    .ok_or_else(|| anyhow::anyhow!("Failed to create image from raw data: buffer length might not match width * height * 3"))?;
        Ok(Self {
            image,
            ..Default::default()
        })
    }

    pub fn from_url(url: &str) -> anyhow::Result<Image> {
        let bytes = ureq::get(url).call()?.into_body().read_to_vec()?;

        Ok(image::load_from_memory(&bytes)?.into())
    }

    pub fn try_read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut path_buf = path.as_ref().to_path_buf();
        let path_str = path_buf.to_str().unwrap_or("");

        // Check remote
        if crate::REMOTE_PROTOCOLS
            .iter()
            .any(|&p| path_str.starts_with(p))
        {
            return Ok(Self {
                image: Self::from_url(path_str)?.into(),
                source: Some(path_buf),
                timestamp: None,
            });
        }

        // Check github Releases
        if !path_buf.exists() {
            if let Ok(p) = Hub::default().try_fetch(path_str) {
                path_buf = PathBuf::from(p);
            } else {
                return Err(anyhow::anyhow!(
                    "Failed to locate path: {:?} (not found locally or in hub)",
                    path_buf.display()
                ));
            }
        }

        // Local
        let image = image::ImageReader::open(&path_buf)?
            .with_guessed_format()?
            .decode()?;

        Ok(Self {
            image: image.into_rgb8(),
            source: Some(path_buf),
            timestamp: None,
        })
    }

    pub fn save<P: AsRef<Path>>(&self, p: P) -> Result<()> {
        match self.image.save(p.as_ref()) {
            Ok(_) => {
                tracing::info!("Saved image to: {:?}", p.as_ref().display());
                Ok(())
            }
            Err(err) => Err(anyhow::anyhow!("Failed to save image: {err:?}")),
        }
    }

    /// (width, height)
    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }

    pub fn height(&self) -> u32 {
        self.image.height()
    }

    pub fn width(&self) -> u32 {
        self.image.width()
    }

    pub fn size(&self) -> u32 {
        self.image.as_raw().len() as u32
    }

    pub fn to_u32s_into(&self, buffer: &mut Vec<u32>) {
        use rayon::prelude::*;
        let raw = self.image.as_raw();
        let (w, h) = self.image.dimensions();
        let len = (w * h) as usize;

        if buffer.len() != len {
            buffer.resize(len, 0);
        }

        // Use a more robust chunking approach
        buffer.par_iter_mut().enumerate().for_each(|(i, pixel)| {
            let offset = i * 3;
            let r = raw[offset] as u32;
            let g = raw[offset + 1] as u32;
            let b = raw[offset + 2] as u32;
            *pixel = (r << 16) | (g << 8) | b;
        });
    }

    pub fn to_f32s(&self) -> Vec<f32> {
        use rayon::prelude::*;

        self.image
            .as_raw()
            .into_par_iter()
            .map(|x| *x as f32)
            .collect()
    }

    pub fn to_dyn(&self) -> DynamicImage {
        DynamicImage::from(self.image.clone())
    }

    pub fn to_rgb8(&self) -> RgbImage {
        self.image.clone()
    }

    pub fn to_rgba8(&self) -> RgbaImage {
        DynamicImage::from(self.image.clone()).to_rgba8()
    }

    pub fn to_luma8(&self) -> GrayImage {
        DynamicImage::from(self.image.clone()).to_luma8()
    }

    pub fn into_dyn(self) -> DynamicImage {
        DynamicImage::from(self.image)
    }

    pub fn into_rgb8(self) -> RgbImage {
        self.image
    }

    pub fn into_rgba8(self) -> RgbaImage {
        self.into_dyn().to_rgba8()
    }

    pub fn into_luma8(self) -> GrayImage {
        self.into_dyn().to_luma8()
    }

    pub fn to_ndarray(&self) -> Result<X> {
        X::from_shape_vec(
            &[self.height() as usize, self.width() as usize, 3],
            self.to_f32s(),
        )
    }
}

/// Extension trait for converting between vectors of different image types.
/// Provides methods to convert between `Vec<Image>` and `Vec<DynamicImage>`.
pub trait ImageVecExt {
    /// Converts the vector into a vector of `DynamicImage`s.
    fn into_dyns(self) -> Vec<DynamicImage>;

    /// Converts the vector into a vector of `Image`s.
    fn into_images(self) -> Vec<Image>;
}

impl ImageVecExt for Vec<Image> {
    fn into_dyns(self) -> Vec<DynamicImage> {
        self.into_iter().map(|x| x.into()).collect()
    }

    fn into_images(self) -> Vec<Image> {
        self
    }
}

impl ImageVecExt for Vec<DynamicImage> {
    fn into_dyns(self) -> Vec<DynamicImage> {
        self
    }

    fn into_images(self) -> Vec<Image> {
        self.into_iter().map(|x| x.into()).collect()
    }
}
