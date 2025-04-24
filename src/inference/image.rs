use aksr::Builder;
use anyhow::Result;
use fast_image_resize::{
    images::{CroppedImageMut, Image as FImage},
    pixels::PixelType,
};
use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};
use std::path::{Path, PathBuf};

use crate::{build_resizer_filter, Hub, Location, MediaType, X};

#[derive(Builder, Debug, Clone, Default)]
pub struct ImageTransformInfo {
    pub width_src: u32,
    pub height_src: u32,
    pub width_dst: u32,
    pub height_dst: u32,
    pub height_scale: f32,
    pub width_scale: f32,
}

#[derive(Debug, Clone, Default)]
pub enum ResizeMode {
    /// StretchToFit
    FitExact,
    FitWidth,
    FitHeight,
    #[default]
    FitAdaptive,
    Letterbox,
}

#[derive(Builder, Clone)]
pub struct Image {
    image: RgbImage,
    source: Option<PathBuf>,
    media_type: MediaType,
}

impl Default for Image {
    fn default() -> Self {
        Self {
            image: RgbImage::new(0, 0),
            source: None,
            media_type: MediaType::Unknown,
        }
    }
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("Height", &self.height())
            .field("Width", &self.width())
            .field("MediaType", &self.media_type)
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
            image: image.to_rgb8(),
            ..Default::default()
        }
    }
}

impl From<GrayImage> for Image {
    fn from(image: GrayImage) -> Self {
        Self {
            image: DynamicImage::from(image).to_rgb8(),
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
            image: DynamicImage::from(image).to_rgb8(),
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

    pub fn try_read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let media_type;
        let mut path = path.as_ref().to_path_buf();

        // try to fetch from hub or local cache
        if !path.exists() {
            let p = match Hub::default()
                .try_fetch(path.to_str().expect("Failed to convert path to str"))
            {
                Ok(p) => {
                    media_type = MediaType::Image(Location::Remote);
                    p
                }
                Err(err) => {
                    return Err(anyhow::anyhow!(
                        "Failed to locate path: {:?} and file also not found in hub. Error: {:?}",
                        path.display(),
                        err
                    ));
                }
            };
            path = PathBuf::from(&p);
        } else {
            media_type = MediaType::Image(Location::Local);
        }

        let image = image::ImageReader::open(&path)
            .map_err(|err| {
                anyhow::anyhow!(
                    "Failed to open image at {:?}. Error: {:?}",
                    path.display(),
                    err
                )
            })?
            .with_guessed_format()
            .map_err(|err| {
                anyhow::anyhow!(
                    "Failed to make a format guess based on the content: {:?}. Error: {:?}",
                    path.display(),
                    err
                )
            })?
            .decode()
            .map_err(|err| {
                anyhow::anyhow!(
                    "Failed to decode image at {:?}. Error: {:?}",
                    path.display(),
                    err
                )
            })?;

        Ok(Self {
            image: image.to_rgb8(),
            media_type,
            source: Some(path),
        })
    }

    pub fn save<P: AsRef<Path>>(&self, p: P) -> Result<()> {
        self.image
            .save(p.as_ref())
            .map_err(|err| anyhow::anyhow!("Failed to save image: {:?}", err))
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

    pub fn to_u32s(&self) -> Vec<u32> {
        use rayon::prelude::*;

        self.image
            .as_raw()
            .par_chunks(3)
            .map(|c| ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32))
            .collect()
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

    pub fn resize(
        &self,
        tw: u32,
        th: u32,
        filter: &str,
        mode: &ResizeMode,
        padding_value: u8,
    ) -> Result<Self> {
        Ok(self
            .resize_with_info(tw, th, filter, mode, padding_value)?
            .0)
    }

    pub fn resize_with_info(
        &self,
        tw: u32,
        th: u32,
        filter: &str,
        mode: &ResizeMode,
        padding_value: u8,
    ) -> Result<(Self, ImageTransformInfo)> {
        if tw + th == 0 {
            anyhow::bail!("Invalid target height: {} or width: {}.", th, tw);
        }

        let (w0, h0) = self.dimensions();
        let mut trans_info = ImageTransformInfo::default()
            .with_width_src(w0)
            .with_height_src(h0)
            .with_width_dst(tw)
            .with_height_dst(th);

        if (w0, h0) == (tw, th) {
            return Ok((
                self.clone(),
                trans_info.with_width_scale(1.).with_height_scale(1.),
            ));
        }

        let (mut resizer, options) = build_resizer_filter(filter)?;
        let x: DynamicImage = self.to_dyn();

        if let ResizeMode::FitExact = mode {
            let mut dst = FImage::new(tw, th, PixelType::U8x3);
            resizer.resize(&x, &mut dst, &options)?;
            trans_info = trans_info
                .with_height_scale(th as f32 / h0 as f32)
                .with_width_scale(tw as f32 / w0 as f32);

            Ok((Self::from_u8s(&dst.into_vec(), tw, th)?, trans_info))
        } else {
            let (w, h) = match mode {
                ResizeMode::Letterbox | ResizeMode::FitAdaptive => {
                    let r = (tw as f32 / w0 as f32).min(th as f32 / h0 as f32);
                    trans_info = trans_info.with_height_scale(r).with_width_scale(r);

                    (
                        (w0 as f32 * r).round() as u32,
                        (h0 as f32 * r).round() as u32,
                    )
                }
                ResizeMode::FitHeight => {
                    let r = th as f32 / h0 as f32;
                    trans_info = trans_info.with_height_scale(1.).with_width_scale(r);

                    ((r * w0 as f32).round() as u32, th)
                }
                ResizeMode::FitWidth => {
                    let r = tw as f32 / w0 as f32;
                    trans_info = trans_info.with_height_scale(r).with_width_scale(1.);

                    (tw, (r * h0 as f32).round() as u32)
                }
                _ => unreachable!(),
            };

            let mut dst = FImage::from_vec_u8(
                tw,
                th,
                vec![padding_value; 3 * th as usize * tw as usize],
                PixelType::U8x3,
            )?;
            let (l, t) = if let ResizeMode::Letterbox = mode {
                if w == tw {
                    (0, (th - h) / 2)
                } else {
                    ((tw - w) / 2, 0)
                }
            } else {
                (0, 0)
            };

            let mut dst_cropped = CroppedImageMut::new(&mut dst, l, t, w, h)?;
            resizer.resize(&x, &mut dst_cropped, &options)?;

            Ok((Self::from_u8s(&dst.into_vec(), tw, th)?, trans_info))
        }
    }

    pub fn to_ndarray(&self) -> Result<X> {
        X::from_shape_vec(
            &[self.height() as usize, self.width() as usize, 3],
            self.to_f32s(),
        )
    }
}

pub trait ImageVecExt {
    fn into_dyns(self) -> Vec<DynamicImage>;
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
