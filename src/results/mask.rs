use aksr::Builder;
use anyhow::Result;
use image::GrayImage;
use rayon::prelude::*;

use crate::{InstanceMeta, Polygon, Style};

/// Mask: Gray Image.
#[derive(Builder, Default, Clone)]
pub struct Mask {
    /// The grayscale image representing the mask.
    mask: GrayImage,
    /// Metadata associated with the mask instance.
    meta: InstanceMeta,
    /// Optional styling information for visualization.
    style: Option<Style>,
}

// #[derive(Builder, Default, Clone)]
// pub struct Masks(Vec<Mask>);

impl std::fmt::Debug for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Mask");
        f.field("dimensions", &self.dimensions());
        if let Some(id) = &self.meta.id() {
            f.field("id", id);
        }
        if let Some(name) = &self.meta.name() {
            f.field("name", name);
        }
        if let Some(confidence) = &self.meta.confidence() {
            f.field("confidence", confidence);
        }
        if let Some(track_id) = &self.meta.track_id() {
            f.field("track_id", track_id);
        }
        f.finish()
    }
}

impl PartialEq for Mask {
    fn eq(&self, other: &Self) -> bool {
        self.mask == other.mask
    }
}

impl Mask {
    impl_meta_methods!();
    pub fn new(u8s: &[u8], width: u32, height: u32) -> Result<Self> {
        let mask: image::ImageBuffer<image::Luma<_>, Vec<_>> =
            image::ImageBuffer::from_raw(width, height, u8s.to_vec())
                .ok_or(anyhow::anyhow!("Failed to build ImageBuffer."))?;

        Ok(Self {
            mask,
            ..Default::default()
        })
    }

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

    pub fn polygon(&self) -> Option<Polygon> {
        let polygons = self.polygons();
        if polygons.is_empty() {
            return None;
        }

        polygons.into_iter().max_by(|x, y| {
            x.area()
                .partial_cmp(&y.area())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn polygons(&self) -> Vec<Polygon> {
        let contours: Vec<imageproc::contours::Contour<i32>> =
            imageproc::contours::find_contours_with_threshold(self.mask(), 0);
        let polygons: Vec<Polygon> = contours
            .into_par_iter()
            .filter_map(|contour| {
                if contour.border_type == imageproc::contours::BorderType::Hole
                    && contour.points.len() <= 2
                {
                    return None;
                }
                let coords: Vec<[f32; 2]> = contour
                    .points
                    .iter()
                    .map(|p| [p.x as f32, p.y as f32])
                    .collect();

                let mut polygon = match Polygon::try_from(coords) {
                    Ok(p) => p.verify(),
                    Err(_) => return None,
                };
                if let Some(x) = self.name() {
                    polygon = polygon.with_name(x);
                }
                if let Some(x) = self.id() {
                    polygon = polygon.with_id(x);
                }
                if let Some(x) = self.confidence() {
                    polygon = polygon.with_confidence(x);
                }
                Some(polygon)
            })
            .collect();

        polygons
    }
}
