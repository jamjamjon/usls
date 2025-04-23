use aksr::Builder;
use anyhow::Result;
use image::GrayImage;
use rayon::prelude::*;

use crate::{InstanceMeta, Polygon, Style};

/// Mask: Gray Image.
#[derive(Builder, Default, Clone)]
pub struct Mask {
    mask: GrayImage,
    meta: InstanceMeta,
    style: Option<Style>,
}

// #[derive(Builder, Default, Clone)]
// pub struct Masks(Vec<Mask>);

impl std::fmt::Debug for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mask")
            .field("dimensions", &self.dimensions())
            .field("uid", &self.meta.uid())
            .field("id", &self.meta.id())
            .field("name", &self.meta.name())
            .field("confidence", &self.meta.confidence())
            .finish()
    }
}

impl PartialEq for Mask {
    fn eq(&self, other: &Self) -> bool {
        self.mask == other.mask
    }
}

impl Mask {
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

        polygons
            .into_iter()
            .max_by(|x, y| x.area().total_cmp(&y.area()))
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
                let mut polygon = Polygon::default()
                    .with_points_imageproc(&contour.points)
                    .verify();
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

impl Mask {
    pub fn with_uid(mut self, uid: usize) -> Self {
        self.meta = self.meta.with_uid(uid);
        self
    }
    pub fn with_id(mut self, id: usize) -> Self {
        self.meta = self.meta.with_id(id);
        self
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.meta = self.meta.with_name(name);
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.meta = self.meta.with_confidence(confidence);
        self
    }

    pub fn uid(&self) -> usize {
        self.meta.uid()
    }

    pub fn name(&self) -> Option<&str> {
        self.meta.name()
    }

    pub fn confidence(&self) -> Option<f32> {
        self.meta.confidence()
    }

    pub fn id(&self) -> Option<usize> {
        self.meta.id()
    }
}
