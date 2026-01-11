use crate::{AnyResStrategy, CropMode, PadMode, ResizeMode};

#[derive(Debug, Clone)]
pub enum ImageTransform {
    Resize(ResizeMode),
    Pad(PadMode),
    Crop(CropMode),
    AnyRes(AnyResStrategy),
}

/// Information about image transformation including source and destination dimensions.
#[derive(aksr::Builder, Debug, Clone, Default)]
pub struct ImageTransformInfo {
    pub width_src: u32,
    pub height_src: u32,
    pub width_dst: u32,
    pub height_dst: u32,
    pub height_scale: f32,
    pub width_scale: f32,
    pub height_pad: f32,
    pub width_pad: f32,
}

impl ImageTransformInfo {
    /// Merge two transformation infos (chain transformations).
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            width_src: if self.width_src > 0 {
                self.width_src
            } else {
                other.width_src
            },
            height_src: if self.height_src > 0 {
                self.height_src
            } else {
                other.height_src
            },
            width_dst: if other.width_dst > 0 {
                other.width_dst
            } else {
                self.width_dst
            },
            height_dst: if other.height_dst > 0 {
                other.height_dst
            } else {
                self.height_dst
            },
            height_scale: if other.height_scale > 0. {
                self.height_scale * other.height_scale
            } else {
                self.height_scale
            },
            width_scale: if other.width_scale > 0. {
                self.width_scale * other.width_scale
            } else {
                self.width_scale
            },
            height_pad: self.height_pad + other.height_pad,
            width_pad: self.width_pad + other.width_pad,
        }
    }
}
