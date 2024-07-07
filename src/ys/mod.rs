#![allow(dead_code)]
mod bbox;
mod embedding;
mod keypoint;
mod mask;
mod mbr;
mod polygon;
mod prob;
mod y;

pub use bbox::Bbox;
pub use embedding::Embedding;
pub use keypoint::Keypoint;
pub use mask::Mask;
pub use mbr::Mbr;
pub use polygon::Polygon;
pub use prob::Prob;
pub use y::Y;

pub trait Nms {
    fn iou(&self, other: &Self) -> f32;
    fn confidence(&self) -> f32;
}
