mod bbox;
// mod embedding;
mod keypoint;
mod mask;
mod mbr;
mod polygon;
mod prob;
mod text;
mod x;
mod xs;
mod y;
mod ys;

pub use bbox::Bbox;
pub use keypoint::Keypoint;
pub use mask::Mask;
pub use mbr::Mbr;
pub use polygon::Polygon;
pub use prob::Prob;
pub use text::Text;
pub use x::X;
pub use xs::Xs;
pub use y::Y;
pub use ys::Ys;

pub trait Nms {
    fn iou(&self, other: &Self) -> f32;
    fn confidence(&self) -> f32;
}
