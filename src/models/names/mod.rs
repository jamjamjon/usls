//! Unified label catalogs used across models.
//!
//! All name catalogs are always loaded. Large sets use lazy initialization
//! to minimize memory usage until first access.
//!
//! # Available Catalogs
//!
//! - `coco`: COCO dataset labels (80/91 classes, keypoints)
//! - `imagenet`: ImageNet-1K classification labels
//! - `dota`: DOTA aerial object detection labels
//! - `object365`: Objects365 detection labels
//! - `hands`: Hand keypoint labels

pub mod coco;
pub mod dota;
pub mod hands;
pub mod imagenet;
pub mod object365;

pub use coco::*;
pub use dota::*;
pub use hands::*;
pub use imagenet::*;
pub use object365::*;
