//! Unified label catalogs used across models.
//!
//! Large sets are embedded as text and lazily loaded at first use to
//! minimize compile time and memory while keeping a stable API.

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
