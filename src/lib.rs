//! # usls
//!
//! **usls** is a cross-platform Rust library powered by ONNX Runtime for efficient inference of SOTA vision and vision-language models (typically under 1B parameters).
//!
//! ## 📘 Documentation
//!
//! - [Full Examples](https://github.com/jamjamjon/usls/tree/main/examples)
//! - [GitHub](https://github.com/jamjamjon/usls/tree/main/README.md#github)
//!

#[doc(hidden)]
pub mod config;

#[doc(hidden)]
pub mod dataloader;

#[doc(hidden)]
pub mod models;

#[doc(hidden)]
pub mod ort;

#[doc(hidden)]
pub mod perf;

#[doc(hidden)]
pub mod processor;

#[doc(hidden)]
pub mod results;

#[doc(hidden)]
pub mod tensor;

#[doc(hidden)]
pub mod utils;

#[cfg(feature = "viewer")]
#[macro_use]
#[doc(hidden)]
pub mod viewer;

#[cfg(feature = "annotator")]
#[doc(hidden)]
pub mod annotator;

#[doc(inline)]
pub use {
    config::*, dataloader::*, models::*, ort::*, perf::*, processor::*, results::*, tensor::*,
    utils::*,
};

#[cfg(feature = "annotator")]
pub use annotator::*;

#[cfg(feature = "viewer")]
pub use viewer::*;

/// The name of the current crate.
pub(crate) const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
