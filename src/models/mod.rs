//! Pre-built models for various vision and vision-language tasks.
//!
//! # Feature Flags
//!
//! - **`vision`**: Pure vision models (detection, segmentation, classification, pose, depth, etc.)
//! - **`vlm`**: Vision-Language models (includes vision + tokenizers)
//! - **`mot`**: Multi-object tracking utilities (ByteTrack)
//!
//! # Usage
//!
//! ```toml
//! # Vision models only (default)
//! usls = { version = "0.1", features = ["vision"] }
//!
//! # Vision + VLM models
//! usls = { version = "0.1", features = ["vlm"] }
//!
//! # All models including tracking
//! usls = { version = "0.1", features = ["all-models"] }
//! ```

// Label names (always loaded, lazy initialization)
pub mod names;

// Vision models
#[cfg(feature = "vision")]
pub mod vision;

// Vision-Language models
#[cfg(feature = "vlm")]
pub mod vlm;

// Multi-object tracking
#[cfg(feature = "mot")]
pub mod mot;

// Re-exports
pub use names::*;

#[cfg(feature = "vision")]
pub use vision::*;

#[cfg(feature = "mot")]
pub use mot::*;
