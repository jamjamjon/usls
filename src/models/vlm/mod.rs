//! Vision-Language Models (VLMs) that understand both images and text.
//!
//! All models are compiled when the `vlm` feature is enabled.
//!
//! # Models
//!
//! - **BLIP**: Image captioning and VQA
//! - **CLIP**: Contrastive Language-Image Pre-training
//! - **FastVLM**: Fast vision-language model
//! - **Florence2**: Microsoft's multimodal model
//! - **GroundingDINO**: Open-set object detection with text prompts
//! - **Moondream2**: Lightweight VLM
//! - **OWLv2**: Open-vocabulary object detection
//! - **SAM3**: Segment Anything with text prompts
//! - **SmolVLM**: Small vision-language model
//! - **TrOCR**: Transformer-based OCR
//! - **YOLO-E**: YOLO with text embeddings

// Pipeline
pub mod pipeline;

// Models
pub mod blip;
pub mod clip;
pub mod fastvlm;
pub mod florence2;
pub mod grounding_dino;
pub mod moondream2;
pub mod owl;
pub mod sam3;
pub mod smolvlm;
pub mod trocr;
pub mod yoloe;

// Re-exports
pub use blip::*;
pub use clip::*;
pub use fastvlm::*;
pub use florence2::*;
pub use grounding_dino::*;
pub use moondream2::*;
pub use owl::*;
pub use pipeline::*;
pub use sam3::*;
pub use smolvlm::*;
pub use trocr::*;
pub use yoloe::*;
