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
//! - **YOLO-E**: YOLO with text embeddings, Real-Time Seeing Anything

// Pipeline
mod pipeline;

// Models
mod blip;
mod clip;
mod fastvlm;
mod florence2;
mod grounding_dino;
mod moondream2;
mod owl;
mod sam3_image;
mod smolvlm;
mod trocr;
mod yoloe;

// Re-exports
pub use blip::*;
pub use clip::*;
pub use fastvlm::*;
pub use florence2::*;
pub use grounding_dino::*;
pub use moondream2::*;
pub use owl::*;
// pub use pipeline::*;
pub use sam3_image::*;
pub use smolvlm::*;
pub use trocr::*;
pub use yoloe::*;
