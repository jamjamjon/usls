use anyhow::Result;
use std::ops::{Deref, DerefMut};

use crate::{Config, Engines, Y};

/// The core Model trait for unified model inference.
///
/// This trait provides a standardized interface for all models, supporting:
/// - Single-engine models (YOLO, RTDETR, DepthAnything, etc.)
/// - Multi-engine models (SAM, Florence2, SmolVLM, etc.)
/// - Various input types (images only, images + text, images + prompts, etc.)
///
/// # Design Philosophy
///
/// The key insight is that `Model` does NOT own `Engine`. Instead, engines are
/// passed in via the `Engines` parameter. This design:
/// 1. Eliminates borrow conflicts (engine.run() needs &mut self)
/// 2. Enables zero-copy postprocessing (can borrow Xs<'_> while accessing model fields)
/// 3. Supports multi-engine models naturally
/// 4. Makes testing easier (can mock engines)
///
/// # Example
///
/// ```ignore
/// impl Model for RFDETR {
///     type Input<'a> = &'a [Image];
///
///     fn build(config: Config) -> Result<(Self, Engines)> {
///         let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
///         let model = Self { ... };
///         Ok((model, Engines::from(engine)))
///     }
///     
///     fn run(&mut self, engines: &mut Engines, images: &[Image]) -> Result<Vec<Y>> {
///         let x = self.processor.process(images)?;
///         let ys = engines.run(&Module::Model, inputs![x]?)?;
///         self.postprocess(&ys)
///     }
/// }
///
/// // Usage:
/// let mut model = RFDETR::new(config)?;  // Returns Runtime<RFDETR>
/// let batch = model.batch;               // Direct field access via Deref
/// let results = model.run(&images)?;
/// ```
pub trait Model: Sized {
    /// The input type for this model.
    ///
    /// Common patterns:
    /// - `&'a [Image]` - Most vision models
    /// - `(&'a [Image], &'a str)` - VLM models with text input
    /// - `(&'a [Image], &'a [SamPrompt])` - SAM-like models with prompts
    type Input<'a>;

    /// Creates a Runtime from config.
    ///
    /// Default implementation: calls `build()` and wraps in `Runtime`.
    fn new(config: Config) -> Result<Runtime<Self>> {
        let (model, engines) = Self::build(config)?;
        Ok(Runtime::new(model, engines))
    }

    /// Builds the model and its engines from config.
    ///
    /// This is the only method besides `run` that you need to implement.
    fn build(config: Config) -> Result<(Self, Engines)>;

    /// Runs the complete inference pipeline.
    ///
    /// This method should:
    /// 1. Preprocess the input
    /// 2. Run inference on one or more engines
    /// 3. Postprocess the outputs
    fn run(&mut self, _engines: &mut Engines, _input: Self::Input<'_>) -> Result<Vec<Y>> {
        unimplemented!("run() not implemented for this model")
    }

    /// Encodes input data into embeddings.
    ///
    /// This method should be implemented by models that support encoding functionality,
    /// such as vision-language models that need to encode images and text together.
    ///
    /// # Arguments
    ///
    /// * `engines` - Mutable reference to the engines for inference
    /// * `input` - The input data to encode (type varies by model)
    ///
    /// # Returns
    ///
    /// Returns encoded embeddings as `Y` (typically tensor data)
    fn encode(&mut self, _engines: &mut Engines, _input: Self::Input<'_>) -> Result<Y> {
        unimplemented!("encode() not implemented for this model")
    }

    /// Encodes images into visual embeddings.
    ///
    /// This method should be implemented by models that support image encoding,
    /// such as vision models or vision-language models that need visual features.
    ///
    /// # Arguments
    ///
    /// * `engines` - Mutable reference to the engines for inference
    /// * `input` - Slice of images to encode
    ///
    /// # Returns
    ///
    /// Returns visual embeddings as `Y` (typically tensor data)
    fn encode_images(&mut self, _engines: &mut Engines, _input: &[crate::Image]) -> Result<Y> {
        unimplemented!("encode_images() not implemented for this model")
    }

    /// Encodes text into textual embeddings.
    ///
    /// This method should be implemented by models that support text encoding,
    /// such as language models or vision-language models that need text features.
    ///
    /// # Arguments
    ///
    /// * `engines` - Mutable reference to the engines for inference
    /// * `input` - Slice of text strings to encode
    ///
    /// # Returns
    ///
    /// Returns textual embeddings as `Y` (typically tensor data)
    fn encode_texts(&mut self, _engines: &mut Engines, _input: &[&str]) -> Result<Y> {
        unimplemented!("encode_texts() not implemented for this model")
    }

    /// Gets the batch size of the model.
    fn batch(&self) -> usize;

    /// Gets the model specification/name.
    fn spec(&self) -> &str;
}

/// Runtime combining a Model with its Engines.
///
/// This is the main entry point for using models. It combines:
/// - The model logic (preprocessing, postprocessing, parameters)
/// - The engine(s) for inference
///
/// Runtime implements `Deref` and `DerefMut` to allow direct access to model fields.
///
#[derive(Debug)]
pub struct Runtime<M: Model> {
    inner: M,
    engines: Engines,
}

impl<M: Model> Runtime<M> {
    /// Creates a new Runtime from a model and engines.
    pub fn new(model: M, engines: Engines) -> Self {
        Self {
            inner: model,
            engines,
        }
    }

    /// Runs inference with the given input.
    pub fn run(&mut self, input: M::Input<'_>) -> Result<Vec<Y>> {
        self.inner.run(&mut self.engines, input)
    }

    /// Forward pass with the given input.
    pub fn forward(&mut self, input: M::Input<'_>) -> Result<Vec<Y>> {
        self.inner.run(&mut self.engines, input)
    }

    /// Encode input with the model.
    pub fn encode(&mut self, input: M::Input<'_>) -> Result<Y> {
        self.inner.encode(&mut self.engines, input)
    }

    /// Encode images with the model.
    pub fn encode_images(&mut self, input: &[crate::Image]) -> Result<Y> {
        self.inner.encode_images(&mut self.engines, input)
    }

    /// Encode texts with the model.
    pub fn encode_texts(&mut self, input: &[&str]) -> Result<Y> {
        self.inner.encode_texts(&mut self.engines, input)
    }

    /// Gets a reference to the engines.
    pub fn engines(&self) -> &Engines {
        &self.engines
    }

    /// Gets a mutable reference to the engines.
    pub fn engines_mut(&mut self) -> &mut Engines {
        &mut self.engines
    }

    /// Provides mutable access to both the model and engines.
    ///
    /// This is useful for methods that need to modify the model state
    /// while also using the engines (e.g., encoding methods in YOLOE).
    pub fn parts_mut(&mut self) -> (&mut M, &mut Engines) {
        (&mut self.inner, &mut self.engines)
    }
}

/// Deref to allow direct access to model fields.
impl<M: Model> Deref for Runtime<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// DerefMut to allow mutable access to model fields.
impl<M: Model> DerefMut for Runtime<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
