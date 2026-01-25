use anyhow::Result;
use std::collections::HashMap;

use crate::{Engine, EngineInputs, Module, Xs};

/// A collection of engines for multi-module models.
///
/// This structure provides unified management of multiple ONNX engines,
/// supporting both single-engine models (like YOLO) and multi-engine models
/// (like SAM, Florence2, SmolVLM).
#[derive(Debug, Default)]
pub struct Engines {
    engines: HashMap<Module, Engine>,
}

impl Engines {
    /// Creates a new empty Engines collection.
    pub fn new() -> Self {
        Self {
            engines: HashMap::new(),
        }
    }

    /// Creates an Engines collection with a single engine.
    pub fn with_engine(module: Module, engine: Engine) -> Self {
        let mut engines = HashMap::new();
        engines.insert(module, engine);
        Self { engines }
    }

    /// Inserts an engine for the specified module.
    pub fn insert(&mut self, module: Module, engine: Engine) -> Option<Engine> {
        self.engines.insert(module, engine)
    }

    /// Gets a reference to the engine for the specified module.
    pub fn get(&self, module: &Module) -> Option<&Engine> {
        self.engines.get(module)
    }

    /// Gets a mutable reference to the engine for the specified module.
    pub fn get_mut(&mut self, module: &Module) -> Option<&mut Engine> {
        self.engines.get_mut(module)
    }

    /// Checks if an engine exists for the specified module.
    pub fn contains(&self, module: &Module) -> bool {
        self.engines.contains_key(module)
    }

    /// Returns the number of engines.
    pub fn len(&self) -> usize {
        self.engines.len()
    }

    /// Returns true if there are no engines.
    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }

    /// Runs inference on the specified module's engine.
    ///
    /// # Arguments
    /// * `module` - The module identifier
    /// * `inputs` - The input tensors
    ///
    /// # Returns
    /// * `Result<Xs<'_>>` - The output tensors wrapped in `Xs`
    pub fn run<'e, 'a, 'i, 'v: 'i, const N: usize>(
        &'e mut self,
        module: &Module,
        inputs: impl Into<EngineInputs<'a, 'i, 'v, N>>,
    ) -> Result<Xs<'e>> {
        self.engines
            .get_mut(module)
            .ok_or_else(|| anyhow::anyhow!("Engine not found for module: {module:?}"))?
            .run(inputs)
    }

    #[deprecated(
        note = "Use engines.run(module, &xany) instead. This preserves zero-copy CUDA semantics."
    )]
    pub fn run_xany(&mut self, module: &Module, input: crate::XAny) -> Result<Xs<'_>> {
        self.run(module, crate::inputs![&input]?)
    }

    /// Iterates over all module-engine pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Module, &Engine)> {
        self.engines.iter()
    }

    /// Iterates over all module-engine pairs mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&Module, &mut Engine)> {
        self.engines.iter_mut()
    }
}

impl From<Engine> for Engines {
    fn from(engine: Engine) -> Self {
        Self::with_engine(Module::Model, engine)
    }
}

impl FromIterator<(Module, Engine)> for Engines {
    fn from_iter<T: IntoIterator<Item = (Module, Engine)>>(iter: T) -> Self {
        Self {
            engines: iter.into_iter().collect(),
        }
    }
}
