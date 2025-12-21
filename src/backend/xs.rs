//! Output wrapper for ONNX Runtime SessionOutputs.
//!
//! Provides convenient zero-copy access to inference outputs via index or name.

use anyhow::Result;
use half::{bf16, f16};
use ndarray::Array;
use num_traits::{cast, NumCast, ToPrimitive};
use ort::{
    session::SessionOutputs, tensor::PrimitiveTensorElementType, tensor::TensorElementType,
    value::ValueType,
};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use crate::{XView, X};

enum CachedX {
    F16(X<f16>),
    BF16(X<bf16>),
    F64(X<f64>),
    I8(X<i8>),
    I16(X<i16>),
    I32(X<i32>),
    I64(X<i64>),
    U8(X<u8>),
    U16(X<u16>),
    U32(X<u32>),
    U64(X<u64>),
    Bool(X<bool>),
}

impl CachedX {
    /// Convert cached tensor directly to target type T using num-traits.
    /// Returns (shape, data) for constructing X<T>.
    fn convert_to<T: NumCast>(&self, name: &str, index: usize) -> Result<(Vec<usize>, Vec<T>)> {
        macro_rules! convert_branch {
            ($tensor:expr) => {{
                let shape = $tensor.0.shape().to_vec();
                let mut data = Vec::with_capacity($tensor.0.len());
                for x in $tensor.0.iter() {
                    let y: T = cast(*x).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Failed to cast output '{}' (index {}) to {} (value: {:?})",
                            name,
                            index,
                            std::any::type_name::<T>(),
                            x.to_f64()
                        )
                    })?;
                    data.push(y);
                }
                Ok((shape, data))
            }};
        }

        match self {
            Self::F16(t) => convert_branch!(t),
            Self::BF16(t) => convert_branch!(t),
            Self::F64(t) => convert_branch!(t),
            Self::I8(t) => convert_branch!(t),
            Self::I16(t) => convert_branch!(t),
            Self::I32(t) => convert_branch!(t),
            Self::I64(t) => convert_branch!(t),
            Self::U8(t) => convert_branch!(t),
            Self::U16(t) => convert_branch!(t),
            Self::U32(t) => convert_branch!(t),
            Self::U64(t) => convert_branch!(t),
            Self::Bool(t) => {
                let shape = t.0.shape().to_vec();
                let mut data = Vec::with_capacity(t.0.len());
                for &x in t.0.iter() {
                    let v = if x { 1i64 } else { 0i64 };
                    let y: T = cast(v).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Failed to cast output '{}' (index {}) bool to {}",
                            name,
                            index,
                            std::any::type_name::<T>()
                        )
                    })?;
                    data.push(y);
                }
                Ok((shape, data))
            }
        }
    }
}

/// A wrapper around ORT `SessionOutputs` that provides convenient zero-copy access patterns.
///
/// Supports both index-based (`output.get(0)`) and name-based (`output.get_by_name("logits")`) access,
/// returning `XView<T>` for zero-copy tensor access.
///
/// # Example
/// ```ignore
/// let output = engine.run(ort_inputs![x]?)?;
/// let bboxes = output.get::<f32>(0)?;           // XView<f32> - zero-copy
/// let logits = output.get_by_name::<f32>("logits")?;  // XView<f32> - zero-copy
/// ```
pub struct Xs<'s> {
    outputs: SessionOutputs<'s>,
    names: Vec<String>,
    name_to_index: HashMap<String, usize>,
    cached_actual: Vec<OnceLock<CachedX>>,
    #[allow(clippy::type_complexity)]
    cached_typed: Vec<OnceLock<Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>>,
}

impl<'s> Xs<'s> {
    /// Create a new Xs wrapper from SessionOutputs.
    pub fn new(outputs: SessionOutputs<'s>) -> Self {
        let names: Vec<String> = outputs.keys().map(|s| s.to_string()).collect();
        let name_to_index: HashMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let cached_actual = (0..names.len()).map(|_| OnceLock::new()).collect();
        let cached_typed = (0..names.len()).map(|_| OnceLock::new()).collect();
        Self {
            outputs,
            names,
            name_to_index,
            cached_actual,
            cached_typed,
        }
    }

    fn typed_cache(&self, index: usize) -> &Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>> {
        self.cached_typed[index].get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Get the number of outputs.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Check if there are no outputs.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Get output names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Get output by index as XView (zero-copy).
    ///
    /// Returns an error if the index is out of bounds or extraction fails.
    pub fn get<T>(&self, index: usize) -> Result<XView<'_, T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + 'static,
    {
        if index >= self.names.len() {
            anyhow::bail!(
                "Output index {} out of bounds (total: {})",
                index,
                self.names.len()
            );
        }
        let name = &self.names[index];
        self.get_by_name::<T>(name)
    }

    /// Get output by name as XView (zero-copy).
    ///
    /// Returns an error if the name is not found or extraction fails.
    pub fn get_by_name<T>(&self, name: &str) -> Result<XView<'_, T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + 'static,
    {
        let index = *self
            .name_to_index
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Output '{}' not found", name))?;
        let value = self
            .outputs
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Output '{}' not found", name))?;

        // Fast path: typed extraction succeeds -> zero-copy.
        if let Ok(arr) = value.try_extract_array::<T>() {
            return Ok(XView::from(arr));
        }

        // Ensure this is a tensor.
        match value.dtype() {
            ValueType::Tensor { .. } => {}
            other => {
                anyhow::bail!(
                    "Output '{}' (index {}) is not a tensor (ValueType: {:?})",
                    name,
                    index,
                    other
                );
            }
        }

        // Fallback: extract actual dtype (owned) once, then convert to requested T (owned) and cache.
        let ty = *value.data_type();

        let actual = if let Some(t) = self.cached_actual[index].get() {
            t
        } else {
            let cached: CachedX = match ty {
                TensorElementType::Float16 => CachedX::F16(
                    value
                        .try_extract_array::<f16>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Bfloat16 => CachedX::BF16(
                    value
                        .try_extract_array::<bf16>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Float64 => CachedX::F64(
                    value
                        .try_extract_array::<f64>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Int8 => CachedX::I8(
                    value
                        .try_extract_array::<i8>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Int16 => CachedX::I16(
                    value
                        .try_extract_array::<i16>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Int32 => CachedX::I32(
                    value
                        .try_extract_array::<i32>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Int64 => CachedX::I64(
                    value
                        .try_extract_array::<i64>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Uint8 => CachedX::U8(
                    value
                        .try_extract_array::<u8>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Uint16 => CachedX::U16(
                    value
                        .try_extract_array::<u16>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Uint32 => CachedX::U32(
                    value
                        .try_extract_array::<u32>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Uint64 => CachedX::U64(
                    value
                        .try_extract_array::<u64>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Bool => CachedX::Bool(
                    value
                        .try_extract_array::<bool>()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                other => {
                    anyhow::bail!(
                        "Output '{}' (index {}) has unsupported tensor dtype {:?} for auto-conversion",
                        name,
                        index,
                        other
                    );
                }
            };

            let _ = self.cached_actual[index].set(cached);
            self.cached_actual[index].get().ok_or_else(|| {
                anyhow::anyhow!("Failed to cache actual output '{}' (index {})", name, index)
            })?
        };

        let type_id = TypeId::of::<T>();
        let cache = self.typed_cache(index);
        let mut guard = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Typed cache mutex poisoned"))?;

        // If we already converted this output to T, return a view to the cached X<T>.
        if let Some(ptr) = guard.get(&type_id).and_then(|any| {
            any.downcast_ref::<Box<X<T>>>()
                .map(|b| (&**b) as *const X<T>)
        }) {
            drop(guard);
            let t: &X<T> = unsafe { &*ptr };
            return Ok(XView::from(t.0.view().into_dyn()));
        }

        // Convert directly from actual dtype to T (no f32 intermediate).
        let (shape, data) = actual.convert_to::<T>(name, index)?;
        let arr = Array::from_shape_vec(ndarray::IxDyn(&shape), data).map_err(|e| {
            anyhow::anyhow!(
                "Failed to build converted tensor for output '{}' (index {}) to {}: {}",
                name,
                index,
                std::any::type_name::<T>(),
                e
            )
        })?;
        let converted: X<T> = arr.into();
        guard.insert(type_id, Box::new(Box::new(converted)));

        let ptr = guard
            .get(&type_id)
            .and_then(|any| {
                any.downcast_ref::<Box<X<T>>>()
                    .map(|b| (&**b) as *const X<T>)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to cache converted output '{}' (index {})",
                    name,
                    index
                )
            })?;

        drop(guard);
        let t: &X<T> = unsafe { &*ptr };
        Ok(XView::from(t.0.view().into_dyn()))
    }

    /// Try to get output by index, returns None if extraction fails.
    pub fn try_get<T>(&self, index: usize) -> Option<XView<'_, T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + 'static,
    {
        self.get::<T>(index).ok()
    }

    /// Try to get output by name, returns None if extraction fails.
    pub fn try_get_by_name<T>(&self, name: &str) -> Option<XView<'_, T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + 'static,
    {
        self.get_by_name::<T>(name).ok()
    }

    /// Get the underlying SessionOutputs reference.
    pub fn raw(&self) -> &SessionOutputs<'s> {
        &self.outputs
    }

    /// Consume self and return the underlying SessionOutputs.
    pub fn into_inner(self) -> SessionOutputs<'s> {
        self.outputs
    }
}

impl<'s> From<SessionOutputs<'s>> for Xs<'s> {
    fn from(outputs: SessionOutputs<'s>) -> Self {
        Self::new(outputs)
    }
}

/// Iterator over output values as raw ORT `Value` references.
pub struct EngineOutputIter<'a, 's> {
    output: &'a Xs<'s>,
    index: usize,
}

impl<'a, 's> Iterator for EngineOutputIter<'a, 's> {
    type Item = &'a ort::value::Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.output.names.len() {
            let name = &self.output.names[self.index];
            self.index += 1;
            self.output.outputs.get(name)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.output.names.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, 's> ExactSizeIterator for EngineOutputIter<'a, 's> {}

impl<'a, 's> IntoIterator for &'a Xs<'s> {
    type Item = &'a ort::value::Value;
    type IntoIter = EngineOutputIter<'a, 's>;

    fn into_iter(self) -> Self::IntoIter {
        EngineOutputIter {
            output: self,
            index: 0,
        }
    }
}
