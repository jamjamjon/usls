//! Output wrapper for ONNX Runtime SessionOutputs.
//!
//! Provides convenient zero-copy access to inference outputs via index or name.

use half::{bf16, f16};
use ndarray::Array;
use num_traits::{cast, NumCast};
use ort::{session::SessionOutputs, tensor::PrimitiveTensorElementType, tensor::TensorElementType};
use rayon::prelude::*;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use crate::{OrtTensorAttr, XView, X};

/// CachedX serves as a "Physical Type Mirror", holding the owned tensor data extracted
/// from the backend (ORT) in its original physical format.
enum CachedX {
    F32(X<f32>),
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
    fn convert_to<T: NumCast + Send + Sync>(&self) -> Option<(Vec<usize>, Vec<T>)> {
        macro_rules! convert_branch {
            ($tensor:expr) => {{
                let shape = $tensor.0.shape().to_vec();
                let data: Option<Vec<T>> = if $tensor.0.len() > 1024 {
                    $tensor
                        .0
                        .as_slice_memory_order()
                        .map(|s| s.par_iter().map(|&x| cast(x)).collect())
                        .unwrap_or_else(|| {
                            $tensor
                                .0
                                .iter()
                                .copied()
                                .collect::<Vec<_>>()
                                .par_iter()
                                .map(|&x| cast(x))
                                .collect()
                        })
                } else {
                    $tensor
                        .0
                        .as_slice_memory_order()
                        .map(|s| s.iter().map(|&x| cast(x)).collect())
                        .unwrap_or_else(|| $tensor.0.iter().map(|&x| cast(x)).collect())
                };
                data.map(|d| (shape, d))
            }};
        }

        match self {
            Self::F32(t) => convert_branch!(t),
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
                let slice: Vec<bool> =
                    t.0.as_slice_memory_order()
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|| t.0.iter().copied().collect());
                let data: Option<Vec<T>> = slice
                    .par_iter()
                    .map(|&x| {
                        let v = if x { 1i64 } else { 0i64 };
                        cast(v)
                    })
                    .collect();
                data.map(|d| (shape, d))
            }
        }
    }
}

#[derive(Default)]
struct PerOutputCache {
    actual: OnceLock<CachedX>,
    typed: OnceLock<Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
}

/// Xs is a "Type Alignment Layer" wrapping ONNX Runtime's SessionOutputs.
pub struct Xs<'s> {
    outputs: SessionOutputs<'s>,
    metadata: Arc<OrtTensorAttr>,
    caches: Vec<PerOutputCache>,
}

impl<'s> Xs<'s> {
    pub fn new(outputs: SessionOutputs<'s>) -> Self {
        let names: Vec<String> = outputs.keys().map(|s| s.to_string()).collect();
        let len = names.len();
        let metadata = Arc::new(OrtTensorAttr::new(names, vec![], vec![], vec![]));
        let mut caches = Vec::with_capacity(len);
        for _ in 0..len {
            caches.push(PerOutputCache::default());
        }
        Self {
            outputs,
            metadata,
            caches,
        }
    }

    pub fn with_metadata(outputs: SessionOutputs<'s>, metadata: Arc<OrtTensorAttr>) -> Self {
        let len = metadata.names.len();
        let mut caches = Vec::with_capacity(len);
        for _ in 0..len {
            caches.push(PerOutputCache::default());
        }
        Self {
            outputs,
            metadata,
            caches,
        }
    }

    fn typed_cache(&self, index: usize) -> &Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>> {
        self.caches[index]
            .typed
            .get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub fn len(&self) -> usize {
        self.metadata.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.names.is_empty()
    }

    pub fn names(&self) -> &[String] {
        &self.metadata.names
    }

    pub fn get<T>(&self, index: usize) -> Option<XView<'_, T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + Clone + 'static,
    {
        if index >= self.metadata.names.len() {
            return None;
        }
        let name = &self.metadata.names[index];
        self.get_by_name::<T>(name)
    }

    fn get_actual(&self, index: usize) -> Option<&CachedX> {
        if let Some(t) = self.caches[index].actual.get() {
            Some(t)
        } else {
            let name = &self.metadata.names[index];
            let value = self.outputs.get(name)?;
            let ty = *value.data_type();
            let cached: CachedX = match ty {
                TensorElementType::Float32 => CachedX::F32(
                    value
                        .try_extract_array::<f32>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Float16 => CachedX::F16(
                    value
                        .try_extract_array::<f16>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Bfloat16 => CachedX::BF16(
                    value
                        .try_extract_array::<bf16>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Float64 => CachedX::F64(
                    value
                        .try_extract_array::<f64>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Int8 => CachedX::I8(
                    value
                        .try_extract_array::<i8>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Int16 => CachedX::I16(
                    value
                        .try_extract_array::<i16>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Int32 => CachedX::I32(
                    value
                        .try_extract_array::<i32>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Int64 => CachedX::I64(
                    value
                        .try_extract_array::<i64>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Uint8 => CachedX::U8(
                    value
                        .try_extract_array::<u8>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Uint16 => CachedX::U16(
                    value
                        .try_extract_array::<u16>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Uint32 => CachedX::U32(
                    value
                        .try_extract_array::<u32>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Uint64 => CachedX::U64(
                    value
                        .try_extract_array::<u64>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                TensorElementType::Bool => CachedX::Bool(
                    value
                        .try_extract_array::<bool>()
                        .ok()?
                        .to_owned()
                        .into_dyn()
                        .into(),
                ),
                _ => return None,
            };

            let _ = self.caches[index].actual.set(cached);
            self.caches[index].actual.get()
        }
    }

    pub fn get_by_name<T>(&self, name: &str) -> Option<XView<'_, T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + Clone + 'static,
    {
        let index = *self.metadata.name_to_index.get(name)?;
        let value = self.outputs.get(name)?;

        let type_id = TypeId::of::<T>();

        // Check if requested type matches actual ORT output type - zero-copy path
        let actual_type = *value.data_type();
        let requested_type = T::into_tensor_element_type();

        if actual_type == requested_type {
            // Types match - use zero-copy extraction
            if let Ok(arr) = value.try_extract_array::<T>() {
                return Some(XView::from(arr.into_dyn()));
            }
        }

        // Types don't match or extraction failed - use conversion path with caching
        let cache = self.typed_cache(index);
        let mut guard = cache.lock().ok()?;

        // Check if we already have a cached host-owned copy of the requested type
        if let Some(ptr) = guard.get(&type_id).and_then(|any| {
            any.downcast_ref::<Box<X<T>>>()
                .map(|b| (&**b) as *const X<T>)
        }) {
            drop(guard);
            let t: &X<T> = unsafe { &*ptr };
            return Some(XView::from(t.0.view().into_dyn()));
        }

        let actual = self.get_actual(index)?;
        let (shape, data) = actual.convert_to::<T>()?;
        let arr = Array::from_shape_vec(ndarray::IxDyn(&shape), data).ok()?;
        let converted: X<T> = arr.into();
        guard.insert(type_id, Box::new(Box::new(converted)));

        let ptr = guard.get(&type_id).and_then(|any| {
            any.downcast_ref::<Box<X<T>>>()
                .map(|b| (&**b) as *const X<T>)
        })?;

        drop(guard);
        let t: &X<T> = unsafe { &*ptr };
        Some(XView::from(t.0.view().into_dyn()))
    }

    pub fn raw(&self) -> &SessionOutputs<'s> {
        &self.outputs
    }

    pub fn into_inner(self) -> SessionOutputs<'s> {
        self.outputs
    }
}

impl<'s> From<SessionOutputs<'s>> for Xs<'s> {
    fn from(outputs: SessionOutputs<'s>) -> Self {
        Self::new(outputs)
    }
}

pub struct EngineOutputIter<'a, 's> {
    output: &'a Xs<'s>,
    index: usize,
}

impl<'a, 's> Iterator for EngineOutputIter<'a, 's> {
    type Item = &'a ort::value::Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.output.metadata.names.len() {
            let name = &self.output.metadata.names[self.index];
            self.index += 1;
            self.output.outputs.get(name)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.output.metadata.names.len() - self.index;
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
