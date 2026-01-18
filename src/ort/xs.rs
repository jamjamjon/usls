//! Output wrapper for ONNX Runtime SessionOutputs.
//!
//! Provides convenient zero-copy access to inference outputs via index or name.

use half::{bf16, f16};
use ndarray::Array;
use num_traits::{cast, NumCast};
use ort::{
    memory::AllocationDevice, session::SessionOutputs, tensor::PrimitiveTensorElementType,
    tensor::TensorElementType, value::ValueType,
};
use rayon::prelude::*;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use crate::{OrtTensorAttr, XView, X};

#[derive(Default)]
struct PerOutputCache {
    host: OnceLock<Option<ort::value::DynValue>>,
    typed: Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
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
        self.get_by_index::<T>(index)
    }

    #[inline]
    fn get_by_index<T>(&self, index: usize) -> Option<XView<'_, T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + Clone + 'static,
    {
        if let Some(v) = self.caches[index].host.get().and_then(|o| o.as_ref()) {
            if let Ok(arr) = v.try_extract_array::<T>() {
                return Some(XView::from(arr.into_dyn()));
            }
        }

        let v = &self.outputs[index];

        if let Ok(tensor) = v.downcast_ref::<ort::value::DynTensorValueType>() {
            if !tensor.memory_info().is_cpu_accessible() {
                if let Some(host) = self.ensure_host(index) {
                    if let Ok(arr) = host.try_extract_array::<T>() {
                        return Some(XView::from(arr.into_dyn()));
                    }
                }
                return self.get_converted::<T>(index);
            }
        }

        if let Ok(arr) = v.try_extract_array::<T>() {
            return Some(XView::from(arr.into_dyn()));
        }

        if let Some(host) = self.ensure_host(index) {
            if let Ok(arr) = host.try_extract_array::<T>() {
                return Some(XView::from(arr.into_dyn()));
            }
        }

        self.get_converted::<T>(index)
    }

    #[inline]
    fn ensure_host(&self, index: usize) -> Option<&ort::value::DynValue> {
        let cached = self.caches[index]
            .host
            .get_or_init(|| self.transfer_to_host(index));
        cached.as_ref()
    }

    fn transfer_to_host(&self, index: usize) -> Option<ort::value::DynValue> {
        let name = &self.metadata.names[index];
        let value = &self.outputs[index];

        let owned = value.view().try_upgrade().ok()?;
        let tensor = owned.downcast::<ort::value::DynTensorValueType>().ok()?;
        let mem_info = tensor.memory_info();
        if mem_info.is_cpu_accessible() {
            return Some(tensor.into_dyn());
        }

        match tensor.to(AllocationDevice::CPU, 0) {
            Ok(cpu_tensor) => {
                if cpu_tensor.memory_info().is_cpu_accessible() {
                    Some(cpu_tensor.into_dyn())
                } else {
                    tracing::warn!(
                        "tensor.to(CPU) returned non-CPU-accessible memory for '{}': device={:?} device_id={} cpu_accessible={} ",
                        name,
                        cpu_tensor.memory_info().allocation_device(),
                        cpu_tensor.memory_info().device_id(),
                        cpu_tensor.memory_info().is_cpu_accessible(),
                    );
                    tensor
                        .to(AllocationDevice::CUDA_PINNED, 0)
                        .ok()
                        .map(|t| t.into_dyn())
                }
            }
            Err(e) => {
                tracing::error!(
                    "Failed to transfer tensor '{}' to CPU (device={:?} device_id={}): {:?}",
                    name,
                    mem_info.allocation_device(),
                    mem_info.device_id(),
                    e
                );
                tensor
                    .to(AllocationDevice::CUDA_PINNED, 0)
                    .ok()
                    .map(|t| t.into_dyn())
            }
        }
    }

    fn get_converted<T>(&self, index: usize) -> Option<XView<'_, T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + Clone + 'static,
    {
        let type_id = TypeId::of::<T>();
        let mut guard = self.caches[index].typed.lock().ok()?;

        if let Some(ptr) = guard
            .get(&type_id)
            .and_then(|any| any.downcast_ref::<X<T>>().map(|t| t as *const X<T>))
        {
            drop(guard);
            let t: &X<T> = unsafe { &*ptr };
            return Some(XView::from(t.0.view().into_dyn()));
        }

        let value = self.caches[index]
            .host
            .get()
            .and_then(|o| o.as_ref())
            .unwrap_or_else(|| &self.outputs[index]);

        let converted = Self::convert_value::<T>(value)?;
        guard.insert(type_id, Box::new(converted));

        let ptr = guard
            .get(&type_id)
            .and_then(|any| any.downcast_ref::<X<T>>().map(|t| t as *const X<T>))?;
        drop(guard);

        let t: &X<T> = unsafe { &*ptr };
        Some(XView::from(t.0.view().into_dyn()))
    }

    fn convert_value<T>(value: &ort::value::DynValue) -> Option<X<T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + Clone + 'static,
    {
        let ValueType::Tensor { ty, .. } = value.dtype() else {
            return None;
        };

        macro_rules! cast_tensor {
            ($src:ty) => {{
                let (shape, data) = value.try_extract_tensor::<$src>().ok()?;
                let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                let data: Option<Vec<T>> = if data.len() > 1024 {
                    data.par_iter().map(|&x| cast(x)).collect()
                } else {
                    data.iter().map(|&x| cast(x)).collect()
                };
                let arr = Array::from_shape_vec(ndarray::IxDyn(&shape), data?).ok()?;
                Some(arr.into())
            }};
        }

        match *ty {
            TensorElementType::Float32 => cast_tensor!(f32),
            TensorElementType::Float16 => cast_tensor!(f16),
            TensorElementType::Bfloat16 => cast_tensor!(bf16),
            TensorElementType::Float64 => cast_tensor!(f64),
            TensorElementType::Int8 => cast_tensor!(i8),
            TensorElementType::Int16 => cast_tensor!(i16),
            TensorElementType::Int32 => cast_tensor!(i32),
            TensorElementType::Int64 => cast_tensor!(i64),
            TensorElementType::Uint8 => cast_tensor!(u8),
            TensorElementType::Uint16 => cast_tensor!(u16),
            TensorElementType::Uint32 => cast_tensor!(u32),
            TensorElementType::Uint64 => cast_tensor!(u64),
            TensorElementType::Bool => {
                let (shape, data) = value.try_extract_tensor::<bool>().ok()?;
                let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                let data: Option<Vec<T>> = if data.len() > 1024 {
                    data.par_iter()
                        .map(|&x| {
                            let v = if x { 1i64 } else { 0i64 };
                            cast(v)
                        })
                        .collect()
                } else {
                    data.iter()
                        .map(|&x| {
                            let v = if x { 1i64 } else { 0i64 };
                            cast(v)
                        })
                        .collect()
                };
                let arr = Array::from_shape_vec(ndarray::IxDyn(&shape), data?).ok()?;
                Some(arr.into())
            }
            _ => None,
        }
    }

    pub fn get_by_name<T>(&self, name: &str) -> Option<XView<'_, T>>
    where
        T: PrimitiveTensorElementType + NumCast + Send + Sync + Clone + 'static,
    {
        let index = *self.metadata.name_to_index.get(name)?;
        self.get_by_index::<T>(index)
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
