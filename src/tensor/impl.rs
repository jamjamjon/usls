//! Tensor module containing all tensor-related functionality

use anyhow::{anyhow, Result};
use half::{bf16, f16};
use ndarray::{ArcArray, Array, Axis, IntoDimension, Ix2, IxDyn};
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::iter::{TensorDimIter, TensorDimIterMut};
use super::macros::{dtype_tensor_self_update_match, dtype_tensor_transform_match};
use super::slice::{IntoSliceSpec, SliceOrIndex};

use crate::{DType, TensorElement, TensorView, TensorViewMut};

/// Multi-dtype tensor with optimized storage and zero-copy operations
#[derive(Clone, PartialEq)]
pub struct Tensor {
    pub data: DTypeTensor,
    pub dtype: DType,
    pub uid: usize,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)?;

        Ok(())
    }
}

/// Enum for different data types with optimized memory layout
#[derive(Debug, Clone, PartialEq)]
pub enum DTypeTensor {
    F32(ArcArray<f32, IxDyn>),
    F64(ArcArray<f64, IxDyn>),
    F16(ArcArray<f16, IxDyn>),
    Bf16(ArcArray<bf16, IxDyn>),
    I8(ArcArray<i8, IxDyn>),
    I16(ArcArray<i16, IxDyn>),
    I32(ArcArray<i32, IxDyn>),
    I64(ArcArray<i64, IxDyn>),
    U8(ArcArray<u8, IxDyn>),
    U16(ArcArray<u16, IxDyn>),
    U32(ArcArray<u32, IxDyn>),
    U64(ArcArray<u64, IxDyn>),
    Bool(ArcArray<bool, IxDyn>),
}

impl DTypeTensor {
    /// Generic clamp with compile-time type safety
    /// This ensures T matches the actual tensor element type at compile time
    pub fn clamp_typed<T: TensorElement>(&self, min_val: T, max_val: T) -> Option<Array<T, IxDyn>> {
        use std::any::TypeId;

        match self {
            DTypeTensor::F32(arr) if TypeId::of::<T>() == TypeId::of::<f32>() => {
                let min_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&min_val) };
                let max_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&max_val) };
                let result = arr.mapv(|x| x.clamp(min_f32, max_f32)).into_dyn();
                Some(unsafe { std::mem::transmute::<Array<f32, IxDyn>, Array<T, IxDyn>>(result) })
            }
            DTypeTensor::F64(arr) if TypeId::of::<T>() == TypeId::of::<f64>() => {
                let min_f64 = unsafe { std::mem::transmute_copy::<T, f64>(&min_val) };
                let max_f64 = unsafe { std::mem::transmute_copy::<T, f64>(&max_val) };
                let result = arr.mapv(|x| x.clamp(min_f64, max_f64)).into_dyn();
                Some(unsafe { std::mem::transmute::<Array<f64, IxDyn>, Array<T, IxDyn>>(result) })
            }
            DTypeTensor::I32(arr) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                let min_i32 = unsafe { std::mem::transmute_copy::<T, i32>(&min_val) };
                let max_i32 = unsafe { std::mem::transmute_copy::<T, i32>(&max_val) };
                let result = arr.mapv(|x| x.clamp(min_i32, max_i32)).into_dyn();
                Some(unsafe { std::mem::transmute::<Array<i32, IxDyn>, Array<T, IxDyn>>(result) })
            }
            DTypeTensor::I64(arr) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                let min_i64 = unsafe { std::mem::transmute_copy::<T, i64>(&min_val) };
                let max_i64 = unsafe { std::mem::transmute_copy::<T, i64>(&max_val) };
                let result = arr.mapv(|x| x.clamp(min_i64, max_i64)).into_dyn();
                Some(unsafe { std::mem::transmute::<Array<i64, IxDyn>, Array<T, IxDyn>>(result) })
            }
            DTypeTensor::U8(arr) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                let min_u8 = unsafe { std::mem::transmute_copy::<T, u8>(&min_val) };
                let max_u8 = unsafe { std::mem::transmute_copy::<T, u8>(&max_val) };
                let result = arr.mapv(|x| x.clamp(min_u8, max_u8)).into_dyn();
                Some(unsafe { std::mem::transmute::<Array<u8, IxDyn>, Array<T, IxDyn>>(result) })
            }
            DTypeTensor::U32(arr) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                let min_u32 = unsafe { std::mem::transmute_copy::<T, u32>(&min_val) };
                let max_u32 = unsafe { std::mem::transmute_copy::<T, u32>(&max_val) };
                let result = arr.mapv(|x| x.clamp(min_u32, max_u32)).into_dyn();
                Some(unsafe { std::mem::transmute::<Array<u32, IxDyn>, Array<T, IxDyn>>(result) })
            }
            // For types without native clamp or type mismatch, return None
            _ => None,
        }
    }
}

impl Default for Tensor {
    #[inline]
    fn default() -> Self {
        Self::zeros(0)
    }
}

impl<T: TensorElement> From<Array<T, IxDyn>> for Tensor {
    #[inline]
    fn from(x: Array<T, IxDyn>) -> Self {
        Self {
            data: T::into_dtype_tensor(x.into_shared()),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        }
    }
}

impl<T: TensorElement> From<ArcArray<T, IxDyn>> for Tensor {
    #[inline]
    fn from(x: ArcArray<T, IxDyn>) -> Self {
        Self {
            data: T::into_dtype_tensor(x),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        }
    }
}

impl<T: TensorElement> From<Vec<T>> for Tensor {
    #[inline]
    fn from(x: Vec<T>) -> Self {
        Self::from_vec(x)
    }
}

impl<T: TensorElement> TryFrom<Vec<Vec<T>>> for Tensor {
    type Error = anyhow::Error;

    fn try_from(x: Vec<Vec<T>>) -> Result<Self> {
        if x.is_empty() {
            return Ok(Self::zeros(vec![0, 0]));
        }

        let rows = x.len();
        let cols = x[0].len();
        let flat: Vec<T> = x.into_iter().flatten().collect();

        Self::from_shape_vec(vec![rows, cols], flat)
    }
}

// TODO: temporary support for u32, u32
impl TryFrom<Vec<(u32, u32)>> for Tensor {
    type Error = anyhow::Error;

    fn try_from(x: Vec<(u32, u32)>) -> Result<Self> {
        let flat: Vec<f32> = x
            .into_iter()
            .flat_map(|(a, b)| [a as f32, b as f32])
            .collect();
        Self::from_shape_vec(vec![flat.len() / 2, 2], flat)
    }
}

// TODO: add support for other dtype
impl Deref for Tensor {
    type Target = ArcArray<f32, IxDyn>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match &self.data {
            DTypeTensor::F32(arr) => arr,
            _ => panic!(
                "Tensor deref failed: expected F32 type, got {:?}. Use explicit methods like shape(), dtype(), etc. instead of dereferencing.",
                self.dtype
            ),
        }
    }
}

impl Tensor {
    /// Generate unique identifier for tensor instances
    pub fn generate_uid() -> usize {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    /// Create tensor from vector with automatic type inference
    #[inline]
    pub fn from_vec<T: TensorElement>(data: Vec<T>) -> Self {
        let array = ArcArray::from_vec(data).into_dyn();
        Self {
            data: T::into_dtype_tensor(array),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        }
    }

    /// Create tensor from shape and vector with automatic type inference
    #[inline]
    pub fn from_shape_vec<D: IntoDimension, T: TensorElement>(
        shape: D,
        data: Vec<T>,
    ) -> Result<Self> {
        let array = ArcArray::from_shape_vec(shape, data)?;
        Ok(Self {
            data: T::into_dtype_tensor(array.into_dyn()),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        })
    }

    #[inline]
    pub fn from_slice<T: TensorElement>(data: &[T]) -> Self {
        let array = ArcArray::from_vec(data.to_vec()).into_dyn();
        Self {
            data: T::into_dtype_tensor(array),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        }
    }

    #[inline]
    pub fn from_shape_slice<D: IntoDimension, T: TensorElement>(
        shape: D,
        data: Vec<T>,
    ) -> Result<Self> {
        let array = ArcArray::from_shape_vec(shape, data.to_vec())?;
        Ok(Self {
            data: T::into_dtype_tensor(array.into_dyn()),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        })
    }

    /// Create zeros tensor with specified shape (default f32)
    #[inline]
    pub fn zeros<D: IntoDimension>(shape: D) -> Self {
        Self::zeros_with_dtype(shape, DType::Fp32)
    }

    /// Create zeros tensor with specific dtype
    #[inline]
    pub fn zeros_with_dtype<D: IntoDimension>(shape: D, dtype: DType) -> Self {
        match dtype {
            DType::Fp32 => Self::zeros_typed::<f32, _>(shape),
            DType::Fp64 => Self::zeros_typed::<f64, _>(shape),
            DType::Fp16 => Self::zeros_typed::<f16, _>(shape),
            DType::Bf16 => Self::zeros_typed::<bf16, _>(shape),
            DType::Int8 => Self::zeros_typed::<i8, _>(shape),
            DType::Int16 => Self::zeros_typed::<i16, _>(shape),
            DType::Int32 => Self::zeros_typed::<i32, _>(shape),
            DType::Int64 => Self::zeros_typed::<i64, _>(shape),
            DType::Uint8 => Self::zeros_typed::<u8, _>(shape),
            DType::Uint16 => Self::zeros_typed::<u16, _>(shape),
            DType::Uint32 => Self::zeros_typed::<u32, _>(shape),
            DType::Uint64 => Self::zeros_typed::<u64, _>(shape),
            _ => panic!("Unsupported dtype: {:?}", dtype),
        }
    }

    /// Create zeros tensor for specific type
    #[inline]
    pub fn zeros_typed<T: TensorElement + Default, D: IntoDimension>(shape: D) -> Self {
        let array = ArcArray::default(shape).into_dyn();
        Self {
            data: T::into_dtype_tensor(array),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        }
    }

    /// Create ones tensor with specified shape (default f32)
    #[inline]
    pub fn ones<D: IntoDimension>(shape: D) -> Self {
        Self::ones_with_dtype(shape, DType::Fp32)
    }

    /// Create ones tensor with specific dtype
    #[inline]
    pub fn ones_with_dtype<D: IntoDimension>(shape: D, dtype: DType) -> Self {
        match dtype {
            DType::Fp32 => Self::ones_typed::<f32, _>(shape),
            DType::Fp64 => Self::ones_typed::<f64, _>(shape),
            DType::Fp16 => Self::ones_typed::<f16, _>(shape),
            DType::Bf16 => Self::ones_typed::<bf16, _>(shape),
            DType::Int8 => Self::ones_typed::<i8, _>(shape),
            DType::Int16 => Self::ones_typed::<i16, _>(shape),
            DType::Int32 => Self::ones_typed::<i32, _>(shape),
            DType::Int64 => Self::ones_typed::<i64, _>(shape),
            DType::Uint8 => Self::ones_typed::<u8, _>(shape),
            DType::Uint16 => Self::ones_typed::<u16, _>(shape),
            DType::Uint32 => Self::ones_typed::<u32, _>(shape),
            DType::Uint64 => Self::ones_typed::<u64, _>(shape),
            _ => panic!("Unsupported dtype: {:?}", dtype),
        }
    }

    /// Create ones tensor for specific type
    #[inline]
    pub fn ones_typed<T: TensorElement, D: IntoDimension>(shape: D) -> Self {
        let array = ArcArray::from_elem(shape, T::from_f32(1.0)).into_dyn();
        Self {
            data: T::into_dtype_tensor(array),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        }
    }

    /// Create tensor filled with specified value (default f32)
    #[inline]
    pub fn full<D: IntoDimension>(shape: D, value: f32) -> Self {
        Self::full_typed::<f32, _>(shape, value)
    }

    /// Create tensor filled with specific value for specific type
    #[inline]
    pub fn full_typed<T: TensorElement, D: IntoDimension>(shape: D, value: T) -> Self {
        let array = ArcArray::from_elem(shape, value).into_dyn();
        Self {
            data: T::into_dtype_tensor(array),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        }
    }

    /// Create zeros tensor with same shape and dtype as another tensor
    #[inline]
    pub fn zeros_like(&self) -> Self {
        Self::zeros_with_dtype(self.dims(), self.dtype)
    }

    /// Create ones tensor with same shape and dtype as another tensor
    #[inline]
    pub fn ones_like(&self) -> Self {
        Self::ones_with_dtype(self.dims(), self.dtype)
    }

    /// Convert tensor to specified dtype
    pub fn to_dtype(&self, target_dtype: DType) -> Result<Self> {
        if self.dtype == target_dtype {
            return Ok(self.clone());
        }

        // Macro to generate type conversion cases
        macro_rules! convert_dtype {
            ($data:expr, $target_variant:ident, $target_type:ty, $target_dtype:expr) => {{
                let converted = $data.mapv(|x| x as $target_type);
                Ok(Self {
                    data: DTypeTensor::$target_variant(converted.into_shared()),
                    dtype: $target_dtype,
                    uid: Self::generate_uid(),
                })
            }};
            // Special case for Bool conversion
            ($data:expr, $target_variant:ident, $target_type:ty, $target_dtype:expr, bool) => {{
                let converted = $data.mapv(|x| {
                    if x {
                        1 as $target_type
                    } else {
                        0 as $target_type
                    }
                });
                Ok(Self {
                    data: DTypeTensor::$target_variant(converted.into_shared()),
                    dtype: $target_dtype,
                    uid: Self::generate_uid(),
                })
            }};
        }

        match (&self.data, target_dtype) {
            // Float conversions (most common)
            (DTypeTensor::F32(data), DType::Fp64) => convert_dtype!(data, F64, f64, DType::Fp64),
            (DTypeTensor::F64(data), DType::Fp32) => convert_dtype!(data, F32, f32, DType::Fp32),

            // F32 to F16/BF16 conversions
            (DTypeTensor::F32(data), DType::Fp16) => {
                let converted = data.mapv(f16::from_f32);
                Ok(Self {
                    data: DTypeTensor::F16(converted.into_shared()),
                    dtype: DType::Fp16,
                    uid: Self::generate_uid(),
                })
            }
            (DTypeTensor::F32(data), DType::Bf16) => {
                let converted = data.mapv(bf16::from_f32);
                Ok(Self {
                    data: DTypeTensor::Bf16(converted.into_shared()),
                    dtype: DType::Bf16,
                    uid: Self::generate_uid(),
                })
            }

            // F16/BF16 to F32 conversions
            (DTypeTensor::F16(data), DType::Fp32) => {
                let converted = data.mapv(|x| x.to_f32());
                Ok(Self {
                    data: DTypeTensor::F32(converted.into_shared()),
                    dtype: DType::Fp32,
                    uid: Self::generate_uid(),
                })
            }
            (DTypeTensor::Bf16(data), DType::Fp32) => {
                let converted = data.mapv(|x| x.to_f32());
                Ok(Self {
                    data: DTypeTensor::F32(converted.into_shared()),
                    dtype: DType::Fp32,
                    uid: Self::generate_uid(),
                })
            }

            // Float to integer conversions
            (DTypeTensor::F32(data), DType::Int32) => convert_dtype!(data, I32, i32, DType::Int32),
            (DTypeTensor::F32(data), DType::Int64) => convert_dtype!(data, I64, i64, DType::Int64),
            (DTypeTensor::F32(data), DType::Uint8) => convert_dtype!(data, U8, u8, DType::Uint8),
            (DTypeTensor::F64(data), DType::Int32) => convert_dtype!(data, I32, i32, DType::Int32),
            (DTypeTensor::F64(data), DType::Int64) => convert_dtype!(data, I64, i64, DType::Int64),

            // Integer to float conversions (upcast safe)
            (DTypeTensor::I8(data), DType::Fp32) => convert_dtype!(data, F32, f32, DType::Fp32),
            (DTypeTensor::I16(data), DType::Fp32) => convert_dtype!(data, F32, f32, DType::Fp32),
            (DTypeTensor::I32(data), DType::Fp32) => convert_dtype!(data, F32, f32, DType::Fp32),
            (DTypeTensor::I32(data), DType::Fp64) => convert_dtype!(data, F64, f64, DType::Fp64),
            (DTypeTensor::I64(data), DType::Fp32) => convert_dtype!(data, F32, f32, DType::Fp32),
            (DTypeTensor::I64(data), DType::Fp64) => convert_dtype!(data, F64, f64, DType::Fp64),

            (DTypeTensor::U8(data), DType::Fp32) => convert_dtype!(data, F32, f32, DType::Fp32),
            (DTypeTensor::U8(data), DType::Fp64) => convert_dtype!(data, F64, f64, DType::Fp64),
            (DTypeTensor::U8(data), DType::Int32) => convert_dtype!(data, I32, i32, DType::Int32),
            (DTypeTensor::U8(data), DType::Int64) => convert_dtype!(data, I64, i64, DType::Int64),
            (DTypeTensor::U16(data), DType::Fp32) => convert_dtype!(data, F32, f32, DType::Fp32),
            (DTypeTensor::U32(data), DType::Fp32) => convert_dtype!(data, F32, f32, DType::Fp32),
            (DTypeTensor::U64(data), DType::Fp32) => convert_dtype!(data, F32, f32, DType::Fp32),

            // Integer size conversions (safe upcasts)
            (DTypeTensor::I32(data), DType::Int64) => convert_dtype!(data, I64, i64, DType::Int64),
            (DTypeTensor::I64(data), DType::Int32) => convert_dtype!(data, I32, i32, DType::Int32),

            // Bool conversions
            (DTypeTensor::Bool(data), DType::Fp32) => {
                let converted = data.mapv(|x| if x { 1.0f32 } else { 0.0f32 });
                Ok(Self {
                    data: DTypeTensor::F32(converted.into_shared()),
                    dtype: DType::Fp32,
                    uid: Self::generate_uid(),
                })
            }
            (DTypeTensor::Bool(data), DType::Int32) => {
                convert_dtype!(data, I32, i32, DType::Int32, bool)
            }
            (DTypeTensor::Bool(data), DType::Int64) => {
                convert_dtype!(data, I64, i64, DType::Int64, bool)
            }
            (DTypeTensor::Bool(data), DType::Uint8) => {
                convert_dtype!(data, U8, u8, DType::Uint8, bool)
            }

            // Float to Bool conversions
            (DTypeTensor::F32(data), DType::Bool) => {
                let converted = data.mapv(|x| x != 0.0f32);
                Ok(Self {
                    data: DTypeTensor::Bool(converted.into_shared()),
                    dtype: DType::Bool,
                    uid: Self::generate_uid(),
                })
            }
            (DTypeTensor::F64(data), DType::Bool) => {
                let converted = data.mapv(|x| x != 0.0f64);
                Ok(Self {
                    data: DTypeTensor::Bool(converted.into_shared()),
                    dtype: DType::Bool,
                    uid: Self::generate_uid(),
                })
            }

            _ => anyhow::bail!(
                "Type conversion from {:?} to {:?} not implemented or unsafe",
                self.dtype,
                target_dtype
            ),
        }
    }

    /// Create random tensor with uniform distribution in range [low, high)
    /// Type is automatically inferred from the low and high parameters
    /// Supports all numeric types except bool
    pub fn rand<T, D>(low: T, high: T, shape: D) -> Result<Self>
    where
        T: TensorElement + rand_distr::uniform::SampleUniform,
        D: IntoDimension,
    {
        use ndarray_rand::RandomExt;
        use rand_distr::Uniform;

        let dist = Uniform::new(low, high);
        let array = ndarray::Array::random(shape, dist).into_dyn();

        Ok(Self {
            data: T::into_dtype_tensor(array.into_shared()),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        })
    }

    /// Create random tensor with standard normal distribution (mean=0, std=1)
    /// Supports floating point types
    pub fn randn<T: TensorElement, D: IntoDimension>(shape: D) -> Result<Self> {
        use ndarray_rand::RandomExt;
        use rand_distr::StandardNormal;
        use std::any::TypeId;

        // Handle each floating point type specifically
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let array = ndarray::Array::random(shape, StandardNormal).into_dyn();
            let result = unsafe {
                std::mem::transmute::<
                    ndarray::Array<f32, ndarray::IxDyn>,
                    ndarray::Array<T, ndarray::IxDyn>,
                >(array)
            };
            return Ok(Self {
                data: T::into_dtype_tensor(result.into_shared()),
                dtype: T::dtype(),
                uid: Self::generate_uid(),
            });
        }

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let array = ndarray::Array::random(shape, StandardNormal).into_dyn();
            let result = unsafe {
                std::mem::transmute::<
                    ndarray::Array<f64, ndarray::IxDyn>,
                    ndarray::Array<T, ndarray::IxDyn>,
                >(array)
            };
            return Ok(Self {
                data: T::into_dtype_tensor(result.into_shared()),
                dtype: T::dtype(),
                uid: Self::generate_uid(),
            });
        }

        // For non-floating point types, generate f32 normal and convert
        let array: ndarray::Array<f32, ndarray::IxDyn> =
            ndarray::Array::random(shape, StandardNormal).into_dyn();
        let converted = array.mapv(|x| T::from_f32(x));
        Ok(Self {
            data: T::into_dtype_tensor(converted.into_shared()),
            dtype: T::dtype(),
            uid: Self::generate_uid(),
        })
    }
}

impl Tensor {
    /// Clamp values to range with optional bounds (lite builder)
    /// If min is None, uses the minimum value for the data type
    /// If max is None, uses the maximum value for the data type
    #[inline]
    pub fn clamp<T: TensorElement + Copy + PartialOrd + 'static>(
        self,
        min: Option<T>,
        max: Option<T>,
    ) -> Result<Self> {
        let min_val = min.unwrap_or_else(|| T::min_value().expect("Type should have min value"));
        let max_val = max.unwrap_or_else(|| T::max_value().expect("Type should have max value"));

        let data = self
            .data
            .clamp_typed(min_val, max_val)
            .ok_or_else(|| anyhow::anyhow!("Clamp operation not supported for this tensor type"))?;

        Ok(data.into())
    }

    /// Clamp values to be non-negative (unsigned) based on tensor's dtype
    /// Automatically selects the appropriate zero value for the tensor's data type
    #[inline]
    pub fn clamp_unsigned(self) -> Result<Self> {
        use crate::DType;
        match self.dtype() {
            DType::Fp32 => self.clamp(Some(0.0f32), None),
            DType::Fp64 => self.clamp(Some(0.0f64), None),
            DType::Fp16 => self.clamp(Some(half::f16::from_f32(0.0)), None),
            DType::Bf16 => self.clamp(Some(half::bf16::from_f32(0.0)), None),
            DType::Int8 => self.clamp(Some(0i8), None),
            DType::Int16 => self.clamp(Some(0i16), None),
            DType::Int32 => self.clamp(Some(0i32), None),
            DType::Int64 => self.clamp(Some(0i64), None),
            DType::Uint8 => self.clamp(Some(0u8), None),
            DType::Uint16 => self.clamp(Some(0u16), None),
            DType::Uint32 => self.clamp(Some(0u32), None),
            DType::Uint64 => self.clamp(Some(0u64), None),
            _ => Err(anyhow::anyhow!(
                "Clamp unsigned operation not supported for dtype: {:?}",
                self.dtype()
            )),
        }
    }
}

impl Tensor {
    /// Reshape tensor (lite builder)
    pub fn reshape<D: IntoDimension>(mut self, shape: D) -> Result<Self> {
        dtype_tensor_self_update_match!(self, arr, {
            arr.clone()
                .into_shape_with_order(shape)?
                .into_dyn()
                .into_shared()
        })
    }

    /// Transpose tensor (lite builder)
    pub fn transpose(&self) -> Result<Self> {
        dtype_tensor_transform_match!(
            &self.data,
            arr,
            { arr.clone().reversed_axes().into_shared() },
            self.dtype
        )
    }

    /// Reverse the order of axes (dimensions)
    ///
    /// This method reverses the order of tensor dimensions. For example,
    /// a tensor with shape [2, 3, 4] becomes [4, 3, 2] after calling this method.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Self>` containing the tensor with reversed axes or an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 3, 4]);
    /// let reversed = tensor.reversed_axes().unwrap();
    /// assert_eq!(reversed.shape(), &[4, 3, 2]);
    /// ```
    pub fn reversed_axes(&self) -> Result<Self> {
        dtype_tensor_transform_match!(
            &self.data,
            arr,
            { arr.clone().reversed_axes().into_shared() },
            self.dtype
        )
    }

    /// Permute dimensions (lite builder)
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                if axes.len() != arr.shape().len() {
                    anyhow::bail!(
                        "permute: Shape length mismatch. Expected: {}, got: {}. Target shape: {:?}, provided shape: {:?}.",
                        arr.shape().len(),
                        axes.len(),
                        arr.shape(),
                        axes
                    );
                }
                let permuted = arr.clone().permuted_axes(axes.to_vec()).into_dyn();
                Ok(Self {
                    data: DTypeTensor::F32(permuted.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("permute currently only supports F32 tensors"),
        }
    }

    /// Insert new axis (lite builder)
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                if arr.shape().len() < axis {
                    anyhow::bail!(
                        "insert_axis: The specified axis position {} exceeds the maximum shape length {}.",
                        axis,
                        arr.shape().len()
                    );
                }
                let expanded = arr.clone().insert_axis(ndarray::Axis(axis));
                Ok(Self {
                    data: DTypeTensor::F32(expanded.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("unsqueeze currently only supports F32 tensors"),
        }
    }

    /// Insert a new axis at the specified position
    pub fn insert_axis(&self, axis: usize) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                if arr.shape().len() < axis {
                    anyhow::bail!(
                        "insert_axis: The specified axis position {} exceeds the maximum shape length {}.",
                        axis,
                        arr.shape().len()
                    );
                }
                let expanded = arr.clone().insert_axis(ndarray::Axis(axis));
                Ok(Self {
                    data: DTypeTensor::F32(expanded.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("insert_axis currently only supports F32 tensors"),
        }
    }

    /// Repeat tensor along specified axis (lite builder)
    pub fn repeat(&self, axis: usize, times: usize) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                if axis >= arr.ndim() {
                    anyhow::bail!("Index {} is out of bounds with size {}.", axis, arr.ndim());
                }
                let mut dim = arr.shape().to_vec();
                dim[axis] = times;
                match arr.broadcast(dim.as_slice()) {
                    Some(broadcasted) => Ok(Self {
                        data: DTypeTensor::F32(broadcasted.to_owned().into_dyn().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    }),
                    None => anyhow::bail!(
                        "Failed to broadcast. Shape: {:?}, dim: {:?}",
                        arr.shape(),
                        dim
                    ),
                }
            }
            _ => anyhow::bail!("repeat currently only supports F32 tensors"),
        }
    }

    /// Remove single-dimensional entries (lite builder)
    pub fn squeeze(&self) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let mut shape: Vec<usize> =
                    arr.shape().iter().filter(|&&d| d != 1).copied().collect();
                if shape.is_empty() {
                    shape.push(1); // Keep at least one dimension
                }
                let squeezed = arr.clone().into_shape_with_order(shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::F32(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("squeeze currently only supports F32 tensors"),
        }
    }

    /// Remove a specific dimension of size 1
    ///
    /// This method removes a dimension at the specified axis if it has size 1.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension index to remove
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` with the specified dimension removed
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dimension index is out of bounds
    /// - The dimension size is not 1
    pub fn squeeze_dim(&self, dim: usize) -> Result<Self> {
        if dim >= self.ndim() {
            anyhow::bail!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                self.ndim()
            );
        }

        if self.shape()[dim] != 1 {
            anyhow::bail!("Cannot squeeze dimension {} with size {}, only dimensions of size 1 can be squeezed", dim, self.shape()[dim]);
        }

        let mut new_shape = Vec::with_capacity(self.ndim() - 1);
        for (i, &size) in self.shape().iter().enumerate() {
            if i != dim {
                new_shape.push(size);
            }
        }

        match &self.data {
            DTypeTensor::F32(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::F32(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::F64(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::F64(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::I32(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::I32(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::I64(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::I64(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::U8(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::U8(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::U32(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::U32(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::Bool(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::Bool(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::F16(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::F16(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::Bf16(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::Bf16(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::I8(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::I8(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::I16(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::I16(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::U16(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::U16(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::U64(arr) => {
                let squeezed = arr.clone().into_shape_with_order(new_shape)?.into_dyn();
                Ok(Self {
                    data: DTypeTensor::U64(squeezed.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
        }
    }

    /// Broadcast to new shape (lite builder)
    pub fn broadcast_to<D: IntoDimension + std::fmt::Debug>(&self, shape: D) -> Result<Self> {
        let target_shape = shape.into_dimension();

        macro_rules! broadcast_impl {
            ($arr:expr, $variant:ident) => {
                match $arr.broadcast(target_shape.clone()) {
                    Some(broadcasted) => Ok(Self {
                        data: DTypeTensor::$variant(
                            broadcasted.to_owned().into_dyn().into_shared(),
                        ),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    }),
                    None => anyhow::bail!(
                        "Failed to broadcast. Shape: {:?}, target: {:?}",
                        $arr.shape(),
                        target_shape
                    ),
                }
            };
        }

        match &self.data {
            DTypeTensor::F32(arr) => broadcast_impl!(arr, F32),
            DTypeTensor::F64(arr) => broadcast_impl!(arr, F64),
            DTypeTensor::F16(arr) => broadcast_impl!(arr, F16),
            DTypeTensor::Bf16(arr) => broadcast_impl!(arr, Bf16),
            DTypeTensor::I8(arr) => broadcast_impl!(arr, I8),
            DTypeTensor::I16(arr) => broadcast_impl!(arr, I16),
            DTypeTensor::I32(arr) => broadcast_impl!(arr, I32),
            DTypeTensor::I64(arr) => broadcast_impl!(arr, I64),
            DTypeTensor::U8(arr) => broadcast_impl!(arr, U8),
            DTypeTensor::U16(arr) => broadcast_impl!(arr, U16),
            DTypeTensor::U32(arr) => broadcast_impl!(arr, U32),
            DTypeTensor::U64(arr) => broadcast_impl!(arr, U64),
            DTypeTensor::Bool(arr) => broadcast_impl!(arr, Bool),
        }
    }
}

// === Normalization Operations (Lite Builder) ===
impl Tensor {
    /// Normalize to [0, 1] range (lite builder)
    pub fn normalize_01(&self) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let mut owned = arr.to_owned();
                let min_val = owned.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = owned.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let range = max_val - min_val;
                if range > 0.0 {
                    owned.par_mapv_inplace(|x| (x - min_val) / range);
                }
                Ok(Self {
                    data: DTypeTensor::F32(owned.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("normalize_01 currently only supports F32 tensors"),
        }
    }

    /// Normalize to specified range (lite builder)
    pub fn normalize(&self, min_val: f32, max_val: f32) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                if min_val >= max_val {
                    anyhow::bail!(
                        "Invalid range in normalize: min ({}) must be less than max ({}).",
                        min_val,
                        max_val
                    );
                }
                let mut owned = arr.to_owned();
                let range = max_val - min_val;
                owned.par_mapv_inplace(|x| (x - min_val) / range);
                Ok(Self {
                    data: DTypeTensor::F32(owned.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("normalize currently only supports F32 tensors"),
        }
    }

    /// Convert to U8 tensor (lite builder)
    pub fn to_u8(&self) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let u8_data = arr.mapv(|x| x.clamp(0.0, 255.0) as u8);
                Ok(Self {
                    data: DTypeTensor::U8(u8_data.into_shared()),
                    dtype: DType::Uint8,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::U8(_) => Ok(self.clone()), // Already U8
            _ => anyhow::bail!("to_u8 conversion not implemented for this dtype"),
        }
    }

    /// Standardize using mean and std (lite builder)
    pub fn standardize(&self, mean: &[f32], std: &[f32], dim: usize) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let mut owned = arr.to_owned();

                // Check if dimensions match
                if dim >= owned.ndim() {
                    anyhow::bail!(
                        "Dimension {} out of bounds for tensor with {} dimensions",
                        dim,
                        owned.ndim()
                    );
                }

                let shape = owned.shape();
                if mean.len() != shape[dim] || std.len() != shape[dim] {
                    anyhow::bail!("Mean and std length must match dimension size");
                }

                // Apply standardization along the specified dimension
                for (i, mut lane) in owned.axis_iter_mut(ndarray::Axis(dim)).enumerate() {
                    let m = mean[i];
                    let s = std[i];
                    if s == 0.0 {
                        anyhow::bail!(
                            "Cannot standardize with zero standard deviation at index {}",
                            i
                        );
                    }
                    lane.par_mapv_inplace(|x| (x - m) / s);
                }

                Ok(Self {
                    data: DTypeTensor::F32(owned.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("standardize currently only supports F32 tensors"),
        }
    }

    /// Z-score normalization (lite builder)
    pub fn zscore(&self, _dim: Option<usize>) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let mut owned = arr.to_owned();
                let mean = owned.mean().unwrap();
                let variance = owned.mapv(|x| (x - mean).powi(2)).mean().unwrap();
                let std = variance.sqrt();
                if std > 0.0 {
                    owned.par_mapv_inplace(|x| (x - mean) / std);
                }
                Ok(Self {
                    data: DTypeTensor::F32(owned.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("zscore currently only supports F32 tensors"),
        }
    }
}

// === Advanced Operations ===
impl Tensor {
    /// Matrix multiplication for 2D tensors
    ///
    /// Performs matrix multiplication between two 2D tensors using optimized operations.
    /// Both tensors must be 2D and have compatible dimensions for multiplication.
    /// Currently only supports F32 tensors.
    ///
    /// # Arguments
    /// * `other` - The right-hand side tensor for multiplication
    ///
    /// # Returns
    /// * Result containing the product tensor or an error
    ///
    /// # Errors
    /// * Returns error if tensors are not 2D
    /// * Returns error if dimensions are incompatible (A.cols != B.rows)
    /// * Returns error if data types are not F32
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// // Create two 2D matrices
    /// let a = Tensor::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let b = Tensor::from_shape_vec(vec![3, 2], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    ///
    /// // Perform matrix multiplication: [2,3] × [3,2] = [2,2]
    /// let result = a.matmul(&b).unwrap();
    /// assert_eq!(result.shape(), &[2, 2]);
    /// ```
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.ndim() != 2 || other.ndim() != 2 {
            anyhow::bail!(
                "matmul requires 2D tensors, got {}D and {}D",
                self.ndim(),
                other.ndim()
            );
        }

        match (&self.data, &other.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                let a_2d = a.as_standard_layout().into_dimensionality::<Ix2>()?;
                let b_2d = b.as_standard_layout().into_dimensionality::<Ix2>()?;

                if a_2d.shape()[1] != b_2d.shape()[0] {
                    anyhow::bail!(
                        "Incompatible shapes for matmul: {:?} and {:?}",
                        a_2d.shape(),
                        b_2d.shape()
                    );
                }

                let result = a_2d.dot(&b_2d);
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("matmul currently only supports F32 tensors"),
        }
    }

    /// Dot product with transpose (returns 2D tensor)
    ///
    /// Computes the dot product of the first tensor with the transpose of the second tensor.
    /// Both tensors must be 2D. Supports all numeric data types.
    ///
    /// # Arguments
    /// * `other` - The tensor to transpose and multiply with
    ///
    /// # Returns
    /// * Result containing the product tensor or an error
    ///
    /// # Errors
    /// * Returns error if tensors are not 2D
    /// * Returns error if dimensions are incompatible (A.cols != B.cols)
    /// * Returns error if data types don't match
    /// * Returns error if data types are not numeric
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// // Create two 2D matrices
    /// let a = Tensor::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let b = Tensor::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    ///
    /// // Compute A × B^T: [2,3] × [3,2] = [2,2]
    /// let result = a.dot(&b).unwrap();
    /// assert_eq!(result.shape(), &[2, 2]);
    /// ```
    pub fn dot(&self, other: &Self) -> Result<Self> {
        // Check dimensions
        if self.ndim() != 2 || other.ndim() != 2 {
            anyhow::bail!(
                "dot requires 2D matrices, got {}D and {}D",
                self.ndim(),
                other.ndim()
            );
        }

        match (&self.data, &other.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                let a = a.as_standard_layout().into_dimensionality::<Ix2>()?;
                let b = b.as_standard_layout().into_dimensionality::<Ix2>()?;

                // Check compatibility for A × B^T: A[m,k] × B^T[k,n] where B is [n,k]
                if a.shape()[1] != b.shape()[1] {
                    anyhow::bail!(
                        "Incompatible dimensions for dot: A{:?} × B^T{:?}, expected A[m,k] × B[n,k]",
                        a.shape(),
                        b.shape()
                    );
                }

                let result = a.dot(&b.t()).into_dyn();
                Ok(result.into())
            }
            (DTypeTensor::F64(a), DTypeTensor::F64(b)) => {
                let a = a.as_standard_layout().into_dimensionality::<Ix2>()?;
                let b = b.as_standard_layout().into_dimensionality::<Ix2>()?;

                if a.shape()[1] != b.shape()[1] {
                    anyhow::bail!(
                        "Incompatible dimensions for dot: A{:?} × B^T{:?}, expected A[m,k] × B[n,k]",
                        a.shape(),
                        b.shape()
                    );
                }

                let result = a.dot(&b.t()).into_dyn();
                Ok(result.into())
            }
            (DTypeTensor::I32(a), DTypeTensor::I32(b)) => {
                let a = a.as_standard_layout().into_dimensionality::<Ix2>()?;
                let b = b.as_standard_layout().into_dimensionality::<Ix2>()?;

                if a.shape()[1] != b.shape()[1] {
                    anyhow::bail!(
                        "Incompatible dimensions for dot: A{:?} × B^T{:?}, expected A[m,k] × B[n,k]",
                        a.shape(),
                        b.shape()
                    );
                }

                let result = a.dot(&b.t()).into_dyn();
                Ok(result.into())
            }
            (DTypeTensor::I64(a), DTypeTensor::I64(b)) => {
                let a = a.as_standard_layout().into_dimensionality::<Ix2>()?;
                let b = b.as_standard_layout().into_dimensionality::<Ix2>()?;

                if a.shape()[1] != b.shape()[1] {
                    anyhow::bail!(
                        "Incompatible dimensions for dot: A{:?} × B^T{:?}, expected A[m,k] × B[n,k]",
                        a.shape(),
                        b.shape()
                    );
                }

                let result = a.dot(&b.t()).into_dyn();
                Ok(result.into())
            }
            _ => anyhow::bail!(
                "dot requires matching numeric data types, got {:?} and {:?}",
                self.dtype(),
                other.dtype()
            ),
        }
    }

    /// Softmax along specified dimension
    pub fn softmax(&self, dim: usize) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                if dim >= arr.ndim() {
                    anyhow::bail!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        dim,
                        arr.ndim()
                    );
                }

                let mut owned = arr.to_owned();

                // Apply softmax along the specified dimension
                // We need to iterate over all other dimensions and apply softmax to slices along the target dimension
                let shape = owned.shape().to_vec();
                let mut indices = vec![0; shape.len()];

                // Generate all possible index combinations for dimensions other than `dim`
                let total_slices = shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != dim)
                    .map(|(_, &size)| size)
                    .product::<usize>();

                for slice_idx in 0..total_slices {
                    // Calculate the indices for this slice
                    let mut temp_idx = slice_idx;
                    for (i, &size) in shape.iter().enumerate() {
                        if i != dim {
                            indices[i] = temp_idx % size;
                            temp_idx /= size;
                        }
                    }

                    // Extract the slice along the target dimension
                    let mut slice_data = Vec::with_capacity(shape[dim]);
                    for j in 0..shape[dim] {
                        indices[dim] = j;
                        slice_data.push(owned[&indices[..]]);
                    }

                    // Apply softmax to this slice
                    let max_val = slice_data
                        .iter()
                        .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

                    // Subtract max and compute exp
                    for val in &mut slice_data {
                        *val = (*val - max_val).exp();
                    }

                    // Compute sum and normalize
                    let sum: f32 = slice_data.iter().sum();
                    if sum == 0.0 {
                        anyhow::bail!("Softmax sum is zero, cannot normalize");
                    }

                    for val in &mut slice_data {
                        *val /= sum;
                    }

                    // Write back the normalized values
                    for (j, &val) in slice_data.iter().enumerate() {
                        indices[dim] = j;
                        owned[&indices[..]] = val;
                    }
                }

                Ok(Self {
                    data: DTypeTensor::F32(owned.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("softmax currently only supports F32 tensors"),
        }
    }

    /// Concatenate tensors along dimension
    pub fn cat(tensors: &[Self], dim: usize) -> Result<Self> {
        Self::concat(tensors, dim)
    }

    /// Concatenate tensors along dimension (alias for cat)
    pub fn concat(tensors: &[Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("Cannot concatenate empty tensor list");
        }

        let arrays: Vec<Array<f32, IxDyn>> = tensors
            .iter()
            .map(|t| match &t.data {
                DTypeTensor::F32(arr) => arr.to_owned(),
                _ => panic!("cat currently only supports F32 tensors"),
            })
            .collect();

        // Validate axis bounds
        if let Some(first) = arrays.first() {
            if dim >= first.ndim() {
                anyhow::bail!("Axis {} out of bounds for concatenation", dim);
            }
        }

        // Validate shape compatibility
        if arrays.len() > 1 {
            let first_shape = arrays[0].shape();
            for (i, arr) in arrays.iter().enumerate().skip(1) {
                for (dim_idx, (&dim1, &dim2)) in
                    first_shape.iter().zip(arr.shape().iter()).enumerate()
                {
                    if dim_idx != dim && dim1 != dim2 {
                        anyhow::bail!(
                            "Shape mismatch at tensor {} dimension {}: {} vs {}",
                            i,
                            dim_idx,
                            dim1,
                            dim2
                        );
                    }
                }
            }
        }

        let views: Vec<_> = arrays.iter().map(|arr| arr.view()).collect();
        let result = ndarray::concatenate(ndarray::Axis(dim), &views)?;
        Ok(result.into())
    }

    /// Concatenate this tensor with another tensor along dimension
    pub fn concatenate(&self, other: &Self, dim: usize) -> Result<Self> {
        Self::concat(&[self.clone(), other.clone()], dim)
    }

    /// Split tensor along dimension
    pub fn split(self, split_size: usize, dim: usize) -> Result<Vec<Self>> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let shape = arr.shape();
                if dim >= shape.len() {
                    anyhow::bail!(
                        "Dimension {} out of bounds for tensor with {} dimensions",
                        dim,
                        shape.len()
                    );
                }

                let total_size = shape[dim];
                let num_splits = (total_size + split_size - 1) / split_size; // Ceiling division
                let mut results = Vec::with_capacity(num_splits);

                for i in 0..num_splits {
                    let start = i * split_size;
                    let end = (start + split_size).min(total_size);

                    let slice =
                        arr.slice_axis(ndarray::Axis(dim), ndarray::Slice::from(start..end));
                    results.push(slice.to_owned().into());
                    // results.push(Self::from_arc_array(slice.to_owned().into_shared()));
                }

                Ok(results)
            }
            _ => anyhow::bail!("split currently only supports F32 tensors"),
        }
    }

    /// Compute norm along specified axis
    ///
    /// # Arguments
    /// * `ord` - Order of the norm. 1 for L1, 2 for L2, other values for Lp norm
    /// * `axis` - Axis along which to compute the norm. If None, compute global norm
    /// * `keepdims` - If true, the reduced axes are left in the result as dimensions with size one
    pub fn norm(&self, ord: i8, axis: Option<usize>, keepdims: bool) -> Result<Self> {
        match &self.data {
            DTypeTensor::F16(_) => {
                // Convert F16 to F32 for norm computation, then convert back to F16
                let f32_tensor = self.to_f32()?;
                let result = f32_tensor.norm(ord, axis, keepdims)?;
                Ok(result.to_f16()?)
            }
            DTypeTensor::F32(arr) => match axis {
                Some(ax) => {
                    if ax >= arr.ndim() {
                        anyhow::bail!(
                            "Axis {} out of bounds for tensor with {} dimensions",
                            ax,
                            arr.ndim()
                        );
                    }

                    let result = match ord {
                        1 => {
                            // L1 norm: sum of absolute values
                            arr.mapv(|x| x.abs()).sum_axis(ndarray::Axis(ax))
                        }
                        2 => {
                            // L2 norm: sqrt of sum of squares
                            arr.mapv(|x| x * x)
                                .sum_axis(ndarray::Axis(ax))
                                .mapv(|x| x.sqrt())
                        }
                        p if p > 0 => {
                            // Lp norm: (sum of |x|^p)^(1/p)
                            let p_f32 = p as f32;
                            arr.mapv(|x| x.abs().powf(p_f32))
                                .sum_axis(ndarray::Axis(ax))
                                .mapv(|x| x.powf(1.0 / p_f32))
                        }
                        _ => anyhow::bail!("Norm order must be positive, got {}", ord),
                    };

                    let mut tensor: Tensor = result.into_dyn().into();

                    // If keepdims is true, insert axis back to maintain dimensionality
                    if keepdims {
                        tensor = tensor.insert_axis(ax)?;
                    }

                    Ok(tensor)
                }
                None => {
                    let norm = match ord {
                        1 => {
                            // L1 norm: sum of absolute values
                            arr.mapv(|x| x.abs()).sum()
                        }
                        2 => {
                            // L2 norm: sqrt of sum of squares
                            arr.mapv(|x| x * x).sum().sqrt()
                        }
                        p if p > 0 => {
                            // Lp norm: (sum of |x|^p)^(1/p)
                            let p_f32 = p as f32;
                            arr.mapv(|x| x.abs().powf(p_f32)).sum().powf(1.0 / p_f32)
                        }
                        _ => anyhow::bail!("Norm order must be positive, got {}", ord),
                    };

                    let mut tensor: Tensor = ndarray::Array0::from_elem((), norm).into_dyn().into();

                    // If keepdims is true for global norm, maintain original dimensionality with all axes as 1
                    if keepdims {
                        let original_ndim = arr.ndim();
                        for i in 0..original_ndim {
                            tensor = tensor.insert_axis(i)?;
                        }
                    }

                    Ok(tensor)
                }
            },
            _ => anyhow::bail!("norm currently only supports F32 tensors"),
        }
    }
}

// === Reduction Operations ===
impl Tensor {
    /// Sum along all dimensions or specified dimension
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to sum along. If None, sums all elements
    /// * `keepdims` - If true, keeps the reduced dimension as size 1
    ///
    /// # Returns
    /// * If `dim` is None: returns scalar tensor with sum of all elements
    /// * If `dim` is Some: returns tensor with sum along specified dimension
    pub fn sum_dim(&self, dim: Option<usize>, keepdims: bool) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => match dim {
                Some(ax) => {
                    if ax >= arr.ndim() {
                        anyhow::bail!(
                            "Dimension {} out of bounds for tensor with {} dimensions",
                            ax,
                            arr.ndim()
                        );
                    }
                    let result = arr.sum_axis(ndarray::Axis(ax));
                    let mut tensor: Tensor = result.into_dyn().into();
                    if keepdims {
                        tensor = tensor.insert_axis(ax)?;
                    }
                    Ok(tensor)
                }
                None => {
                    let sum_val = arr.sum();
                    let mut tensor: Tensor =
                        ndarray::Array0::from_elem((), sum_val).into_dyn().into();
                    if keepdims {
                        for i in 0..arr.ndim() {
                            tensor = tensor.insert_axis(i)?;
                        }
                    }
                    Ok(tensor)
                }
            },
            _ => anyhow::bail!("sum_dim currently only supports F32 tensors"),
        }
    }

    /// Mean along specified dimension or all dimensions
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to compute mean along. If None, computes global mean
    /// * `keepdims` - If true, keeps the reduced dimension as size 1
    pub fn mean_dim(&self, dim: Option<usize>, keepdims: bool) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => match dim {
                Some(ax) => {
                    if ax >= arr.ndim() {
                        anyhow::bail!(
                            "Dimension {} out of bounds for tensor with {} dimensions",
                            ax,
                            arr.ndim()
                        );
                    }
                    let result = arr.mean_axis(ndarray::Axis(ax)).unwrap();
                    let mut tensor: Tensor = result.into_dyn().into();
                    if keepdims {
                        tensor = tensor.insert_axis(ax)?;
                    }
                    Ok(tensor)
                }
                None => {
                    let mean_val = arr.mean().unwrap_or(0.0);
                    let mut tensor: Tensor =
                        ndarray::Array0::from_elem((), mean_val).into_dyn().into();
                    if keepdims {
                        for i in 0..arr.ndim() {
                            tensor = tensor.insert_axis(i)?;
                        }
                    }
                    Ok(tensor)
                }
            },
            _ => anyhow::bail!("mean_dim currently only supports F32 tensors"),
        }
    }

    /// Maximum value along specified dimension or all dimensions
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to find max along. If None, finds global max
    /// * `keepdims` - If true, keeps the reduced dimension as size 1
    pub fn max_dim(&self, dim: Option<usize>, keepdims: bool) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => match dim {
                Some(ax) => {
                    if ax >= arr.ndim() {
                        anyhow::bail!(
                            "Dimension {} out of bounds for tensor with {} dimensions",
                            ax,
                            arr.ndim()
                        );
                    }
                    let result =
                        arr.fold_axis(ndarray::Axis(ax), f32::NEG_INFINITY, |&a, &b| a.max(b));
                    let mut tensor: Tensor = result.into_dyn().into();
                    if keepdims {
                        tensor = tensor.insert_axis(ax)?;
                    }
                    Ok(tensor)
                }
                None => {
                    let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut tensor: Tensor =
                        ndarray::Array0::from_elem((), max_val).into_dyn().into();
                    if keepdims {
                        for i in 0..arr.ndim() {
                            tensor = tensor.insert_axis(i)?;
                        }
                    }
                    Ok(tensor)
                }
            },
            _ => anyhow::bail!("max_dim currently only supports F32 tensors"),
        }
    }

    /// Maximum value
    pub fn max(&self) -> f32 {
        match &self.data {
            DTypeTensor::F32(arr) => arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            _ => panic!("max currently only supports F32 tensors"),
        }
    }

    /// Minimum value along specified dimension or all dimensions
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to find min along. If None, finds global min
    /// * `keepdims` - If true, keeps the reduced dimension as size 1
    pub fn min_dim(&self, dim: Option<usize>, keepdims: bool) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => match dim {
                Some(ax) => {
                    if ax >= arr.ndim() {
                        anyhow::bail!(
                            "Dimension {} out of bounds for tensor with {} dimensions",
                            ax,
                            arr.ndim()
                        );
                    }
                    let result = arr.fold_axis(ndarray::Axis(ax), f32::INFINITY, |&a, &b| a.min(b));
                    let mut tensor: Tensor = result.into_dyn().into();
                    if keepdims {
                        tensor = tensor.insert_axis(ax)?;
                    }
                    Ok(tensor)
                }
                None => {
                    let min_val = arr.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let mut tensor: Tensor =
                        ndarray::Array0::from_elem((), min_val).into_dyn().into();
                    if keepdims {
                        for i in 0..arr.ndim() {
                            tensor = tensor.insert_axis(i)?;
                        }
                    }
                    Ok(tensor)
                }
            },
            _ => anyhow::bail!("min_dim currently only supports F32 tensors"),
        }
    }

    /// Minimum value
    pub fn min(&self) -> f32 {
        match &self.data {
            DTypeTensor::F32(arr) => arr.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            _ => panic!("min currently only supports F32 tensors"),
        }
    }

    /// Standard deviation along specified dimension or all dimensions
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to compute std along. If None, computes global std
    /// * `keepdims` - If true, keeps the reduced dimension as size 1
    pub fn std_dim(&self, dim: Option<usize>, keepdims: bool) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => match dim {
                Some(ax) => {
                    if ax >= arr.ndim() {
                        anyhow::bail!(
                            "Dimension {} out of bounds for tensor with {} dimensions",
                            ax,
                            arr.ndim()
                        );
                    }
                    let mean_vals = arr.mean_axis(ndarray::Axis(ax)).unwrap();
                    let mut variance = Array::zeros(mean_vals.raw_dim());

                    for (i, lane) in arr.axis_iter(ndarray::Axis(ax)).enumerate() {
                        let mean_val = mean_vals[i];
                        let var_val = lane.mapv(|x| (x - mean_val).powi(2)).mean().unwrap_or(0.0);
                        variance[i] = var_val;
                    }

                    let result = variance.mapv(|x| x.sqrt());
                    let mut tensor: Tensor = result.into_dyn().into();
                    if keepdims {
                        tensor = tensor.insert_axis(ax)?;
                    }
                    Ok(tensor)
                }
                None => {
                    let mean_val = arr.mean().unwrap_or(0.0);
                    let variance = arr.mapv(|x| (x - mean_val).powi(2)).mean().unwrap_or(0.0);
                    let std_val = variance.sqrt();
                    let mut tensor: Tensor =
                        ndarray::Array0::from_elem((), std_val).into_dyn().into();
                    if keepdims {
                        for i in 0..arr.ndim() {
                            tensor = tensor.insert_axis(i)?;
                        }
                    }
                    Ok(tensor)
                }
            },
            _ => anyhow::bail!("std_dim currently only supports F32 tensors"),
        }
    }

    /// Standard deviation
    #[deprecated(
        since = "0.1.0",
        note = "Use `std_dim(None, false)` instead for better API consistency"
    )]
    pub fn std(&self) -> f32 {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let mean_val = arr.mean().unwrap_or(0.0);
                let variance = arr.mapv(|x| (x - mean_val).powi(2)).mean().unwrap_or(0.0);
                variance.sqrt()
            }
            _ => panic!("std currently only supports F32 tensors"),
        }
    }

    /// Variance along specified dimension or all dimensions
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to compute variance along. If None, computes global variance
    /// * `keepdims` - If true, keeps the reduced dimension as size 1
    pub fn var_dim(&self, dim: Option<usize>, keepdims: bool) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => match dim {
                Some(ax) => {
                    if ax >= arr.ndim() {
                        anyhow::bail!(
                            "Dimension {} out of bounds for tensor with {} dimensions",
                            ax,
                            arr.ndim()
                        );
                    }
                    let mean_vals = arr.mean_axis(ndarray::Axis(ax)).unwrap();
                    let mut variance = Array::zeros(mean_vals.raw_dim());

                    for (i, lane) in arr.axis_iter(ndarray::Axis(ax)).enumerate() {
                        let mean_val = mean_vals[i];
                        let var_val = lane.mapv(|x| (x - mean_val).powi(2)).mean().unwrap_or(0.0);
                        variance[i] = var_val;
                    }

                    let mut tensor: Tensor = variance.into_dyn().into();
                    if keepdims {
                        tensor = tensor.insert_axis(ax)?;
                    }
                    Ok(tensor)
                }
                None => {
                    let mean_val = arr.mean().unwrap_or(0.0);
                    let variance = arr.mapv(|x| (x - mean_val).powi(2)).mean().unwrap_or(0.0);
                    let mut tensor: Tensor =
                        ndarray::Array0::from_elem((), variance).into_dyn().into();
                    if keepdims {
                        for i in 0..arr.ndim() {
                            tensor = tensor.insert_axis(i)?;
                        }
                    }
                    Ok(tensor)
                }
            },
            _ => anyhow::bail!("var_dim currently only supports F32 tensors"),
        }
    }

    /// Variance
    #[deprecated(
        since = "0.1.0",
        note = "Use `var_dim(None, false)` instead for better API consistency"
    )]
    pub fn var(&self) -> f32 {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let mean_val = arr.mean().unwrap_or(0.0);
                arr.mapv(|x| (x - mean_val).powi(2)).mean().unwrap_or(0.0)
            }
            _ => panic!("var currently only supports F32 tensors"),
        }
    }

    /// Find indices of maximum values along specified dimension
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to find argmax along. If None, finds global argmax
    /// * `keepdims` - If true, keeps the reduced dimension as size 1
    pub fn argmax(&self, dim: Option<usize>, keepdims: bool) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => match dim {
                Some(ax) => {
                    if ax >= arr.ndim() {
                        anyhow::bail!(
                            "Dimension {} out of bounds for tensor with {} dimensions",
                            ax,
                            arr.ndim()
                        );
                    }
                    let mut result_shape = arr.shape().to_vec();
                    result_shape.remove(ax);
                    if result_shape.is_empty() {
                        result_shape.push(1);
                    }

                    let mut result = Array::zeros(result_shape);
                    for (i, lane) in arr.axis_iter(ndarray::Axis(ax)).enumerate() {
                        let (max_idx, _) = lane
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap_or((0, &0.0));
                        result[i] = max_idx as f32;
                    }

                    let mut tensor: Tensor = result.into_dyn().into();
                    if keepdims {
                        tensor = tensor.insert_axis(ax)?;
                    }
                    Ok(tensor)
                }
                None => {
                    let (max_idx, _) = arr
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or((0, &0.0));
                    let mut tensor: Tensor = ndarray::Array0::from_elem((), max_idx as f32)
                        .into_dyn()
                        .into();
                    if keepdims {
                        for i in 0..arr.ndim() {
                            tensor = tensor.insert_axis(i)?;
                        }
                    }
                    Ok(tensor)
                }
            },
            _ => anyhow::bail!("argmax currently only supports F32 tensors"),
        }
    }

    /// Find indices of minimum values along specified dimension
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to find argmin along. If None, finds global argmin
    /// * `keepdims` - If true, keeps the reduced dimension as size 1
    pub fn argmin(&self, dim: Option<usize>, keepdims: bool) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => match dim {
                Some(ax) => {
                    if ax >= arr.ndim() {
                        anyhow::bail!(
                            "Dimension {} out of bounds for tensor with {} dimensions",
                            ax,
                            arr.ndim()
                        );
                    }
                    let mut result_shape = arr.shape().to_vec();
                    result_shape.remove(ax);
                    if result_shape.is_empty() {
                        result_shape.push(1);
                    }

                    let mut result = Array::zeros(result_shape);
                    for (i, lane) in arr.axis_iter(ndarray::Axis(ax)).enumerate() {
                        let (min_idx, _) = lane
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap_or((0, &0.0));
                        result[i] = min_idx as f32;
                    }

                    let mut tensor: Tensor = result.into_dyn().into();
                    if keepdims {
                        tensor = tensor.insert_axis(ax)?;
                    }
                    Ok(tensor)
                }
                None => {
                    let (min_idx, _) = arr
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or((0, &0.0));
                    let mut tensor: Tensor = ndarray::Array0::from_elem((), min_idx as f32)
                        .into_dyn()
                        .into();
                    if keepdims {
                        for i in 0..arr.ndim() {
                            tensor = tensor.insert_axis(i)?;
                        }
                    }
                    Ok(tensor)
                }
            },
            _ => anyhow::bail!("argmin currently only supports F32 tensors"),
        }
    }
}

// === Conversion and Utility Methods ===
impl Tensor {
    /// Convert to F16 for memory efficiency (lite builder)
    pub fn to_f16(&self) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let f16_arr = arr.mapv(f16::from_f32).into_shared();
                Ok(Self {
                    data: DTypeTensor::F16(f16_arr),
                    dtype: DType::Fp16,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::F16(_) => Ok(self.clone()),
            _ => anyhow::bail!("to_f16 conversion not supported for this dtype"),
        }
    }

    /// Convert to F32 (lite builder)
    pub fn to_f32(&self) -> Result<Self> {
        match &self.data {
            DTypeTensor::F16(arr) => {
                let f32_arr = arr.mapv(|x| x.to_f32()).into_shared();
                Ok(Self {
                    data: DTypeTensor::F32(f32_arr),
                    dtype: DType::Fp32,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::F32(_) => Ok(self.clone()),
            _ => anyhow::bail!("to_f32 conversion not supported for this dtype"),
        }
    }

    /// Convert tensor data to Vec<f32> regardless of original dtype
    ///
    /// This method handles type conversion automatically and returns a Vec<f32>
    /// that can be used directly with APIs expecting f32 data.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        match &self.data {
            DTypeTensor::F32(arr) => Ok(arr.iter().copied().collect()),
            DTypeTensor::F16(arr) => Ok(arr.iter().map(|x| x.to_f32()).collect()),
            DTypeTensor::Bf16(arr) => Ok(arr.iter().map(|x| x.to_f32()).collect()),
            DTypeTensor::F64(arr) => Ok(arr.iter().map(|x| *x as f32).collect()),
            DTypeTensor::I8(arr) => Ok(arr.iter().map(|x| *x as f32).collect()),
            DTypeTensor::I16(arr) => Ok(arr.iter().map(|x| *x as f32).collect()),
            DTypeTensor::I32(arr) => Ok(arr.iter().map(|x| *x as f32).collect()),
            DTypeTensor::I64(arr) => Ok(arr.iter().map(|x| *x as f32).collect()),
            DTypeTensor::U8(arr) => Ok(arr.iter().map(|x| *x as f32).collect()),
            DTypeTensor::U16(arr) => Ok(arr.iter().map(|x| *x as f32).collect()),
            DTypeTensor::U32(arr) => Ok(arr.iter().map(|x| *x as f32).collect()),
            DTypeTensor::U64(arr) => Ok(arr.iter().map(|x| *x as f32).collect()),
            DTypeTensor::Bool(arr) => Ok(arr.iter().map(|x| if *x { 1.0 } else { 0.0 }).collect()),
        }
    }

    /// Set data type (lite builder)
    #[inline]
    pub fn with_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            data: self.data.clone(),
            dtype,
            uid: Self::generate_uid(),
        })
    }

    /// Clone with new UID
    #[inline]
    pub fn clone_with_new_uid(&self) -> Self {
        let mut cloned = self.clone();
        cloned.uid = Self::generate_uid();
        cloned
    }
}

// === Property Access Methods ===
impl Tensor {
    /// Get reference to underlying data
    #[inline]
    pub fn data(&self) -> &DTypeTensor {
        &self.data
    }

    /// Get mutable reference to underlying data
    #[inline]
    pub fn data_mut(&mut self) -> &mut DTypeTensor {
        &mut self.data
    }

    /// Get tensor shape
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.dims()
    }

    /// Get tensor dimensions
    #[inline]
    pub fn dims(&self) -> &[usize] {
        match &self.data {
            DTypeTensor::F32(arr) => arr.shape(),
            DTypeTensor::F64(arr) => arr.shape(),
            DTypeTensor::F16(arr) => arr.shape(),
            DTypeTensor::Bf16(arr) => arr.shape(),
            DTypeTensor::I8(arr) => arr.shape(),
            DTypeTensor::I16(arr) => arr.shape(),
            DTypeTensor::I32(arr) => arr.shape(),
            DTypeTensor::I64(arr) => arr.shape(),
            DTypeTensor::U8(arr) => arr.shape(),
            DTypeTensor::U16(arr) => arr.shape(),
            DTypeTensor::U32(arr) => arr.shape(),
            DTypeTensor::U64(arr) => arr.shape(),
            DTypeTensor::Bool(arr) => arr.shape(),
        }
    }

    /// Get number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Get total number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Check if tensor is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims().iter().any(|&d| d == 0)
    }

    /// Get data type
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get unique identifier
    #[inline]
    pub fn uid(&self) -> usize {
        self.uid
    }

    /// Convert tensor data to Vec<T>
    pub fn to_vec<T: TensorElement>(&self) -> Result<Vec<T>> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                // Create a contiguous copy first to avoid alignment issues
                let owned = arr.to_owned();
                let vec: Vec<T> = owned.into_iter().map(|x| T::from_f32(x)).collect();
                Ok(vec)
            }
            DTypeTensor::F64(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            DTypeTensor::F16(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x.to_f32())).collect();
                Ok(vec)
            }
            DTypeTensor::Bf16(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x.to_f32())).collect();
                Ok(vec)
            }
            DTypeTensor::I8(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            DTypeTensor::I16(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            DTypeTensor::I32(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            DTypeTensor::I64(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            DTypeTensor::U8(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            DTypeTensor::U16(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            DTypeTensor::U32(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            DTypeTensor::U64(arr) => {
                let vec: Vec<T> = arr.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            DTypeTensor::Bool(arr) => {
                let vec: Vec<T> = arr
                    .iter()
                    .map(|&x| T::from_f32(if x { 1.0 } else { 0.0 }))
                    .collect();
                Ok(vec)
            }
        }
    }

    /// Get slice view of tensor data for contiguous tensors
    /// Returns None if tensor is not contiguous in memory
    pub fn as_slice<T: TensorElement>(&self) -> Option<&[T]> {
        use std::any::TypeId;
        match &self.data {
            DTypeTensor::F32(arr) if TypeId::of::<T>() == TypeId::of::<f32>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::F64(arr) if TypeId::of::<T>() == TypeId::of::<f64>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::F16(arr) if TypeId::of::<T>() == TypeId::of::<f16>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::Bf16(arr) if TypeId::of::<T>() == TypeId::of::<bf16>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::I8(arr) if TypeId::of::<T>() == TypeId::of::<i8>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::I16(arr) if TypeId::of::<T>() == TypeId::of::<i16>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::I32(arr) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::I64(arr) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::U8(arr) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::U16(arr) if TypeId::of::<T>() == TypeId::of::<u16>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::U32(arr) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::U64(arr) if TypeId::of::<T>() == TypeId::of::<u64>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            DTypeTensor::Bool(arr) if TypeId::of::<T>() == TypeId::of::<bool>() => {
                arr.as_slice().map(|slice| unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len())
                })
            }
            _ => None,
        }
    }

    /// Get mutable slice view of tensor data for contiguous tensors
    /// Returns None if tensor is not contiguous in memory
    pub fn as_slice_mut<T: TensorElement>(&mut self) -> Option<&mut [T]> {
        use std::any::TypeId;
        match &mut self.data {
            DTypeTensor::F32(arr) if TypeId::of::<T>() == TypeId::of::<f32>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::F64(arr) if TypeId::of::<T>() == TypeId::of::<f64>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::F16(arr) if TypeId::of::<T>() == TypeId::of::<f16>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::Bf16(arr) if TypeId::of::<T>() == TypeId::of::<bf16>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::I8(arr) if TypeId::of::<T>() == TypeId::of::<i8>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::I16(arr) if TypeId::of::<T>() == TypeId::of::<i16>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::I32(arr) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::I64(arr) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::U8(arr) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::U16(arr) if TypeId::of::<T>() == TypeId::of::<u16>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::U32(arr) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::U64(arr) if TypeId::of::<T>() == TypeId::of::<u64>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            DTypeTensor::Bool(arr) if TypeId::of::<T>() == TypeId::of::<bool>() => {
                arr.as_slice_mut().map(|slice| unsafe {
                    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, slice.len())
                })
            }
            _ => None,
        }
    }

    /// Get underlying ArcArray<f32, IxDyn>
    #[inline]
    pub fn as_array(&self) -> &ArcArray<f32, IxDyn> {
        match &self.data {
            DTypeTensor::F32(arr) => arr,
            _ => panic!("Tensor is not F32 type"),
        }
    }

    /// Get mutable reference to underlying ArcArray<f32, IxDyn>
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut ArcArray<f32, IxDyn> {
        match &mut self.data {
            DTypeTensor::F32(arr) => arr,
            _ => panic!("Tensor is not F32 type"),
        }
    }

    /// Get reference to underlying array with specific type
    pub fn as_array_typed<T: TensorElement>(&self) -> Result<&ArcArray<T, IxDyn>> {
        use std::any::TypeId;
        match &self.data {
            DTypeTensor::F32(arr) if TypeId::of::<T>() == TypeId::of::<f32>() => {
                // Safety: We know T is f32 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<f32>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::F64(arr) if TypeId::of::<T>() == TypeId::of::<f64>() => {
                // Safety: We know T is f64 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<f64>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::I64(arr) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                // Safety: We know T is i64 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<i64>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::I32(arr) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                // Safety: We know T is i32 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<i32>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::I16(arr) if TypeId::of::<T>() == TypeId::of::<i16>() => {
                // Safety: We know T is i16 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<i16>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::I8(arr) if TypeId::of::<T>() == TypeId::of::<i8>() => {
                // Safety: We know T is i8 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<i8>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::U64(arr) if TypeId::of::<T>() == TypeId::of::<u64>() => {
                // Safety: We know T is u64 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<u64>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::U32(arr) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                // Safety: We know T is u32 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<u32>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::U16(arr) if TypeId::of::<T>() == TypeId::of::<u16>() => {
                // Safety: We know T is u16 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<u16>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::U8(arr) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                // Safety: We know T is u8 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<u8>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::Bool(arr) if TypeId::of::<T>() == TypeId::of::<bool>() => {
                // Safety: We know T is bool due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<bool>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::F16(arr) if TypeId::of::<T>() == TypeId::of::<f16>() => {
                // Safety: We know T is f16 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<half::f16>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            DTypeTensor::Bf16(arr) if TypeId::of::<T>() == TypeId::of::<bf16>() => {
                // Safety: We know T is bf16 due to TypeId check
                Ok(unsafe {
                    std::mem::transmute::<
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<half::bf16>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                        &ndarray::ArrayBase<
                            ndarray::OwnedArcRepr<T>,
                            ndarray::Dim<ndarray::IxDynImpl>,
                        >,
                    >(arr)
                })
            }
            _ => Err(anyhow!(
                "Type mismatch: tensor is {:?}, requested type does not match",
                self.dtype
            )),
        }
    }

    /// Get element at index with automatic type conversion
    pub fn get_element<T: TensorElement>(&self, index: usize) -> Result<T> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index]))
            }
            DTypeTensor::F64(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index] as f32))
            }
            DTypeTensor::I64(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index] as f32))
            }
            DTypeTensor::I32(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index] as f32))
            }
            DTypeTensor::I16(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index] as f32))
            }
            DTypeTensor::I8(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index] as f32))
            }
            DTypeTensor::U64(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index] as f32))
            }
            DTypeTensor::U32(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index] as f32))
            }
            DTypeTensor::U16(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index] as f32))
            }
            DTypeTensor::U8(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index] as f32))
            }
            DTypeTensor::Bool(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(if slice[index] { 1.0 } else { 0.0 }))
            }
            DTypeTensor::F16(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index].to_f32()))
            }
            DTypeTensor::Bf16(arr) => {
                let slice = arr
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                if index >= slice.len() {
                    return Err(anyhow!(
                        "Index {} out of bounds for tensor of length {}",
                        index,
                        slice.len()
                    ));
                }
                Ok(T::from_f32(slice[index].to_f32()))
            }
        }
    }

    /// Get memory size in bytes
    #[inline]
    pub fn memory_size(&self) -> usize {
        let element_count = self.len();
        match self.dtype {
            DType::Fp32 => element_count * std::mem::size_of::<f32>(),
            DType::Fp64 => element_count * std::mem::size_of::<f64>(),
            DType::Fp16 => element_count * std::mem::size_of::<half::f16>(),
            DType::Bf16 => element_count * std::mem::size_of::<half::bf16>(),
            DType::Int8 => element_count * std::mem::size_of::<i8>(),
            DType::Int16 => element_count * std::mem::size_of::<i16>(),
            DType::Int32 => element_count * std::mem::size_of::<i32>(),
            DType::Int64 => element_count * std::mem::size_of::<i64>(),
            DType::Uint8 => element_count * std::mem::size_of::<u8>(),
            DType::Uint16 => element_count * std::mem::size_of::<u16>(),
            DType::Uint32 => element_count * std::mem::size_of::<u32>(),
            DType::Uint64 => element_count * std::mem::size_of::<u64>(),
            DType::Auto => 0,                        // Auto type has no fixed size
            DType::Bool => element_count,            // 1 bit per element
            DType::Int4 => (element_count + 1) / 2,  // 4-bit integers, 2 per byte
            DType::Uint4 => (element_count + 1) / 2, // 4-bit integers, 2 per byte
            DType::Bnb4 => (element_count + 1) / 2,  // 4-bit BNB, 2 per byte
            DType::Q4 => (element_count + 1) / 2,    // 4-bit quantized, 2 per byte
            DType::Q4f16 => element_count * 2,       // Q4 with f16 scale
            DType::Q8 => element_count,              // 8-bit quantized
            DType::Fp8e4m3fn => element_count,       // 8-bit float
            DType::Fp8e4m3fnuz => element_count,     // 8-bit float
            DType::Fp8e5m2 => element_count,         // 8-bit float
            DType::Fp8e5m2fnuz => element_count,     // 8-bit float
            DType::Fp4e2m1 => (element_count + 1) / 2, // 4-bit float, 2 per byte
            DType::Complex64 => element_count * std::mem::size_of::<f32>() * 2, // Complex f32
            DType::Complex128 => element_count * std::mem::size_of::<f64>() * 2, // Complex f64
        }
    }

    /// Iterate over a specific dimension, returning zero-copy views
    ///
    /// This method provides efficient iteration over slices along a specified dimension,
    /// returning `TensorView` instances for zero-copy access. This is particularly useful
    /// for processing batches, sequences, or other dimension-wise operations.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension index to iterate over (0-based)
    ///
    /// # Returns
    ///
    /// Returns a `TensorDimIter` that yields `TensorView` instances
    ///
    /// # Panics
    ///
    /// Panics if `dim` is greater than or equal to the number of dimensions
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![3, 4, 5]);
    ///
    /// // Iterate over the first dimension (3 slices of shape [4, 5])
    /// for view in tensor.iter_dim(0) {
    ///     assert_eq!(view.shape(), &[4, 5]);
    /// }
    ///
    /// // Iterate over the second dimension (4 slices of shape [3, 5])
    /// for view in tensor.iter_dim(1) {
    ///     assert_eq!(view.shape(), &[3, 5]);
    /// }
    /// ```
    pub fn iter_dim(&self, dim: usize) -> TensorDimIter<'_> {
        assert!(
            dim < self.ndim(),
            "Dimension {} is out of bounds for tensor with {} dimensions",
            dim,
            self.ndim()
        );
        TensorDimIter::new(self, dim)
    }

    /// Create a mutable iterator over slices along a specific dimension
    ///
    /// This method returns a mutable iterator that yields tensor slices along the specified dimension.
    /// Each slice is a tensor with one less dimension than the original tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension index to iterate along (0-indexed)
    ///
    /// # Returns
    ///
    /// A `TensorDimIterMut` that yields mutable tensor slices
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of bounds for the tensor dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// let mut tensor = Tensor::zeros(vec![3, 4, 5]);
    /// let mut iter = tensor.iter_mut_dim(0);
    ///
    /// // Iterate over the first dimension (3 slices of shape [4, 5])
    /// while let Some(mut slice_tensor) = iter.next_slice() {
    ///     assert_eq!(slice_tensor.shape(), &[4, 5]);
    ///     // Can modify slice_tensor here
    /// }
    /// ```
    pub fn iter_mut_dim(&mut self, dim: usize) -> TensorDimIterMut<'_> {
        assert!(
            dim < self.ndim(),
            "Dimension {} is out of bounds for tensor with {} dimensions",
            dim,
            self.ndim()
        );
        TensorDimIterMut::new(self, dim)
    }
}

// === Slice and View Operations ===
impl Tensor {
    /// Create an immutable view of the entire tensor
    ///
    /// This method provides a zero-copy view of the entire tensor, which is useful
    /// for operations that need to work with tensor views consistently.
    ///
    /// # Returns
    ///
    /// Returns a `TensorView` containing the entire tensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![4, 4]);
    /// let view = tensor.view();
    /// assert_eq!(view.shape(), tensor.shape());
    /// ```
    pub fn view(&self) -> TensorView<'_> {
        let ranges: Vec<std::ops::Range<usize>> =
            self.shape().iter().map(|&size| 0..size).collect();
        TensorView::new(&self.data, self.dtype, &ranges)
            .expect("Creating view of entire tensor should never fail")
    }

    /// Create a mutable view of the entire tensor
    ///
    /// This method provides a zero-copy mutable view of the entire tensor, enabling
    /// in-place modifications while maintaining the view interface.
    ///
    /// # Returns
    ///
    /// Returns a `TensorViewMut` containing the entire tensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// let mut tensor = Tensor::zeros(vec![4, 4]);
    /// let mut view = tensor.view_mut();
    /// view.fill(1.0).unwrap();
    /// ```
    pub fn view_mut(&mut self) -> TensorViewMut<'_> {
        let ranges: Vec<std::ops::Range<usize>> =
            self.shape().iter().map(|&size| 0..size).collect();
        TensorViewMut::new(&mut self.data, self.dtype, &ranges)
            .expect("Creating mutable view of entire tensor should never fail")
    }

    /// Create a zero-copy immutable view of the tensor with specified slice specification
    ///
    /// This method provides efficient slicing without copying data, which is crucial
    /// for performance-critical operations like KV cache management.
    ///
    /// # Arguments
    ///
    /// * `spec` - Slice specification that can be converted to slice ranges
    ///
    /// # Returns
    ///
    /// Returns a `Result<TensorView>` containing the view or an error if slicing fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The slice specification is invalid
    /// - Any range is out of bounds
    /// - The tensor data type is not supported for views
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![4, 4]);
    /// let view = tensor.slice(&[0..2, 0..2]).unwrap();
    /// assert_eq!(view.shape(), &[2, 2]);
    /// ```
    pub fn slice<S: IntoSliceSpec>(&self, spec: S) -> Result<TensorView<'_>> {
        let mut slice_spec = spec.into_slice_spec();
        let shape = self.shape();

        // Auto-fill missing dimensions with FullSlice
        while slice_spec.len() < shape.len() {
            slice_spec.push(SliceOrIndex::FullSlice);
        }

        TensorView::new_with_specs(&self.data, self.dtype, &slice_spec)
    }

    /// Create a zero-copy mutable view of the tensor with specified slice specification
    ///
    /// This method provides efficient mutable slicing without copying data, enabling
    /// in-place modifications for operations like KV cache updates.
    ///
    /// # Arguments
    ///
    /// * `spec` - Slice specification that can be converted to slice ranges
    ///
    /// # Returns
    ///
    /// Returns a `Result<TensorViewMut>` containing the mutable view or an error if slicing fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The slice specification is invalid
    /// - Any range is out of bounds
    /// - The tensor data type is not supported for mutable views
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// let mut tensor = Tensor::zeros(vec![4, 4]);
    /// let mut view = tensor.slice_mut(&[0..2, 0..2]).unwrap();
    /// view.fill(1.0).unwrap();
    /// ```
    pub fn slice_mut<S: IntoSliceSpec>(&mut self, spec: S) -> Result<TensorViewMut<'_>> {
        let mut slice_spec = spec.into_slice_spec();
        let shape = self.shape();

        // Auto-fill missing dimensions with FullSlice
        while slice_spec.len() < shape.len() {
            slice_spec.push(SliceOrIndex::FullSlice);
        }

        TensorViewMut::new_with_specs(&mut self.data, self.dtype, &slice_spec)
    }
}

// === Split Operations ===
impl Tensor {
    /// Split the tensor along the specified axis at the given index
    ///
    /// This method splits the tensor into two parts along the specified axis.
    /// The first tensor contains elements before the split index,
    /// and the second tensor contains elements from the split index onwards.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to split
    /// * `index` - The index at which to split (must be within bounds)
    ///
    /// # Returns
    ///
    /// Returns a `Result<(Self, Self)>` containing the two split tensors or an error
    ///
    /// # Panics
    ///
    /// Panics if axis or index is out of bounds.
    ///
    pub fn split_at(&self, axis: usize, index: usize) -> Result<(Self, Self)> {
        // Check if axis is valid
        if axis >= self.ndim() {
            anyhow::bail!(
                "Axis {} is out of bounds for tensor with {} dimensions",
                axis,
                self.ndim()
            );
        }

        // Check if index is valid
        let dim_size = self.shape()[axis];
        if index > dim_size {
            anyhow::bail!(
                "Index {} is out of bounds for axis {} with size {}",
                index,
                axis,
                dim_size
            );
        }

        // Use ndarray's efficient split_at method when possible
        match &self.data {
            DTypeTensor::F32(arr) => {
                // Convert to view first, then use ndarray's split_at
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::F32(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::F32(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::F64(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::F64(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::F64(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::F16(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::F16(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::F16(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::Bf16(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::Bf16(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::Bf16(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::I8(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::I8(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::I8(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::I16(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::I16(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::I16(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::I32(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::I32(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::I32(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::I64(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::I64(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::I64(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::U8(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::U8(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::U8(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::U16(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::U16(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::U16(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::U32(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::U32(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::U32(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::U64(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::U64(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::U64(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
            DTypeTensor::Bool(arr) => {
                let view = arr.view();
                let (left_view, right_view) = view.split_at(Axis(axis), index);
                Ok((
                    Self {
                        data: DTypeTensor::Bool(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                    Self {
                        data: DTypeTensor::Bool(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Self::generate_uid(),
                    },
                ))
            }
        }
    }
}

// === Image Processing Methods ===
impl Tensor {
    /// Convert NHWC to NCHW format (lite builder)
    pub fn nhwc_to_nchw(&self) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                // NHWC to NCHW: [0, 1, 2, 3] -> [0, 3, 1, 2]
                let converted = arr.to_owned().permuted_axes(vec![0, 3, 1, 2]);
                Ok(Self {
                    data: DTypeTensor::F32(converted.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("nhwc_to_nchw currently only supports F32 tensors"),
        }
    }

    /// Convert NCHW to NHWC format (lite builder)
    pub fn nchw_to_nhwc(&self) -> Result<Self> {
        match &self.data {
            DTypeTensor::F32(arr) => {
                // NCHW to NHWC: [0, 1, 2, 3] -> [0, 2, 3, 1]
                let converted = arr.to_owned().permuted_axes(vec![0, 2, 3, 1]);
                Ok(Self {
                    data: DTypeTensor::F32(converted.into_shared()),
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("nchw_to_nhwc currently only supports F32 tensors"),
        }
    }
}
