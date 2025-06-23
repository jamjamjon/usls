//! Tensor module containing all tensor-related functionality
use half::{bf16, f16};
use ndarray::{ArcArray, IxDyn};

use crate::{DType, DTypeTensor};

/// Trait for types that can be used as tensor elements
pub trait TensorElement: Clone + Send + Sync + 'static {
    /// Get the corresponding DType for this element type
    fn dtype() -> DType;

    /// Convert to DTypeTensor variant
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor;

    /// Convert from f32 (for type casting)
    fn from_f32(val: f32) -> Self;

    /// Convert to f32 (for type casting)
    fn to_f32(self) -> f32;

    /// Get the minimum value for this type
    fn min_value() -> Option<Self>;

    /// Get the maximum value for this type
    fn max_value() -> Option<Self>;
}

// Implement TensorElement for all supported types
impl TensorElement for f32 {
    fn dtype() -> DType {
        DType::Fp32
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::F32(data)
    }
    fn from_f32(val: f32) -> Self {
        val
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn min_value() -> Option<Self> {
        Some(f32::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(f32::MAX)
    }
}

impl TensorElement for f64 {
    fn dtype() -> DType {
        DType::Fp64
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::F64(data)
    }
    fn from_f32(val: f32) -> Self {
        val as f64
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn min_value() -> Option<Self> {
        Some(f64::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(f64::MAX)
    }
}

impl TensorElement for f16 {
    fn dtype() -> DType {
        DType::Fp16
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::F16(data)
    }
    fn from_f32(val: f32) -> Self {
        f16::from_f32(val)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn min_value() -> Option<Self> {
        Some(f16::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(f16::MAX)
    }
}

impl TensorElement for bf16 {
    fn dtype() -> DType {
        DType::Bf16
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::Bf16(data)
    }
    fn from_f32(val: f32) -> Self {
        bf16::from_f32(val)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn min_value() -> Option<Self> {
        Some(bf16::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(bf16::MAX)
    }
}

impl TensorElement for i8 {
    fn dtype() -> DType {
        DType::Int8
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::I8(data)
    }
    fn from_f32(val: f32) -> Self {
        val as i8
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn min_value() -> Option<Self> {
        Some(i8::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(i8::MAX)
    }
}

impl TensorElement for i16 {
    fn dtype() -> DType {
        DType::Int16
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::I16(data)
    }
    fn from_f32(val: f32) -> Self {
        val as i16
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn min_value() -> Option<Self> {
        Some(i16::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(i16::MAX)
    }
}

impl TensorElement for i32 {
    fn dtype() -> DType {
        DType::Int32
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::I32(data)
    }
    fn from_f32(val: f32) -> Self {
        val as i32
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn min_value() -> Option<Self> {
        Some(i32::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(i32::MAX)
    }
}

impl TensorElement for i64 {
    fn dtype() -> DType {
        DType::Int64
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::I64(data)
    }
    fn from_f32(val: f32) -> Self {
        val as i64
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn min_value() -> Option<Self> {
        Some(i64::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(i64::MAX)
    }
}

impl TensorElement for u8 {
    fn dtype() -> DType {
        DType::Uint8
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::U8(data)
    }
    fn from_f32(val: f32) -> Self {
        val as u8
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn min_value() -> Option<Self> {
        Some(u8::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(u8::MAX)
    }
}

impl TensorElement for u16 {
    fn dtype() -> DType {
        DType::Uint16
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::U16(data)
    }
    fn from_f32(val: f32) -> Self {
        val as u16
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn min_value() -> Option<Self> {
        Some(u16::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(u16::MAX)
    }
}

impl TensorElement for u32 {
    fn dtype() -> DType {
        DType::Uint32
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::U32(data)
    }
    fn from_f32(val: f32) -> Self {
        val as u32
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn min_value() -> Option<Self> {
        Some(u32::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(u32::MAX)
    }
}

impl TensorElement for u64 {
    fn dtype() -> DType {
        DType::Uint64
    }
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::U64(data)
    }
    fn from_f32(val: f32) -> Self {
        val as u64
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn min_value() -> Option<Self> {
        Some(u64::MIN)
    }
    fn max_value() -> Option<Self> {
        Some(u64::MAX)
    }
}

impl TensorElement for bool {
    fn dtype() -> DType {
        DType::Uint8
    } // Bool maps to Uint8 in DType
    fn into_dtype_tensor(data: ArcArray<Self, IxDyn>) -> DTypeTensor {
        DTypeTensor::Bool(data)
    }
    fn from_f32(val: f32) -> Self {
        val != 0.0
    }
    fn to_f32(self) -> f32 {
        if self {
            1.0
        } else {
            0.0
        }
    }
    fn min_value() -> Option<Self> {
        None
    }
    fn max_value() -> Option<Self> {
        None
    }
}
