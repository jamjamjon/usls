use ort::tensor::TensorElementType;

impl From<TensorElementType> for crate::DType {
    fn from(dtype: TensorElementType) -> Self {
        match dtype {
            TensorElementType::Int4 => Self::Int4,
            TensorElementType::Int8 => Self::Int8,
            TensorElementType::Int16 => Self::Int16,
            TensorElementType::Int32 => Self::Int32,
            TensorElementType::Int64 => Self::Int64,
            TensorElementType::Uint4 => Self::Uint4,
            TensorElementType::Uint8 => Self::Uint8,
            TensorElementType::Uint16 => Self::Uint16,
            TensorElementType::Uint32 => Self::Uint32,
            TensorElementType::Uint64 => Self::Uint64,
            TensorElementType::Float16 => Self::Fp16,
            TensorElementType::Float32 => Self::Fp32,
            TensorElementType::Float64 => Self::Fp64,
            TensorElementType::Bfloat16 => Self::Bf16,
            TensorElementType::Float8E4M3FN => Self::Fp8e4m3fn,
            TensorElementType::Float8E4M3FNUZ => Self::Fp8e4m3fnuz,
            TensorElementType::Float8E5M2 => Self::Fp8e5m2,
            TensorElementType::Float8E5M2FNUZ => Self::Fp8e5m2fnuz,
            TensorElementType::Complex64 => Self::Complex64,
            TensorElementType::Complex128 => Self::Complex128,
            TensorElementType::Bool => Self::Uint8,
            TensorElementType::String | TensorElementType::Undefined => Self::Auto,
        }
    }
}
