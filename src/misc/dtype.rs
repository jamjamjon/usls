use ort::tensor::TensorElementType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    Auto,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Fp16,
    Fp32,
    Fp64,
    Bf16,
    Bool,
    String,
    Bnb4,
    Q4,
    Q4f16,
}

impl TryFrom<&str> for DType {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "auto" | "dyn" => Ok(Self::Auto),
            "u8" | "uint8" => Ok(Self::Uint8),
            "u16" | "uint16" => Ok(Self::Uint16),
            "u32" | "uint32" => Ok(Self::Uint32),
            "u64" | "uint64" => Ok(Self::Uint64),
            "i8" | "int8" => Ok(Self::Int8),
            "i16" | "int=16" => Ok(Self::Int16),
            "i32" | "int32" => Ok(Self::Int32),
            "i64" | "int64" => Ok(Self::Int64),
            "f16" | "fp16" => Ok(Self::Fp16),
            "f32" | "fp32" => Ok(Self::Fp32),
            "f64" | "fp64" => Ok(Self::Fp64),
            "b16" | "bf16" => Ok(Self::Bf16),
            "q4f16" => Ok(Self::Q4f16),
            "q4" => Ok(Self::Q4),
            "bnb4" => Ok(Self::Bnb4),
            x => anyhow::bail!("Unsupported Model DType: {}", x),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let x = match self {
            Self::Auto => "auto",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Uint8 => "uint8",
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
            Self::Uint64 => "uint64",
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
            Self::Fp64 => "fp64",
            Self::Bf16 => "bf16",
            Self::String => "string",
            Self::Bool => "bool",
            Self::Bnb4 => "bnb4",
            Self::Q4 => "q4",
            Self::Q4f16 => "q4f16",
        };
        write!(f, "{}", x)
    }
}

impl DType {
    pub fn to_ort(&self) -> TensorElementType {
        match self {
            Self::Int8 => TensorElementType::Int8,
            Self::Int16 => TensorElementType::Int16,
            Self::Int32 => TensorElementType::Int32,
            Self::Int64 => TensorElementType::Int64,
            Self::Uint8 => TensorElementType::Uint8,
            Self::Uint16 => TensorElementType::Uint16,
            Self::Uint32 => TensorElementType::Uint32,
            Self::Uint64 => TensorElementType::Uint64,
            Self::Fp16 => TensorElementType::Float16,
            Self::Fp32 => TensorElementType::Float32,
            Self::Fp64 => TensorElementType::Float64,
            Self::Bf16 => TensorElementType::Bfloat16,
            _ => todo!(),
        }
    }

    pub fn from_ort(dtype: &TensorElementType) -> Self {
        match dtype {
            TensorElementType::Int8 => Self::Int8,
            TensorElementType::Int16 => Self::Int16,
            TensorElementType::Int32 => Self::Int32,
            TensorElementType::Int64 => Self::Int64,
            TensorElementType::Uint8 => Self::Uint8,
            TensorElementType::Uint16 => Self::Uint16,
            TensorElementType::Uint32 => Self::Uint32,
            TensorElementType::Uint64 => Self::Uint64,
            TensorElementType::Float16 => Self::Fp16,
            TensorElementType::Float32 => Self::Fp32,
            TensorElementType::Float64 => Self::Fp64,
            TensorElementType::Bfloat16 => Self::Bf16,
            TensorElementType::String => Self::String,
            TensorElementType::Bool => Self::Bool,
        }
    }
}
