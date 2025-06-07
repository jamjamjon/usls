#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Data type enumeration for tensor elements.
pub enum DType {
    #[default]
    Auto,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint4,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Fp16,
    Fp32,
    Fp64,
    Bf16,
    Bnb4,
    Q4,
    Q4f16,
    Q8,
    Fp8e4m3fn,
    Fp8e4m3fnuz,
    Fp8e5m2,
    Fp8e5m2fnuz,
    Fp4e2m1,
    Complex64,
    Complex128,
}

impl std::str::FromStr for DType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" | "dyn" => Ok(Self::Auto),
            "u4" | "uint4" => Ok(Self::Uint4),
            "u8" | "uint8" => Ok(Self::Uint8),
            "u16" | "uint16" => Ok(Self::Uint16),
            "u32" | "uint32" => Ok(Self::Uint32),
            "u64" | "uint64" => Ok(Self::Uint64),
            "i4" | "int4" => Ok(Self::Int4),
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
            "q8" => Ok(Self::Q8),
            "bnb4" => Ok(Self::Bnb4),
            "f8e4m3fn" => Ok(Self::Fp8e4m3fn),
            "f8e4m3fnuz" => Ok(Self::Fp8e4m3fnuz),
            "f8e5m2" => Ok(Self::Fp8e5m2),
            "f8e5m2fnuz" => Ok(Self::Fp8e5m2fnuz),
            "f4e2m1" => Ok(Self::Fp4e2m1),
            "complex64" => Ok(Self::Complex64),
            "complex128" => Ok(Self::Complex128),
            x => anyhow::bail!("Unsupported DType: {}", x),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let x = match self {
            Self::Auto => "auto",
            Self::Int4 => "int4",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Uint4 => "uint4",
            Self::Uint8 => "uint8",
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
            Self::Uint64 => "uint64",
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
            Self::Fp64 => "fp64",
            Self::Bf16 => "bf16",
            Self::Bnb4 => "bnb4",
            Self::Q4 => "q4",
            Self::Q4f16 => "q4f16",
            Self::Q8 => "q8",
            Self::Fp8e4m3fn => "f8e4m3fn",
            Self::Fp8e4m3fnuz => "f8e4m3fnuz",
            Self::Fp8e5m2 => "f8e5m2",
            Self::Fp8e5m2fnuz => "f8e5m2fnuz",
            Self::Fp4e2m1 => "f4e2m1",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
        };
        write!(f, "{}", x)
    }
}
