//! Unary operations for Tensor
//!
//! This module provides unary mathematical operations for tensors,
//! including activation functions, trigonometric functions, and basic math operations.

use crate::tensor::{DTypeTensor, Tensor};
use anyhow::Result;
use half::{bf16, f16};

/// Macro to generate unary operations for all supported data types
macro_rules! unary_op {
    ($method_name:ident, $op_trait:ident) => {
        impl Tensor {
            /// Apply $method_name operation element-wise
            pub fn $method_name(&self) -> Result<Self> {
                let new_data = match &self.data {
                    DTypeTensor::F32(arr) => {
                        let result = arr.mapv(|x| x.$method_name());
                        DTypeTensor::F32(result.into_shared())
                    }
                    DTypeTensor::F64(arr) => {
                        let result = arr.mapv(|x| x.$method_name());
                        DTypeTensor::F64(result.into_shared())
                    }
                    DTypeTensor::F16(arr) => {
                        let result = arr.mapv(|x| f16::from_f32(x.to_f32().$method_name()));
                        DTypeTensor::F16(result.into_shared())
                    }
                    DTypeTensor::Bf16(arr) => {
                        let result = arr.mapv(|x| bf16::from_f32(x.to_f32().$method_name()));
                        DTypeTensor::Bf16(result.into_shared())
                    }
                    _ => {
                        anyhow::bail!(
                            "{} operation is only supported for floating-point tensors",
                            stringify!($method_name)
                        );
                    }
                };

                Ok(Self {
                    data: new_data,
                    dtype: self.dtype,
                    uid: Self::generate_uid(),
                })
            }
        }
    };
}

// Generate floating-point unary operations
unary_op!(exp, Exp);
unary_op!(sin, Sin);
unary_op!(cos, Cos);
// Custom implementation for log (natural logarithm)
impl Tensor {
    /// Natural logarithm operation: ln(x)
    pub fn log(&self) -> Result<Self> {
        let new_data = match &self.data {
            DTypeTensor::F32(arr) => {
                let result = arr.mapv(|x| x.ln());
                DTypeTensor::F32(result.into_shared())
            }
            DTypeTensor::F64(arr) => {
                let result = arr.mapv(|x| x.ln());
                DTypeTensor::F64(result.into_shared())
            }
            DTypeTensor::F16(arr) => {
                let result = arr.mapv(|x| f16::from_f32(x.to_f32().ln()));
                DTypeTensor::F16(result.into_shared())
            }
            DTypeTensor::Bf16(arr) => {
                let result = arr.mapv(|x| bf16::from_f32(x.to_f32().ln()));
                DTypeTensor::Bf16(result.into_shared())
            }
            _ => {
                anyhow::bail!(
                    "Natural logarithm operation is only supported for floating-point tensors"
                );
            }
        };

        Ok(Self {
            data: new_data,
            dtype: self.dtype,
            uid: Self::generate_uid(),
        })
    }
}
unary_op!(tanh, Tanh);
unary_op!(floor, Floor);
unary_op!(ceil, Ceil);
unary_op!(round, Round);
unary_op!(sqrt, Sqrt);
unary_op!(recip, Recip);

// Custom implementation for abs operation
impl Tensor {
    /// Absolute value operation: |x|
    pub fn abs(&self) -> Result<Self> {
        let new_data = match &self.data {
            DTypeTensor::F32(arr) => {
                let result = arr.mapv(|x| x.abs());
                DTypeTensor::F32(result.into_shared())
            }
            DTypeTensor::F64(arr) => {
                let result = arr.mapv(|x| x.abs());
                DTypeTensor::F64(result.into_shared())
            }
            DTypeTensor::F16(arr) => {
                let result = arr.mapv(|x| f16::from_f32(x.to_f32().abs()));
                DTypeTensor::F16(result.into_shared())
            }
            DTypeTensor::Bf16(arr) => {
                let result = arr.mapv(|x| bf16::from_f32(x.to_f32().abs()));
                DTypeTensor::Bf16(result.into_shared())
            }
            DTypeTensor::I8(arr) => {
                let result = arr.mapv(|x| x.wrapping_abs());
                DTypeTensor::I8(result.into_shared())
            }
            DTypeTensor::I16(arr) => {
                let result = arr.mapv(|x| x.wrapping_abs());
                DTypeTensor::I16(result.into_shared())
            }
            DTypeTensor::I32(arr) => {
                let result = arr.mapv(|x| x.wrapping_abs());
                DTypeTensor::I32(result.into_shared())
            }
            DTypeTensor::I64(arr) => {
                let result = arr.mapv(|x| x.wrapping_abs());
                DTypeTensor::I64(result.into_shared())
            }
            DTypeTensor::U8(arr) => {
                // For unsigned types, abs is identity
                DTypeTensor::U8(arr.clone())
            }
            DTypeTensor::U16(arr) => DTypeTensor::U16(arr.clone()),
            DTypeTensor::U32(arr) => DTypeTensor::U32(arr.clone()),
            DTypeTensor::U64(arr) => DTypeTensor::U64(arr.clone()),
            DTypeTensor::Bool(_) => {
                anyhow::bail!("Absolute value operation is not supported for boolean tensors");
            }
        };

        Ok(Self {
            data: new_data,
            dtype: self.dtype,
            uid: Self::generate_uid(),
        })
    }
}

// Special implementations for operations that need custom logic
impl Tensor {
    /// ReLU activation function: max(0, x)
    pub fn relu(&self) -> Result<Self> {
        let new_data = match &self.data {
            DTypeTensor::F32(arr) => {
                let result = arr.mapv(|x| x.max(0.0));
                DTypeTensor::F32(result.into_shared())
            }
            DTypeTensor::F64(arr) => {
                let result = arr.mapv(|x| x.max(0.0));
                DTypeTensor::F64(result.into_shared())
            }
            DTypeTensor::F16(arr) => {
                let result = arr.mapv(|x| f16::from_f32(x.to_f32().max(0.0)));
                DTypeTensor::F16(result.into_shared())
            }
            DTypeTensor::Bf16(arr) => {
                let result = arr.mapv(|x| bf16::from_f32(x.to_f32().max(0.0)));
                DTypeTensor::Bf16(result.into_shared())
            }
            DTypeTensor::I8(arr) => {
                let result = arr.mapv(|x| x.max(0));
                DTypeTensor::I8(result.into_shared())
            }
            DTypeTensor::I16(arr) => {
                let result = arr.mapv(|x| x.max(0));
                DTypeTensor::I16(result.into_shared())
            }
            DTypeTensor::I32(arr) => {
                let result = arr.mapv(|x| x.max(0));
                DTypeTensor::I32(result.into_shared())
            }
            DTypeTensor::I64(arr) => {
                let result = arr.mapv(|x| x.max(0));
                DTypeTensor::I64(result.into_shared())
            }
            DTypeTensor::U8(arr) => {
                // For unsigned types, ReLU is identity since they're already >= 0
                DTypeTensor::U8(arr.clone())
            }
            DTypeTensor::U16(arr) => DTypeTensor::U16(arr.clone()),
            DTypeTensor::U32(arr) => DTypeTensor::U32(arr.clone()),
            DTypeTensor::U64(arr) => DTypeTensor::U64(arr.clone()),
            DTypeTensor::Bool(_) => {
                anyhow::bail!("ReLU operation is not supported for boolean tensors");
            }
        };

        Ok(Self {
            data: new_data,
            dtype: self.dtype,
            uid: Self::generate_uid(),
        })
    }

    /// Sigmoid activation function: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Result<Self> {
        let new_data = match &self.data {
            DTypeTensor::F32(arr) => {
                let result = arr.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                DTypeTensor::F32(result.into_shared())
            }
            DTypeTensor::F64(arr) => {
                let result = arr.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                DTypeTensor::F64(result.into_shared())
            }
            DTypeTensor::F16(arr) => {
                let result = arr.mapv(|x| {
                    let x_f32 = x.to_f32();
                    f16::from_f32(1.0 / (1.0 + (-x_f32).exp()))
                });
                DTypeTensor::F16(result.into_shared())
            }
            DTypeTensor::Bf16(arr) => {
                let result = arr.mapv(|x| {
                    let x_f32 = x.to_f32();
                    bf16::from_f32(1.0 / (1.0 + (-x_f32).exp()))
                });
                DTypeTensor::Bf16(result.into_shared())
            }
            _ => {
                anyhow::bail!("Sigmoid operation is only supported for floating-point tensors");
            }
        };

        Ok(Self {
            data: new_data,
            dtype: self.dtype,
            uid: Self::generate_uid(),
        })
    }

    /// SiLU (Swish) activation function: x * sigmoid(x)
    pub fn silu(&self) -> Result<Self> {
        let sigmoid_result = self.sigmoid()?;
        self * &sigmoid_result
    }

    /// Negation operation: -x
    pub fn neg(&self) -> Result<Self> {
        let new_data = match &self.data {
            DTypeTensor::F32(arr) => {
                let result = arr.mapv(|x| -x);
                DTypeTensor::F32(result.into_shared())
            }
            DTypeTensor::F64(arr) => {
                let result = arr.mapv(|x| -x);
                DTypeTensor::F64(result.into_shared())
            }
            DTypeTensor::F16(arr) => {
                let result = arr.mapv(|x| -x);
                DTypeTensor::F16(result.into_shared())
            }
            DTypeTensor::Bf16(arr) => {
                let result = arr.mapv(|x| -x);
                DTypeTensor::Bf16(result.into_shared())
            }
            DTypeTensor::I8(arr) => {
                let result = arr.mapv(|x| -x);
                DTypeTensor::I8(result.into_shared())
            }
            DTypeTensor::I16(arr) => {
                let result = arr.mapv(|x| -x);
                DTypeTensor::I16(result.into_shared())
            }
            DTypeTensor::I32(arr) => {
                let result = arr.mapv(|x| -x);
                DTypeTensor::I32(result.into_shared())
            }
            DTypeTensor::I64(arr) => {
                let result = arr.mapv(|x| -x);
                DTypeTensor::I64(result.into_shared())
            }
            DTypeTensor::U8(_)
            | DTypeTensor::U16(_)
            | DTypeTensor::U32(_)
            | DTypeTensor::U64(_) => {
                anyhow::bail!("Negation operation is not supported for unsigned integer tensors");
            }
            DTypeTensor::Bool(_) => {
                anyhow::bail!("Negation operation is not supported for boolean tensors");
            }
        };

        Ok(Self {
            data: new_data,
            dtype: self.dtype,
            uid: Self::generate_uid(),
        })
    }
}
