use anyhow::Result;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{DType, DTypeTensor, Tensor};

impl Add<&Tensor> for Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: &Tensor) -> Self::Output {
        match (&self.data, &other.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                // Check if shapes are exactly the same
                if a.shape() == b.shape() {
                    let result = (a + b).into_shared();
                    return Ok(Tensor {
                        data: DTypeTensor::F32(result),
                        dtype: DType::Fp32,
                        uid: Self::generate_uid(),
                    });
                }

                // Try broadcasting: attempt to broadcast the smaller tensor to the larger shape
                let (a_broadcasted, b_broadcasted) = if a.len() >= b.len() {
                    // Broadcast b to a's shape
                    match b.broadcast(a.raw_dim()) {
                        Some(b_bc) => (a.clone(), b_bc.to_owned().into_shared()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                } else {
                    // Broadcast a to b's shape
                    match a.broadcast(b.raw_dim()) {
                        Some(a_bc) => (a_bc.to_owned().into_shared(), b.clone()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                };

                let result = (&a_broadcasted + &b_broadcasted).into_shared();
                Ok(Tensor {
                    data: DTypeTensor::F32(result),
                    dtype: DType::Fp32,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("Element-wise addition currently only supports F32 tensors"),
        }
    }
}

// Tensor - &Tensor
impl Sub<&Tensor> for Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: &Tensor) -> Self::Output {
        match (&self.data, &other.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                // Check if shapes are exactly the same
                if a.shape() == b.shape() {
                    let result = (a - b).into_shared();
                    return Ok(Tensor {
                        data: DTypeTensor::F32(result),
                        dtype: DType::Fp32,
                        uid: Self::generate_uid(),
                    });
                }

                // Try broadcasting: attempt to broadcast the smaller tensor to the larger shape
                let (a_broadcasted, b_broadcasted) = if a.len() >= b.len() {
                    // Broadcast b to a's shape
                    match b.broadcast(a.raw_dim()) {
                        Some(b_bc) => (a.clone(), b_bc.to_owned().into_shared()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                } else {
                    // Broadcast a to b's shape
                    match a.broadcast(b.raw_dim()) {
                        Some(a_bc) => (a_bc.to_owned().into_shared(), b.clone()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                };

                let result = (&a_broadcasted - &b_broadcasted).into_shared();
                Ok(Tensor {
                    data: DTypeTensor::F32(result),
                    dtype: DType::Fp32,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("Element-wise subtraction currently only supports F32 tensors"),
        }
    }
}

// Tensor * &Tensor
impl Mul<&Tensor> for Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: &Tensor) -> Self::Output {
        match (&self.data, &other.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                // Check if shapes are exactly the same
                if a.shape() == b.shape() {
                    let result = (a * b).into_shared();
                    return Ok(Tensor {
                        data: DTypeTensor::F32(result),
                        dtype: DType::Fp32,
                        uid: Self::generate_uid(),
                    });
                }

                // Try broadcasting: attempt to broadcast the smaller tensor to the larger shape
                let (a_broadcasted, b_broadcasted) = if a.len() >= b.len() {
                    // Broadcast b to a's shape
                    match b.broadcast(a.raw_dim()) {
                        Some(b_bc) => (a.clone(), b_bc.to_owned().into_shared()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                } else {
                    // Broadcast a to b's shape
                    match a.broadcast(b.raw_dim()) {
                        Some(a_bc) => (a_bc.to_owned().into_shared(), b.clone()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                };

                let result = (&a_broadcasted * &b_broadcasted).into_shared();
                Ok(Tensor {
                    data: DTypeTensor::F32(result),
                    dtype: DType::Fp32,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!("Element-wise multiplication currently only supports F32 tensors"),
        }
    }
}

// Tensor / &Tensor
impl Div<&Tensor> for Tensor {
    type Output = Result<Tensor>;

    fn div(self, other: &Tensor) -> Self::Output {
        // Ensure both tensors have the same dtype
        if self.dtype != other.dtype {
            anyhow::bail!(
                "Cannot perform division on tensors with different dtypes: {:?} and {:?}",
                self.dtype,
                other.dtype
            );
        }

        match (&self.data, &other.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                // Check if shapes are exactly the same
                if a.shape() == b.shape() {
                    let result = (a / b).into_shared();
                    return Ok(Tensor {
                        data: DTypeTensor::F32(result),
                        dtype: DType::Fp32,
                        uid: Self::generate_uid(),
                    });
                }

                // Try broadcasting: attempt to broadcast the smaller tensor to the larger shape
                let (a_broadcasted, b_broadcasted) = if a.len() >= b.len() {
                    // Broadcast b to a's shape
                    match b.broadcast(a.raw_dim()) {
                        Some(b_bc) => (a.clone(), b_bc.to_owned().into_shared()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                } else {
                    // Broadcast a to b's shape
                    match a.broadcast(b.raw_dim()) {
                        Some(a_bc) => (a_bc.to_owned().into_shared(), b.clone()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                };

                let result = (&a_broadcasted / &b_broadcasted).into_shared();
                Ok(Tensor {
                    data: DTypeTensor::F32(result),
                    dtype: DType::Fp32,
                    uid: Self::generate_uid(),
                })
            }
            (DTypeTensor::F64(a), DTypeTensor::F64(b)) => {
                // Check if shapes are exactly the same
                if a.shape() == b.shape() {
                    let result = (a / b).into_shared();
                    return Ok(Tensor {
                        data: DTypeTensor::F64(result),
                        dtype: DType::Fp64,
                        uid: Self::generate_uid(),
                    });
                }

                // Try broadcasting: attempt to broadcast the smaller tensor to the larger shape
                let (a_broadcasted, b_broadcasted) = if a.len() >= b.len() {
                    // Broadcast b to a's shape
                    match b.broadcast(a.raw_dim()) {
                        Some(b_bc) => (a.clone(), b_bc.to_owned().into_shared()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                } else {
                    // Broadcast a to b's shape
                    match a.broadcast(b.raw_dim()) {
                        Some(a_bc) => (a_bc.to_owned().into_shared(), b.clone()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                };

                let result = (&a_broadcasted / &b_broadcasted).into_shared();
                Ok(Tensor {
                    data: DTypeTensor::F64(result),
                    dtype: DType::Fp64,
                    uid: Self::generate_uid(),
                })
            }
            (DTypeTensor::I32(a), DTypeTensor::I32(b)) => {
                // Check if shapes are exactly the same
                if a.shape() == b.shape() {
                    let result = (a / b).into_shared();
                    return Ok(Tensor {
                        data: DTypeTensor::I32(result),
                        dtype: DType::Int32,
                        uid: Self::generate_uid(),
                    });
                }

                // Try broadcasting: attempt to broadcast the smaller tensor to the larger shape
                let (a_broadcasted, b_broadcasted) = if a.len() >= b.len() {
                    // Broadcast b to a's shape
                    match b.broadcast(a.raw_dim()) {
                        Some(b_bc) => (a.clone(), b_bc.to_owned().into_shared()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                } else {
                    // Broadcast a to b's shape
                    match a.broadcast(b.raw_dim()) {
                        Some(a_bc) => (a_bc.to_owned().into_shared(), b.clone()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                };

                let result = (&a_broadcasted / &b_broadcasted).into_shared();
                Ok(Tensor {
                    data: DTypeTensor::I32(result),
                    dtype: DType::Int32,
                    uid: Self::generate_uid(),
                })
            }
            (DTypeTensor::I64(a), DTypeTensor::I64(b)) => {
                // Check if shapes are exactly the same
                if a.shape() == b.shape() {
                    let result = (a / b).into_shared();
                    return Ok(Tensor {
                        data: DTypeTensor::I64(result),
                        dtype: DType::Int64,
                        uid: Self::generate_uid(),
                    });
                }

                // Try broadcasting: attempt to broadcast the smaller tensor to the larger shape
                let (a_broadcasted, b_broadcasted) = if a.len() >= b.len() {
                    // Broadcast b to a's shape
                    match b.broadcast(a.raw_dim()) {
                        Some(b_bc) => (a.clone(), b_bc.to_owned().into_shared()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                } else {
                    // Broadcast a to b's shape
                    match a.broadcast(b.raw_dim()) {
                        Some(a_bc) => (a_bc.to_owned().into_shared(), b.clone()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                };

                let result = (&a_broadcasted / &b_broadcasted).into_shared();
                Ok(Tensor {
                    data: DTypeTensor::I64(result),
                    dtype: DType::Int64,
                    uid: Self::generate_uid(),
                })
            }
            (DTypeTensor::F16(a), DTypeTensor::F16(b)) => {
                // Check if shapes are exactly the same
                if a.shape() == b.shape() {
                    let result = (a / b).into_shared();
                    return Ok(Tensor {
                        data: DTypeTensor::F16(result),
                        dtype: DType::Fp16,
                        uid: Self::generate_uid(),
                    });
                }

                // Try broadcasting: attempt to broadcast the smaller tensor to the larger shape
                let (a_broadcasted, b_broadcasted) = if a.len() >= b.len() {
                    // Broadcast b to a's shape
                    match b.broadcast(a.raw_dim()) {
                        Some(b_bc) => (a.clone(), b_bc.to_owned().into_shared()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                } else {
                    // Broadcast a to b's shape
                    match a.broadcast(b.raw_dim()) {
                        Some(a_bc) => (a_bc.to_owned().into_shared(), b.clone()),
                        None => anyhow::bail!(
                            "Cannot broadcast shapes: {:?} and {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    }
                };

                let result = (&a_broadcasted / &b_broadcasted).into_shared();
                Ok(Tensor {
                    data: DTypeTensor::F16(result),
                    dtype: DType::Fp16,
                    uid: Self::generate_uid(),
                })
            }
            _ => anyhow::bail!(
                "Element-wise division not supported for this tensor type combination"
            ),
        }
    }
}

// === Compound Assignment Operations ===

// DivAssign for scalar f32
impl DivAssign<f32> for Tensor {
    fn div_assign(&mut self, other: f32) {
        match &mut self.data {
            DTypeTensor::F32(arr) => {
                let result = arr.view().mapv(|x| x / other);
                self.data = DTypeTensor::F32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::F64(arr) => {
                let result = arr.view().mapv(|x| x / other as f64);
                self.data = DTypeTensor::F64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::F16(arr) => {
                let result = arr.view().mapv(|x| x / half::f16::from_f32(other));
                self.data = DTypeTensor::F16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::Bf16(arr) => {
                let result = arr.view().mapv(|x| x / half::bf16::from_f32(other));
                self.data = DTypeTensor::Bf16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I8(arr) => {
                let result = arr.view().mapv(|x| x / other as i8);
                self.data = DTypeTensor::I8(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I16(arr) => {
                let result = arr.view().mapv(|x| x / other as i16);
                self.data = DTypeTensor::I16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I32(arr) => {
                let result = arr.view().mapv(|x| x / other as i32);
                self.data = DTypeTensor::I32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I64(arr) => {
                let result = arr.view().mapv(|x| x / other as i64);
                self.data = DTypeTensor::I64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U8(arr) => {
                let result = arr.view().mapv(|x| x / other as u8);
                self.data = DTypeTensor::U8(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U16(arr) => {
                let result = arr.view().mapv(|x| x / other as u16);
                self.data = DTypeTensor::U16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U32(arr) => {
                let result = arr.view().mapv(|x| x / other as u32);
                self.data = DTypeTensor::U32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U64(arr) => {
                let result = arr.view().mapv(|x| x / other as u64);
                self.data = DTypeTensor::U64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::Bool(_) => panic!("Division assignment is not supported for Bool tensors"),
        }
    }
}

// DivAssign for tensor reference
impl DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, other: &Tensor) {
        let result = self.clone().div(other).expect("Division failed");
        *self = result;
    }
}

// MulAssign for scalar f32
impl MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, other: f32) {
        match &mut self.data {
            DTypeTensor::F32(arr) => {
                let result = arr.view().mapv(|x| x * other);
                self.data = DTypeTensor::F32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::F64(arr) => {
                let result = arr.view().mapv(|x| x * other as f64);
                self.data = DTypeTensor::F64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::F16(arr) => {
                let result = arr.view().mapv(|x| x * half::f16::from_f32(other));
                self.data = DTypeTensor::F16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::Bf16(arr) => {
                let result = arr.view().mapv(|x| x * half::bf16::from_f32(other));
                self.data = DTypeTensor::Bf16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I8(arr) => {
                let result = arr.view().mapv(|x| x * other as i8);
                self.data = DTypeTensor::I8(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I16(arr) => {
                let result = arr.view().mapv(|x| x * other as i16);
                self.data = DTypeTensor::I16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I32(arr) => {
                let result = arr.view().mapv(|x| x * other as i32);
                self.data = DTypeTensor::I32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I64(arr) => {
                let result = arr.view().mapv(|x| x * other as i64);
                self.data = DTypeTensor::I64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U8(arr) => {
                let result = arr.view().mapv(|x| x * other as u8);
                self.data = DTypeTensor::U8(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U16(arr) => {
                let result = arr.view().mapv(|x| x * other as u16);
                self.data = DTypeTensor::U16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U32(arr) => {
                let result = arr.view().mapv(|x| x * other as u32);
                self.data = DTypeTensor::U32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U64(arr) => {
                let result = arr.view().mapv(|x| x * other as u64);
                self.data = DTypeTensor::U64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::Bool(_) => {
                panic!("Multiplication assignment is not supported for Bool tensors")
            }
        }
    }
}

// MulAssign for tensor reference
impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        let result = self.clone().mul(other).expect("Multiplication failed");
        *self = result;
    }
}

// AddAssign for scalar f32
impl AddAssign<f32> for Tensor {
    fn add_assign(&mut self, other: f32) {
        match &mut self.data {
            DTypeTensor::F32(arr) => {
                let result = arr.view().mapv(|x| x + other);
                self.data = DTypeTensor::F32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::F64(arr) => {
                let result = arr.view().mapv(|x| x + other as f64);
                self.data = DTypeTensor::F64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::F16(arr) => {
                let result = arr.view().mapv(|x| x + half::f16::from_f32(other));
                self.data = DTypeTensor::F16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::Bf16(arr) => {
                let result = arr.view().mapv(|x| x + half::bf16::from_f32(other));
                self.data = DTypeTensor::Bf16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I8(arr) => {
                let result = arr.view().mapv(|x| x + other as i8);
                self.data = DTypeTensor::I8(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I16(arr) => {
                let result = arr.view().mapv(|x| x + other as i16);
                self.data = DTypeTensor::I16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I32(arr) => {
                let result = arr.view().mapv(|x| x + other as i32);
                self.data = DTypeTensor::I32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I64(arr) => {
                let result = arr.view().mapv(|x| x + other as i64);
                self.data = DTypeTensor::I64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U8(arr) => {
                let result = arr.view().mapv(|x| x + other as u8);
                self.data = DTypeTensor::U8(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U16(arr) => {
                let result = arr.view().mapv(|x| x + other as u16);
                self.data = DTypeTensor::U16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U32(arr) => {
                let result = arr.view().mapv(|x| x + other as u32);
                self.data = DTypeTensor::U32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U64(arr) => {
                let result = arr.view().mapv(|x| x + other as u64);
                self.data = DTypeTensor::U64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::Bool(_) => panic!("Addition assignment is not supported for Bool tensors"),
        }
    }
}

/// Element-wise addition
impl Add for Tensor {
    type Output = Result<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        match (&self.data, &rhs.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F64(a), DTypeTensor::F64(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F16(a), DTypeTensor::F16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::Bf16(a), DTypeTensor::Bf16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I8(a), DTypeTensor::I8(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I16(a), DTypeTensor::I16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Addition requires tensors of the same data type"),
        }
    }
}

/// Element-wise addition with reference
impl Add<&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, rhs: &Tensor) -> Self::Output {
        match (&self.data, &rhs.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F64(a), DTypeTensor::F64(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F16(a), DTypeTensor::F16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::Bf16(a), DTypeTensor::Bf16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I8(a), DTypeTensor::I8(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I16(a), DTypeTensor::I16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for addition: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a + b;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Addition requires tensors of the same data type"),
        }
    }
}

/// Element-wise subtraction
impl Sub for Tensor {
    type Output = Result<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (&self.data, &rhs.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F64(a), DTypeTensor::F64(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F16(a), DTypeTensor::F16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::Bf16(a), DTypeTensor::Bf16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I8(a), DTypeTensor::I8(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I16(a), DTypeTensor::I16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Subtraction requires tensors of the same data type"),
        }
    }
}

/// Element-wise subtraction with reference
impl Sub<&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        match (&self.data, &rhs.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F64(a), DTypeTensor::F64(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F16(a), DTypeTensor::F16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::Bf16(a), DTypeTensor::Bf16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I8(a), DTypeTensor::I8(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I16(a), DTypeTensor::I16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for subtraction: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a - b;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Subtraction requires tensors of the same data type"),
        }
    }
}

/// Element-wise multiplication
impl Mul for Tensor {
    type Output = Result<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (&self.data, &rhs.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F64(a), DTypeTensor::F64(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F16(a), DTypeTensor::F16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::Bf16(a), DTypeTensor::Bf16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I8(a), DTypeTensor::I8(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I16(a), DTypeTensor::I16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Multiplication requires tensors of the same data type"),
        }
    }
}

/// Element-wise multiplication with reference
impl Mul<&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        match (&self.data, &rhs.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::F16(a), DTypeTensor::F16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::Bf16(a), DTypeTensor::Bf16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I8(a), DTypeTensor::I8(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            (DTypeTensor::I16(a), DTypeTensor::I16(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for multiplication: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a * b;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Multiplication requires tensors of the same data type"),
        }
    }
}

/// Scalar multiplication
impl Mul<f32> for Tensor {
    type Output = Result<Self>;

    fn mul(self, rhs: f32) -> Self::Output {
        match &self.data {
            DTypeTensor::F32(a) => {
                let result = a * rhs;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Scalar multiplication currently only supports F32 tensors"),
        }
    }
}

/// Scalar multiplication with reference
impl Mul<f32> for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, rhs: f32) -> Self::Output {
        match &self.data {
            DTypeTensor::F32(a) => {
                let result = a * rhs;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Scalar multiplication currently only supports F32 tensors"),
        }
    }
}

/// Element-wise division
impl Div for Tensor {
    type Output = Result<Self>;

    fn div(self, rhs: Self) -> Self::Output {
        match (&self.data, &rhs.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for division: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a / b;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Division currently only supports F32 tensors"),
        }
    }
}

/// Element-wise division with reference
impl Div<&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn div(self, rhs: &Tensor) -> Self::Output {
        match (&self.data, &rhs.data) {
            (DTypeTensor::F32(a), DTypeTensor::F32(b)) => {
                if a.shape() != b.shape() {
                    anyhow::bail!(
                        "Shape mismatch for division: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    );
                }
                let result = a / b;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Division currently only supports F32 tensors"),
        }
    }
}

/// Scalar division
impl Div<f32> for Tensor {
    type Output = Result<Self>;

    fn div(self, rhs: f32) -> Self::Output {
        match &self.data {
            DTypeTensor::F32(a) => {
                let result = a / rhs;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Scalar division currently only supports F32 tensors"),
        }
    }
}

/// Scalar division with reference
impl Div<f32> for &Tensor {
    type Output = Result<Tensor>;

    fn div(self, rhs: f32) -> Self::Output {
        match &self.data {
            DTypeTensor::F32(a) => {
                let result = a / rhs;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Scalar division currently only supports F32 tensors"),
        }
    }
}

/// Negation
impl Neg for Tensor {
    type Output = Result<Self>;

    fn neg(self) -> Self::Output {
        match &self.data {
            DTypeTensor::F32(a) => {
                let result = -a;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Negation currently only supports F32 tensors"),
        }
    }
}

/// Negation with reference
impl Neg for &Tensor {
    type Output = Result<Tensor>;

    fn neg(self) -> Self::Output {
        match &self.data {
            DTypeTensor::F32(a) => {
                let result = -a;
                Ok(result.into_dyn().into())
            }
            _ => anyhow::bail!("Negation currently only supports F32 tensors"),
        }
    }
}

// AddAssign for tensor reference
impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, other: &Tensor) {
        let result = self.clone().add(other).expect("Addition failed");
        *self = result;
    }
}

// SubAssign for scalar f32
impl SubAssign<f32> for Tensor {
    fn sub_assign(&mut self, other: f32) {
        match &mut self.data {
            DTypeTensor::F32(arr) => {
                let result = arr.view().mapv(|x| x - other);
                self.data = DTypeTensor::F32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::F64(arr) => {
                let result = arr.view().mapv(|x| x - other as f64);
                self.data = DTypeTensor::F64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::F16(arr) => {
                let result = arr.view().mapv(|x| x - half::f16::from_f32(other));
                self.data = DTypeTensor::F16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::Bf16(arr) => {
                let result = arr.view().mapv(|x| x - half::bf16::from_f32(other));
                self.data = DTypeTensor::Bf16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I8(arr) => {
                let result = arr.view().mapv(|x| x - other as i8);
                self.data = DTypeTensor::I8(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I16(arr) => {
                let result = arr.view().mapv(|x| x - other as i16);
                self.data = DTypeTensor::I16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I32(arr) => {
                let result = arr.view().mapv(|x| x - other as i32);
                self.data = DTypeTensor::I32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::I64(arr) => {
                let result = arr.view().mapv(|x| x - other as i64);
                self.data = DTypeTensor::I64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U8(arr) => {
                let result = arr.view().mapv(|x| x - other as u8);
                self.data = DTypeTensor::U8(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U16(arr) => {
                let result = arr.view().mapv(|x| x - other as u16);
                self.data = DTypeTensor::U16(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U32(arr) => {
                let result = arr.view().mapv(|x| x - other as u32);
                self.data = DTypeTensor::U32(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::U64(arr) => {
                let result = arr.view().mapv(|x| x - other as u64);
                self.data = DTypeTensor::U64(result.into_shared());
                self.uid = Self::generate_uid();
            }
            DTypeTensor::Bool(_) => {
                panic!("Subtraction assignment is not supported for Bool tensors")
            }
        }
    }
}

// SubAssign for tensor reference
impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        let result = self.clone().sub(other).expect("Subtraction failed");
        *self = result;
    }
}
