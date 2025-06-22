//! Macros for reducing boilerplate in tensor operations
//!
//! This module provides macros to eliminate repetitive pattern matching
//! across different tensor data types, improving code maintainability
//! and reducing the chance of errors.

/// Macro for applying the same operation to all tensor view data types
///
/// This macro generates a match statement that applies the same operation
/// to all variants of TensorViewData, reducing boilerplate code.
///
/// # Arguments
///
/// * `$data` - The TensorViewData to match on
/// * `$view_var` - Variable name for the view in each match arm
/// * `$operation` - The operation to apply to each view
///
/// # Example
/// ```rust,ignore
/// tensor_view_match!(&self.data, view, {
///     let owned = view.to_owned().into_dyn();
///     Ok(Tensor::from_array(owned))
/// })
/// ```
macro_rules! tensor_view_match {
    ($data:expr, $view_var:ident, $operation:block) => {
        match $data {
            TensorViewData::F32($view_var) => $operation,
            TensorViewData::F64($view_var) => $operation,
            TensorViewData::F16($view_var) => $operation,
            TensorViewData::Bf16($view_var) => $operation,
            TensorViewData::I8($view_var) => $operation,
            TensorViewData::I16($view_var) => $operation,
            TensorViewData::I32($view_var) => $operation,
            TensorViewData::I64($view_var) => $operation,
            TensorViewData::U8($view_var) => $operation,
            TensorViewData::U16($view_var) => $operation,
            TensorViewData::U32($view_var) => $operation,
            TensorViewData::U64($view_var) => $operation,
            TensorViewData::Bool($view_var) => $operation,
        }
    };
}

/// Macro for applying the same operation to all mutable tensor view data types
///
/// Similar to `tensor_view_match!` but for TensorViewMutData.
///
/// # Arguments
///
/// * `$data` - The TensorViewMutData to match on
/// * `$view_var` - Variable name for the view in each match arm
/// * `$operation` - The operation to apply to each view
macro_rules! tensor_view_mut_match {
    ($data:expr, $view_var:ident, $operation:block) => {
        match $data {
            TensorViewMutData::F32($view_var) => $operation,
            TensorViewMutData::F64($view_var) => $operation,
            TensorViewMutData::F16($view_var) => $operation,
            TensorViewMutData::Bf16($view_var) => $operation,
            TensorViewMutData::I8($view_var) => $operation,
            TensorViewMutData::I16($view_var) => $operation,
            TensorViewMutData::I32($view_var) => $operation,
            TensorViewMutData::I64($view_var) => $operation,
            TensorViewMutData::U8($view_var) => $operation,
            TensorViewMutData::U16($view_var) => $operation,
            TensorViewMutData::U32($view_var) => $operation,
            TensorViewMutData::U64($view_var) => $operation,
            TensorViewMutData::Bool($view_var) => $operation,
        }
    };
}

/// Macro for handling DTypeTensor operations that return a new tensor
/// Useful for operations like reshape, transpose, etc.
///
/// # Arguments
/// * `$data` - The DTypeTensor data to match against
/// * `$var` - Variable name to bind the matched array
/// * `$operation` - Code block that transforms the array and returns new data
/// * `$dtype` - The dtype field to preserve
///
/// # Example
/// ```rust,ignore
/// dtype_tensor_transform_match!(&self.data, arr, {
///     arr.clone().into_shape_with_order(shape)?.into_dyn().into_shared()
/// }, self.dtype)
/// ```
macro_rules! dtype_tensor_transform_match {
    ($data:expr, $var:ident, $operation:block, $dtype:expr) => {
        match $data {
            DTypeTensor::F32($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::F32(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::F64($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::F64(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::F16($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::F16(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::Bf16($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::Bf16(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::I8($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::I8(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::I16($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::I16(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::I32($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::I32(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::I64($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::I64(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::U8($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::U8(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::U16($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::U16(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::U32($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::U32(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::U64($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::U64(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
            DTypeTensor::Bool($var) => {
                let result = $operation;
                Ok(Self {
                    data: DTypeTensor::Bool(result),
                    dtype: $dtype,
                    uid: Self::generate_uid(),
                })
            }
        }
    };
}

/// Macro for handling DTypeTensor operations that update the tensor in place
/// Useful for operations like reshape that consume self
///
/// # Arguments
/// * `$self` - Mutable self reference
/// * `$var` - Variable name to bind the matched array
/// * `$operation` - Code block that transforms the array and returns new data
///
/// # Example
/// ```rust,ignore
/// dtype_tensor_self_update_match!(self, arr, {
///     arr.clone().into_shape_with_order(shape)?.into_dyn().into_shared()
/// })
/// ```
macro_rules! dtype_tensor_self_update_match {
    ($self:ident, $var:ident, $operation:block) => {
        match &$self.data {
            DTypeTensor::F32($var) => {
                let result = $operation;
                $self.data = DTypeTensor::F32(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::F64($var) => {
                let result = $operation;
                $self.data = DTypeTensor::F64(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::F16($var) => {
                let result = $operation;
                $self.data = DTypeTensor::F16(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::Bf16($var) => {
                let result = $operation;
                $self.data = DTypeTensor::Bf16(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::I8($var) => {
                let result = $operation;
                $self.data = DTypeTensor::I8(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::I16($var) => {
                let result = $operation;
                $self.data = DTypeTensor::I16(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::I32($var) => {
                let result = $operation;
                $self.data = DTypeTensor::I32(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::I64($var) => {
                let result = $operation;
                $self.data = DTypeTensor::I64(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::U8($var) => {
                let result = $operation;
                $self.data = DTypeTensor::U8(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::U16($var) => {
                let result = $operation;
                $self.data = DTypeTensor::U16(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::U32($var) => {
                let result = $operation;
                $self.data = DTypeTensor::U32(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::U64($var) => {
                let result = $operation;
                $self.data = DTypeTensor::U64(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
            DTypeTensor::Bool($var) => {
                let result = $operation;
                $self.data = DTypeTensor::Bool(result);
                $self.uid = Self::generate_uid();
                Ok($self)
            }
        }
    };
}

pub(crate) use dtype_tensor_self_update_match;
pub(crate) use dtype_tensor_transform_match;
pub(crate) use tensor_view_match;
pub(crate) use tensor_view_mut_match;
