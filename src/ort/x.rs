//! ORT Tensor conversion for `X<A>` and `XView<A>`
//!
//! # Conversion Overview
//!
//! | Input Type | Output Lifetime | Zero-Copy | Notes |
//! |------------|-----------------|-----------|-------|
//! | `X<A>` (owned) | `'static` | ✅* | *if standard layout |
//! | `XView<'a, A>` | `'a` | ✅ | Always zero-copy |
//! | `&'a X<A>` via `x.view()` | `'a` | ✅ | Recommended for references |
//!
//! For `&[X]` inputs to `Engine::run()`, the engine automatically uses
//! zero-copy when dtype matches and data is in standard layout.

use ::ort::value::Tensor as OrtTensor;

impl<A> TryFrom<&crate::X<A>> for OrtTensor<A>
where
    A: Clone + 'static + ::ort::tensor::PrimitiveTensorElementType + std::fmt::Debug,
{
    type Error = anyhow::Error;

    fn try_from(x: &crate::X<A>) -> std::result::Result<Self, Self::Error> {
        let shape = x.0.shape().to_vec();
        let data: Vec<A> = x.0.as_standard_layout().iter().cloned().collect();

        Ok(OrtTensor::<A>::from_array((shape, data))?)
    }
}

impl<A> TryFrom<&&crate::X<A>> for OrtTensor<A>
where
    A: Clone + 'static + ::ort::tensor::PrimitiveTensorElementType + std::fmt::Debug,
{
    type Error = anyhow::Error;

    fn try_from(x: &&crate::X<A>) -> std::result::Result<Self, Self::Error> {
        (*x).try_into()
    }
}

impl<A> TryFrom<crate::X<A>> for OrtTensor<A>
where
    A: Clone + 'static + ::ort::tensor::PrimitiveTensorElementType + std::fmt::Debug,
{
    type Error = anyhow::Error;

    fn try_from(x: crate::X<A>) -> std::result::Result<Self, Self::Error> {
        let shape = x.0.shape().to_vec();
        let arr = x.0;
        let len = arr.len();
        let data: Vec<A> = if arr.is_standard_layout() {
            // Standard layout: extract Vec directly (no copy)
            let (mut data, offset) = arr.into_raw_vec_and_offset();
            if let Some(off) = offset {
                if off != 0 && off < data.len() {
                    data = data.split_off(off);
                }
            }
            data.truncate(len);
            data
        } else {
            // Non-standard layout: must copy to contiguous memory
            arr.as_standard_layout().iter().cloned().collect()
        };

        Ok(OrtTensor::<A>::from_array((shape, data))?)
    }
}

impl<A> TryFrom<crate::X<A>> for ::ort::session::SessionInputValue<'static>
where
    A: Clone + 'static + ::ort::tensor::PrimitiveTensorElementType + std::fmt::Debug,
{
    type Error = anyhow::Error;

    fn try_from(x: crate::X<A>) -> std::result::Result<Self, Self::Error> {
        Ok(OrtTensor::<A>::try_from(x)?.into())
    }
}

/// Zero-copy conversion from XView to SessionInputValue (with auto-fallback).
///
/// - **Contiguous layout**: zero-copy via `TensorRef::from_array_view`
/// - **Non-contiguous layout**: auto-fallback to copy (creates owned tensor)
impl<'a, A> TryFrom<crate::XView<'a, A>> for ::ort::session::SessionInputValue<'a>
where
    A: Clone + 'static + ::ort::tensor::PrimitiveTensorElementType + std::fmt::Debug,
{
    type Error = anyhow::Error;

    fn try_from(view: crate::XView<'a, A>) -> std::result::Result<Self, Self::Error> {
        if view.0.is_standard_layout() {
            // Fast path: zero-copy
            let tensor_ref = ::ort::value::TensorRef::from_array_view(view.0)?;
            Ok(::ort::session::SessionInputValue::from(tensor_ref))
        } else {
            // Fallback: copy to contiguous memory (non-standard layout)
            let shape = view.0.shape().to_vec();
            let data: Vec<A> = view.0.as_standard_layout().iter().cloned().collect();
            let tensor = ::ort::value::Tensor::<A>::from_array((shape, data))?;

            Ok(::ort::session::SessionInputValue::from(tensor))
        }
    }
}

/// Conversion from &X to SessionInputValue.
///
/// **Note**: This implementation requires `'static` lifetime, which means
/// it must clone the data. For zero-copy conversion, use `x.view()` instead:
///
/// ```ignore
/// // ❌ Clones data (requires 'static)
/// let input = SessionInputValue::try_from(&x)?;
///
/// // ✅ Zero-copy (uses actual lifetime)
/// let input = SessionInputValue::try_from(x.view())?;
/// ```
impl<A> TryFrom<&crate::X<A>> for ::ort::session::SessionInputValue<'static>
where
    A: Clone + 'static + ::ort::tensor::PrimitiveTensorElementType + std::fmt::Debug,
{
    type Error = anyhow::Error;

    fn try_from(x: &crate::X<A>) -> std::result::Result<Self, Self::Error> {
        Ok(OrtTensor::<A>::try_from(x)?.into())
    }
}
