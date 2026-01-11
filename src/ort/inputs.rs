use ort::session::{input::SessionInputs, SessionInputValue};

use crate::{XAny, X};

/// Pre-converted ORT input value (owned), intended for dynamic inputs.
///
/// This is a `usls`-level alias so call sites do not need to import `ort` types.
pub type OrtInput = SessionInputValue<'static>;

/// A list of pre-converted ORT input values (owned).
pub type OrtInputs = Vec<OrtInput>;

/// Unified input type for `Engine::run()`.
///
/// Supports multiple input formats with automatic dtype alignment:
/// - `inputs![...]` macro output
/// - `OrtInputs` / `Vec<SessionInputValue>`
/// - `&[X]` / `&Vec<X>` slices
/// - `&[XAny]` / `&XAny` (supports zero-copy CUDA)
pub enum EngineInputs<'a, 'i, 'v, const N: usize> {
    /// Pre-converted ORT inputs (from inputs! or manual conversion)
    Session(SessionInputs<'i, 'v, N>),
    /// Raw X slices that need dtype alignment
    XSlice(&'a [X]),
    /// XAny slice (supports CUDA zero-copy)
    ProcessedSlice(&'a [XAny]),
}

impl<'a, 'i, 'v, const N: usize> From<SessionInputs<'i, 'v, N>> for EngineInputs<'a, 'i, 'v, N> {
    fn from(inputs: SessionInputs<'i, 'v, N>) -> Self {
        EngineInputs::Session(inputs)
    }
}

impl<'a> From<&'a [X]> for EngineInputs<'a, 'static, 'static, 0> {
    fn from(xs: &'a [X]) -> Self {
        EngineInputs::XSlice(xs)
    }
}

impl<'a> From<&'a Vec<X>> for EngineInputs<'a, 'static, 'static, 0> {
    fn from(xs: &'a Vec<X>) -> Self {
        EngineInputs::XSlice(xs.as_slice())
    }
}

// Allow fixed-size arrays from inputs! macro
impl<'a, 'v, const N: usize> From<[SessionInputValue<'v>; N]> for EngineInputs<'a, 'v, 'v, N> {
    fn from(arr: [SessionInputValue<'v>; N]) -> Self {
        EngineInputs::Session(SessionInputs::ValueArray(arr))
    }
}

// Allow OrtInputs (Vec<SessionInputValue>)
impl<'a, const N: usize> From<OrtInputs> for EngineInputs<'a, 'static, 'static, N> {
    fn from(inputs: OrtInputs) -> Self {
        EngineInputs::Session(SessionInputs::ValueMap(
            inputs
                .into_iter()
                .enumerate()
                .map(|(i, v)| (std::borrow::Cow::Owned(i.to_string()), v))
                .collect(),
        ))
    }
}

// Allow &[SessionInputValue] slice
impl<'a, 'i, 'v, const N: usize> From<&'i [SessionInputValue<'v>]> for EngineInputs<'a, 'i, 'v, N> {
    fn from(slice: &'i [SessionInputValue<'v>]) -> Self {
        EngineInputs::Session(SessionInputs::ValueSlice(slice))
    }
}

// Allow XAny inputs
impl<'a> From<&'a [XAny]> for EngineInputs<'a, 'static, 'static, 0> {
    fn from(tensors: &'a [XAny]) -> Self {
        EngineInputs::ProcessedSlice(tensors)
    }
}

impl<'a> From<&'a XAny> for EngineInputs<'a, 'static, 'static, 0> {
    fn from(tensor: &'a XAny) -> Self {
        EngineInputs::ProcessedSlice(std::slice::from_ref(tensor))
    }
}
