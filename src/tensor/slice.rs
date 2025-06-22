//! Tensor slicing utilities and types
//!
//! This module provides the core slicing functionality for tensors, including
//! the SliceOrIndex enum and traits for converting Rust slice syntax into
//! tensor slice specifications. Users should primarily use the `tensor.slice()` method
//! for all slicing operations.
//!
//! ## New Features
//! - Support for negative indexing (-1, -2, etc.)
//! - Full compatibility with ndarray's s! macro behavior
//! - Enhanced error handling for out-of-bounds access

use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// Slice specification for tensor slicing operations
///
/// This enum provides a flexible way to specify slices similar to ndarray's s! macro.
/// It supports ranges, single indices, full slices (..), and negative indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceOrIndex {
    /// A range slice like 0..5 or -3..-1
    Range(Range<isize>),
    /// A range from start to end like 2.. or -2..
    RangeFrom(isize),
    /// A range to end like ..5 or ..-1
    RangeTo(isize),
    /// A full slice like .. (equivalent to 0..dim_size)
    FullSlice,
    /// A single index like 3 or -1
    Index(isize),
}

/// Convert negative index to positive index based on dimension size
///
/// # Arguments
/// * `idx` - The index (can be negative)
/// * `dim_size` - The size of the dimension
///
/// # Returns
/// The resolved positive index
///
/// # Examples
/// ```
/// // For a dimension of size 5:
/// // -1 -> 4, -2 -> 3, 0 -> 0, 4 -> 4
/// ```
pub fn resolve_negative_index(idx: isize, dim_size: usize) -> anyhow::Result<usize> {
    if idx < 0 {
        let resolved = dim_size as isize + idx;
        if resolved < 0 {
            anyhow::bail!(
                "Negative index {} out of bounds for dimension size {}",
                idx,
                dim_size
            );
        }
        Ok(resolved as usize)
    } else {
        if idx as usize >= dim_size {
            anyhow::bail!(
                "Index {} out of bounds for dimension size {}",
                idx,
                dim_size
            );
        }
        Ok(idx as usize)
    }
}

/// Convert negative index to positive index for range end (allows dim_size)
///
/// This function is specifically for range end indices, which can be equal to
/// the dimension size since ranges are exclusive on the end.
///
/// # Arguments
/// * `idx` - The index (can be negative)
/// * `dim_size` - The size of the dimension
///
/// # Returns
/// The resolved positive index
pub fn resolve_negative_index_for_range_end(idx: isize, dim_size: usize) -> anyhow::Result<usize> {
    if idx < 0 {
        let resolved = dim_size as isize + idx;
        if resolved < 0 {
            anyhow::bail!(
                "Negative index {} out of bounds for dimension size {}",
                idx,
                dim_size
            );
        }
        Ok(resolved as usize)
    } else {
        if idx as usize > dim_size {
            anyhow::bail!(
                "Index {} out of bounds for dimension size {}",
                idx,
                dim_size
            );
        }
        Ok(idx as usize)
    }
}

/// Resolve a range with potential negative indices
pub fn resolve_range(start: isize, end: isize, dim_size: usize) -> anyhow::Result<Range<usize>> {
    let resolved_start = resolve_negative_index(start, dim_size)?;
    let resolved_end = resolve_negative_index_for_range_end(end, dim_size)?;

    if resolved_start > resolved_end {
        anyhow::bail!(
            "Invalid range: start {} > end {} (resolved from {}..{} for dimension size {})",
            resolved_start,
            resolved_end,
            start,
            end,
            dim_size
        );
    }

    Ok(resolved_start..resolved_end)
}

// Legacy support for usize types (for backward compatibility)
impl From<Range<usize>> for SliceOrIndex {
    fn from(range: Range<usize>) -> Self {
        SliceOrIndex::Range(range.start as isize..range.end as isize)
    }
}

impl From<usize> for SliceOrIndex {
    fn from(index: usize) -> Self {
        SliceOrIndex::Index(index as isize)
    }
}

// New support for isize types (negative indexing)
impl From<Range<isize>> for SliceOrIndex {
    fn from(range: Range<isize>) -> Self {
        SliceOrIndex::Range(range)
    }
}

impl From<isize> for SliceOrIndex {
    fn from(index: isize) -> Self {
        SliceOrIndex::Index(index)
    }
}

impl From<RangeFull> for SliceOrIndex {
    fn from(_: RangeFull) -> Self {
        SliceOrIndex::FullSlice
    }
}

impl From<RangeFrom<usize>> for SliceOrIndex {
    fn from(range: RangeFrom<usize>) -> Self {
        SliceOrIndex::RangeFrom(range.start as isize)
    }
}

impl From<RangeFrom<isize>> for SliceOrIndex {
    fn from(range: RangeFrom<isize>) -> Self {
        SliceOrIndex::RangeFrom(range.start)
    }
}

impl From<RangeTo<usize>> for SliceOrIndex {
    fn from(range: RangeTo<usize>) -> Self {
        SliceOrIndex::RangeTo(range.end as isize)
    }
}

impl From<RangeTo<isize>> for SliceOrIndex {
    fn from(range: RangeTo<isize>) -> Self {
        SliceOrIndex::RangeTo(range.end)
    }
}

/// Convenience constants and functions for slice creation
impl SliceOrIndex {
    /// Create a full slice (equivalent to ..)
    pub fn full() -> Self {
        SliceOrIndex::FullSlice
    }

    /// Create an index slice
    pub fn index(idx: isize) -> Self {
        SliceOrIndex::Index(idx)
    }

    /// Create a range slice
    pub fn range(start: isize, end: isize) -> Self {
        SliceOrIndex::Range(Range { start, end })
    }

    /// Create a slice from start to end (equivalent to ..end)
    pub fn to(end: isize) -> Self {
        SliceOrIndex::RangeTo(end)
    }

    /// Create a slice from start to dimension end (equivalent to start..)
    pub fn from(start: isize) -> Self {
        SliceOrIndex::RangeFrom(start)
    }
}

/// Trait for types that can be converted to slice specifications
/// This trait allows direct use of Rust's native slice syntax like .., ..4, 1..5, 3
pub trait IntoSliceSpec {
    fn into_slice_spec(self) -> Vec<SliceOrIndex>;
}

/// Trait for individual slice elements that can be converted to SliceOrIndex
pub trait IntoSliceElement {
    fn into_slice_element(self) -> SliceOrIndex;
}

// Implement for basic Rust slice types
impl IntoSliceElement for RangeFull {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::FullSlice
    }
}

impl IntoSliceElement for RangeTo<usize> {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::RangeTo(self.end as isize)
    }
}

impl IntoSliceElement for RangeTo<isize> {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::RangeTo(self.end)
    }
}

impl IntoSliceElement for RangeTo<i32> {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::RangeTo(self.end as isize)
    }
}

impl IntoSliceElement for RangeFrom<usize> {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::RangeFrom(self.start as isize)
    }
}

impl IntoSliceElement for RangeFrom<isize> {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::RangeFrom(self.start)
    }
}

impl IntoSliceElement for RangeFrom<i32> {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::RangeFrom(self.start as isize)
    }
}

impl IntoSliceElement for Range<usize> {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::Range(self.start as isize..self.end as isize)
    }
}

impl IntoSliceElement for Range<isize> {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::Range(self)
    }
}

impl IntoSliceElement for Range<i32> {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::Range(self.start as isize..self.end as isize)
    }
}

impl IntoSliceElement for usize {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::Index(self as isize)
    }
}

impl IntoSliceElement for isize {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::Index(self)
    }
}

impl IntoSliceElement for i32 {
    fn into_slice_element(self) -> SliceOrIndex {
        SliceOrIndex::Index(self as isize)
    }
}

// Allow SliceOrIndex to be used directly in tuples
impl IntoSliceElement for SliceOrIndex {
    fn into_slice_element(self) -> SliceOrIndex {
        self
    }
}

// Implement IntoSliceSpec for tuples of slice elements
impl<T1> IntoSliceSpec for (T1,)
where
    T1: IntoSliceElement,
{
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        vec![self.0.into_slice_element()]
    }
}

impl<T1, T2> IntoSliceSpec for (T1, T2)
where
    T1: IntoSliceElement,
    T2: IntoSliceElement,
{
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        vec![self.0.into_slice_element(), self.1.into_slice_element()]
    }
}

impl<T1, T2, T3> IntoSliceSpec for (T1, T2, T3)
where
    T1: IntoSliceElement,
    T2: IntoSliceElement,
    T3: IntoSliceElement,
{
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        vec![
            self.0.into_slice_element(),
            self.1.into_slice_element(),
            self.2.into_slice_element(),
        ]
    }
}

impl<T1, T2, T3, T4> IntoSliceSpec for (T1, T2, T3, T4)
where
    T1: IntoSliceElement,
    T2: IntoSliceElement,
    T3: IntoSliceElement,
    T4: IntoSliceElement,
{
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        vec![
            self.0.into_slice_element(),
            self.1.into_slice_element(),
            self.2.into_slice_element(),
            self.3.into_slice_element(),
        ]
    }
}

impl<T1, T2, T3, T4, T5> IntoSliceSpec for (T1, T2, T3, T4, T5)
where
    T1: IntoSliceElement,
    T2: IntoSliceElement,
    T3: IntoSliceElement,
    T4: IntoSliceElement,
    T5: IntoSliceElement,
{
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        vec![
            self.0.into_slice_element(),
            self.1.into_slice_element(),
            self.2.into_slice_element(),
            self.3.into_slice_element(),
            self.4.into_slice_element(),
        ]
    }
}

impl<T1, T2, T3, T4, T5, T6> IntoSliceSpec for (T1, T2, T3, T4, T5, T6)
where
    T1: IntoSliceElement,
    T2: IntoSliceElement,
    T3: IntoSliceElement,
    T4: IntoSliceElement,
    T5: IntoSliceElement,
    T6: IntoSliceElement,
{
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        vec![
            self.0.into_slice_element(),
            self.1.into_slice_element(),
            self.2.into_slice_element(),
            self.3.into_slice_element(),
            self.4.into_slice_element(),
            self.5.into_slice_element(),
        ]
    }
}

// Implementation for slice references to support legacy slice_dyn calls
impl IntoSliceSpec for &[SliceOrIndex] {
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        self.to_vec()
    }
}

// Implementation for Vec<SliceOrIndex> to support direct usage
impl IntoSliceSpec for Vec<SliceOrIndex> {
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        self
    }
}

// Implementation for Vec<Range<usize>> to support legacy slice calls
impl IntoSliceSpec for Vec<Range<usize>> {
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        self.into_iter()
            .map(|r| SliceOrIndex::Range(r.start as isize..r.end as isize))
            .collect()
    }
}

// Implementation for &[Range<usize>] to support legacy slice calls
impl IntoSliceSpec for &[Range<usize>] {
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        self.iter()
            .map(|r| SliceOrIndex::Range(r.start as isize..r.end as isize))
            .collect()
    }
}

// Implementation for array references of different sizes
impl<const N: usize> IntoSliceSpec for &[Range<usize>; N] {
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        self.iter()
            .map(|r| SliceOrIndex::Range(r.start as isize..r.end as isize))
            .collect()
    }
}

impl<const N: usize> IntoSliceSpec for [Range<usize>; N] {
    fn into_slice_spec(self) -> Vec<SliceOrIndex> {
        self.into_iter()
            .map(|r| SliceOrIndex::Range(r.start as isize..r.end as isize))
            .collect()
    }
}

// Note: Only tuple syntax is supported for ergonomic slicing to keep the API simple.
