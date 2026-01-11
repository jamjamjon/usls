use anyhow::Result;
use half::{bf16, f16};
use ndarray::{Array, ArrayView, Axis, Dim, IntoDimension, Ix2, IxDyn, IxDynImpl, ScalarOperand};
use num_traits::One;

use crate::Ops;

/// Generic tensor wrapper over [`Array<A, IxDyn>`].
///
/// # Type Parameters
/// * `A` - Element type, defaults to `f32`
///
/// # Example
/// ```ignore
/// let t: X<f32> = X::zeros(&[2, 3, 4]);
/// let t: X<i64> = X::ones(&[1, 10]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct X<A = f32>(pub Array<A, IxDyn>);

/// ArrayView wrapper for zero-copy tensor access.
#[derive(Debug, Clone)]
pub struct XView<'a, A = f32>(pub ndarray::ArrayView<'a, A, IxDyn>);

impl<A, D> From<ndarray::Array<A, D>> for X<A>
where
    D: ndarray::Dimension,
{
    fn from(x: ndarray::Array<A, D>) -> Self {
        Self(x.into_dyn())
    }
}

impl<A: Clone> From<Vec<A>> for X<A> {
    fn from(x: Vec<A>) -> Self {
        Self(Array::from_vec(x).into_dyn().into_owned())
    }
}

impl<'a, A: Clone> From<XView<'a, A>> for X<A> {
    fn from(view: XView<'a, A>) -> Self {
        Self(view.0.to_owned())
    }
}

impl TryFrom<Vec<(u32, u32)>> for X {
    type Error = anyhow::Error;

    fn try_from(values: Vec<(u32, u32)>) -> Result<Self, Self::Error> {
        if values.is_empty() {
            Ok(Self::default())
        } else {
            let mut flattened: Vec<u32> = Vec::new();
            for &(a, b) in values.iter() {
                flattened.push(a);
                flattened.push(b);
            }
            let shape = (values.len(), 2);
            let x = Array::from_shape_vec(shape, flattened)?
                .map(|x| *x as f32)
                .into_dyn();
            Ok(Self(x))
        }
    }
}

impl TryFrom<Vec<Vec<f32>>> for X {
    type Error = anyhow::Error;

    fn try_from(xs: Vec<Vec<f32>>) -> Result<Self, Self::Error> {
        if xs.is_empty() {
            Ok(Self::default())
        } else {
            let shape = (xs.len(), xs[0].len());
            let flattened: Vec<f32> = xs.iter().flatten().cloned().collect();
            let x = Array::from_shape_vec(shape, flattened)?.into_dyn();
            Ok(Self(x))
        }
    }
}

impl<A> std::ops::Deref for X<A> {
    type Target = Array<A, IxDyn>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A> std::ops::DerefMut for X<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, A> From<ndarray::ArrayView<'a, A, IxDyn>> for XView<'a, A> {
    fn from(x: ndarray::ArrayView<'a, A, IxDyn>) -> Self {
        Self(x)
    }
}

impl<'a, A> std::ops::Deref for XView<'a, A> {
    type Target = ndarray::ArrayView<'a, A, IxDyn>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, A: Clone> XView<'a, A> {
    /// Returns the shape of the tensor.
    pub fn dims(&self) -> &[usize] {
        self.0.shape()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.0.ndim()
    }

    /// Converts this view to an owned X tensor.
    pub fn to_owned(&self) -> X<A> {
        X(self.0.to_owned())
    }
}

impl<A: Clone + std::ops::Mul<Output = A> + ScalarOperand> std::ops::Mul<A> for X<A> {
    type Output = Self;

    fn mul(self, other: A) -> Self::Output {
        Self(self.0 * other)
    }
}

impl<A: Clone + std::ops::Div<Output = A> + ScalarOperand> std::ops::Div<A> for X<A> {
    type Output = Self;

    fn div(self, other: A) -> Self::Output {
        Self(self.0 / other)
    }
}

impl<A: Clone + std::ops::Add<Output = A> + ScalarOperand> std::ops::Add<A> for X<A> {
    type Output = Self;

    fn add(self, other: A) -> Self::Output {
        Self(self.0 + other)
    }
}

impl<A: Clone + std::ops::Sub<Output = A> + ScalarOperand> std::ops::Sub<A> for X<A> {
    type Output = Self;

    fn sub(self, other: A) -> Self::Output {
        Self(self.0 - other)
    }
}

impl<A: Clone + std::ops::Div<Output = A>> std::ops::Div<X<A>> for X<A> {
    type Output = Self;

    fn div(self, other: X<A>) -> Self::Output {
        Self(&self.0 / &other.0)
    }
}

impl<A: Clone + std::ops::Mul<Output = A>> std::ops::Mul<X<A>> for X<A> {
    type Output = Self;

    fn mul(self, other: X<A>) -> Self::Output {
        Self(&self.0 * &other.0)
    }
}

// ============================================================================
// Generic methods for X<A> with trait bounds
// ============================================================================

impl<A: Clone + Default> X<A> {
    pub fn zeros_generic<Sh>(shape: Sh) -> Self
    where
        Sh: AsRef<[usize]>,
    {
        Self::from(Array::from_elem(IxDyn(shape.as_ref()), A::default()))
    }

    pub fn zeros_like_generic(x: &Self) -> Self {
        Self::from(Array::default(x.raw_dim()))
    }
}

impl<A: Clone + One> X<A> {
    pub fn ones_generic<Sh>(shape: Sh) -> Self
    where
        Sh: AsRef<[usize]>,
    {
        Self::from(Array::from_elem(IxDyn(shape.as_ref()), A::one()))
    }

    pub fn ones_like_generic(x: &Self) -> Self {
        Self::from(Array::from_elem(x.raw_dim(), A::one()))
    }
}

impl<A: Clone> X<A> {
    pub fn full_generic<Sh>(shape: Sh, x: A) -> Self
    where
        Sh: AsRef<[usize]>,
    {
        Self::from(Array::from_elem(IxDyn(shape.as_ref()), x))
    }

    pub fn from_shape_vec_generic<Sh>(shape: Sh, xs: Vec<A>) -> Result<Self>
    where
        Sh: AsRef<[usize]>,
    {
        Ok(Self::from(Array::from_shape_vec(
            IxDyn(shape.as_ref()),
            xs,
        )?))
    }

    pub fn dims(&self) -> &[usize] {
        self.0.shape()
    }

    pub fn ndim(&self) -> usize {
        self.0.ndim()
    }

    /// Returns a zero-copy view of this tensor.
    pub fn view(&self) -> XView<'_, A> {
        XView(self.0.view())
    }
}

// ============================================================================
// X<f32> specific implementations (backward compatibility)
// ============================================================================

impl X {
    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::from(Array::zeros(Dim(IxDynImpl::from(shape.to_vec()))))
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::from(Array::ones(Dim(IxDynImpl::from(shape.to_vec()))))
    }

    pub fn zeros_like(x: &Self) -> Self {
        Self::from(Array::zeros(x.raw_dim()))
    }

    pub fn ones_like(x: &Self) -> Self {
        Self::from(Array::ones(x.raw_dim()))
    }

    pub fn full(shape: &[usize], x: f32) -> Self {
        Self::from(Array::from_elem(shape, x))
    }

    pub fn from_shape_vec(shape: &[usize], xs: Vec<f32>) -> Result<Self> {
        Ok(Self::from(Array::from_shape_vec(shape, xs)?))
    }

    pub fn sigmoid(mut self) -> Result<Self> {
        self.0 = Ops::sigmoid(self.0);
        Ok(self)
    }

    pub fn broadcast<D: IntoDimension + std::fmt::Debug + Copy>(mut self, dim: D) -> Result<Self> {
        self.0 = Ops::broadcast(self.0, dim)?;
        Ok(self)
    }

    pub fn to_shape<D: ndarray::ShapeArg>(mut self, dim: D) -> Result<Self> {
        self.0 = Ops::to_shape(self.0, dim)?;
        Ok(self)
    }

    pub fn permute(mut self, shape: &[usize]) -> Result<Self> {
        self.0 = Ops::permute(self.0, shape)?;
        Ok(self)
    }

    pub fn nhwc2nchw(mut self) -> Result<Self> {
        self.0 = Ops::nhwc2nchw(self.0)?;
        Ok(self)
    }

    pub fn nchw2nhwc(mut self) -> Result<Self> {
        self.0 = Ops::nchw2nhwc(self.0)?;
        Ok(self)
    }

    pub fn insert_axis(mut self, d: usize) -> Result<Self> {
        self.0 = Ops::insert_axis(self.0, d)?;
        Ok(self)
    }

    pub fn repeat(mut self, d: usize, n: usize) -> Result<Self> {
        self.0 = Ops::repeat(self.0, d, n)?;
        Ok(self)
    }

    pub fn concatenate(mut self, other: &Self, d: usize) -> Result<Self> {
        self.0 = Ops::concatenate(&self.0, other, d)?;
        Ok(self)
    }

    pub fn concat(xs: &[Self], d: usize) -> Result<Self> {
        let xs = xs.iter().cloned().map(|x| x.0).collect::<Vec<_>>();
        let x = Ops::concat(&xs, d)?;
        Ok(x.into())
    }

    pub fn stack(xs: &[Self], d: usize) -> Result<Self> {
        let views: Vec<_> = xs.iter().map(|x| x.0.view()).collect();
        let stacked = ndarray::stack(ndarray::Axis(d), &views)?;
        Ok(X(stacked))
    }

    pub fn normalize(mut self, min_: f32, max_: f32) -> Result<Self> {
        Ops::normalize(&mut self.0, min_, max_)?;

        Ok(self)
    }

    pub fn standardize(mut self, mean: &[f32], std: &[f32], dim: usize) -> Result<Self> {
        Ops::standardize(&mut self.0, mean.into(), std.into(), dim)?;
        Ok(self)
    }

    pub fn norm(mut self, d: usize) -> Result<Self> {
        self.0 = Ops::norm(self.0, d)?;
        Ok(self)
    }

    pub fn dot2(&self, other: &Self) -> Result<Self> {
        // Check dimensions
        if self.ndim() != 2 || other.ndim() != 2 {
            anyhow::bail!(
                "dot2 requires 2D matrices, got {}D and {}D",
                self.ndim(),
                other.ndim()
            );
        }

        let a = self.0.as_standard_layout().into_dimensionality::<Ix2>()?;
        let b = other.0.as_standard_layout().into_dimensionality::<Ix2>()?;

        // Check compatibility
        if a.shape()[1] != b.shape()[1] {
            anyhow::bail!(
                "Incompatible dimensions for dot2: {:?} and {:?}",
                a.shape(),
                b.shape()
            );
        }

        Ok(a.dot(&b.t()).into_dyn().into())
    }

    pub fn softmax(mut self, d: usize) -> Result<Self> {
        self.0 = Ops::softmax(self.0, d)?;
        Ok(self)
    }

    pub fn unsigned(mut self) -> Self {
        self.0.par_mapv_inplace(|x| if x < 0.0 { 0.0 } else { x });
        self
    }

    /// Matrix multiplication for 2D tensors: self (1, nm) @ other (nm, mh*mw) -> (1, mh*mw)
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.ndim() != 2 || other.ndim() != 2 {
            anyhow::bail!(
                "matmul requires 2D matrices, got {}D and {}D",
                self.ndim(),
                other.ndim()
            );
        }

        let a = self.0.view().into_dimensionality::<Ix2>()?;
        let b = other.0.view().into_dimensionality::<Ix2>()?;

        if a.shape()[1] != b.shape()[0] {
            anyhow::bail!(
                "Incompatible dimensions for matmul: {:?} and {:?}",
                a.shape(),
                b.shape()
            );
        }

        Ok(a.dot(&b).into_dyn().into())
    }

    /// Iterate over dimension 0, returning views
    pub fn iter_dim(&self, axis: usize) -> Vec<ArrayView<'_, f32, IxDyn>> {
        self.0.axis_iter(Axis(axis)).collect()
    }

    /// Convert to f16
    pub fn to_f16(&self) -> X<f16> {
        X(self.0.mapv(f16::from_f32))
    }

    /// Convert to bf16
    pub fn to_bf16(&self) -> X<bf16> {
        X(self.0.mapv(bf16::from_f32))
    }

    /// Convert to f64
    pub fn to_f64(&self) -> X<f64> {
        X(self.0.mapv(|x| x as f64))
    }

    /// Convert to i8
    pub fn to_i8(&self) -> X<i8> {
        X(self.0.mapv(|x| x as i8))
    }

    /// Convert to i16
    pub fn to_i16(&self) -> X<i16> {
        X(self.0.mapv(|x| x as i16))
    }

    /// Convert to i32
    pub fn to_i32(&self) -> X<i32> {
        X(self.0.mapv(|x| x as i32))
    }

    /// Convert to i64
    pub fn to_i64(&self) -> X<i64> {
        X(self.0.mapv(|x| x as i64))
    }

    /// Convert to u8
    pub fn to_u8(&self) -> X<u8> {
        X(self.0.mapv(|x| x as u8))
    }

    /// Convert to u16
    pub fn to_u16(&self) -> X<u16> {
        X(self.0.mapv(|x| x as u16))
    }

    /// Convert to u32
    pub fn to_u32(&self) -> X<u32> {
        X(self.0.mapv(|x| x as u32))
    }

    /// Convert to u64
    pub fn to_u64(&self) -> X<u64> {
        X(self.0.mapv(|x| x as u64))
    }

    /// Convert to bool (non-zero = true)
    pub fn to_bool(&self) -> X<bool> {
        X(self.0.mapv(|x| x != 0.0))
    }

    /// Concatenate tensors along axis
    pub fn cat(xs: &[Self], axis: usize) -> Result<Self> {
        let views: Vec<_> = xs.iter().map(|x| x.0.view()).collect();
        let result = ndarray::concatenate(Axis(axis), &views)?;
        Ok(Self(result))
    }

    /// Unsqueeze: insert a new axis at position
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        Ok(Self(self.0.clone().insert_axis(Axis(axis))))
    }

    /// Broadcast to shape
    pub fn broadcast_to(&self, shape: (usize, usize, usize)) -> Result<Self> {
        let broadcasted = self
            .0
            .broadcast(IxDyn(&[shape.0, shape.1, shape.2]))
            .ok_or_else(|| anyhow::anyhow!("Cannot broadcast to shape"))?;
        Ok(Self(broadcasted.to_owned()))
    }

    /// L2 normalization along axis, keeping dimension
    pub fn norm_l2_keepdim(&self, axis: isize) -> Result<Self> {
        let ndim = self.ndim();
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };

        // Compute L2 norm: sqrt(sum(x^2))
        let squared = self.0.mapv(|x| x * x);
        let sum = squared.sum_axis(Axis(axis));
        let norm = sum.mapv(|x| x.sqrt());

        // Keep dimension by inserting axis back
        Ok(Self(norm.insert_axis(Axis(axis))))
    }

    /// Transpose (reverse all axes)
    pub fn t(&self) -> Result<Self> {
        Ok(Self(self.0.clone().reversed_axes()))
    }

    /// Softmax along axis (supports negative indexing)
    pub fn softmax_axis(&self, axis: isize) -> Result<Self> {
        let ndim = self.ndim();
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };
        let x = Ops::softmax(self.0.clone(), axis)?;
        Ok(Self(x))
    }
}
