use std::mem::MaybeUninit;

/// A safe wrapper for uninitialized Vec operations
///
/// This struct provides both safe closure-based interface and flexible pointer access
/// for working with uninitialized memory in tensor operations.
pub struct UninitVec<T> {
    vec: Vec<T>,
    capacity: usize,
    finalized: bool,
}

impl<T> UninitVec<T> {
    /// Create a new uninitialized vector with specified capacity
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            vec: Vec::with_capacity(capacity),
            capacity,
            finalized: false,
        }
    }

    /// Get the capacity of this uninitialized vector
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity.saturating_sub(self.vec.len())
    }

    /// Get a mutable slice to the uninitialized memory
    ///
    /// # Safety
    /// The returned slice points to uninitialized memory.
    /// Caller must ensure all elements are properly initialized before calling `finalize()`
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        debug_assert!(!self.finalized, "Cannot get slice after finalization");
        let spare = self.vec.spare_capacity_mut();
        unsafe { &mut *(spare as *mut [MaybeUninit<T>] as *mut [T]) }
    }

    /// Get a mutable pointer to the uninitialized memory
    ///
    /// # Safety
    /// The returned pointer points to uninitialized memory.
    /// Caller must ensure proper initialization before calling `finalize()`
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        debug_assert!(!self.finalized, "Cannot get pointer after finalization");
        self.vec.spare_capacity_mut().as_mut_ptr() as *mut T
    }

    /// Finalize the vector, marking all elements as initialized
    ///
    /// # Safety
    /// Caller must ensure all elements in the capacity range are properly initialized
    #[inline]
    pub unsafe fn finalize(&mut self) -> Vec<T> {
        debug_assert!(!self.finalized, "Vector already finalized");
        self.vec.set_len(self.capacity);
        self.finalized = true;
        std::mem::take(&mut self.vec)
    }

    /// Process the uninitialized memory with a closure and return the initialized Vec
    ///
    /// This is the safe way to work with uninitialized memory.
    /// The closure receives a mutable slice that must be fully initialized.
    #[inline]
    pub fn init_with<F>(mut self, f: F) -> Vec<T>
    where
        F: FnOnce(&mut [T]),
    {
        debug_assert!(!self.finalized, "Vector already finalized");

        let slice = self.as_mut_slice();
        f(slice);

        unsafe { self.finalize() }
    }

    /// Initialize the vector with a given value and return the initialized Vec
    ///
    /// This is equivalent to `vec![value; capacity]` but more efficient
    /// as it avoids the intermediate allocation.
    #[inline]
    pub fn full(mut self, value: T) -> Vec<T>
    where
        T: Clone,
    {
        debug_assert!(!self.finalized, "Vector already finalized");

        let slice = self.as_mut_slice();
        slice.fill(value);

        unsafe { self.finalize() }
    }
}

impl<T> Drop for UninitVec<T> {
    fn drop(&mut self) {
        if !self.finalized {
            #[cfg(debug_assertions)]
            eprintln!("Warning: UninitVec was dropped without being initialized");
        }
    }
}
