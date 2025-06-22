use anyhow::Result;
use rayon::prelude::*;
use std::ops::{Deref, Index};

use crate::Tensor;

/// High-performance batch tensor container with optional naming and parallel processing capabilities
#[derive(Debug, Default, Clone)]
pub struct Xs {
    tensors: Vec<Tensor>,
    names: Option<Vec<String>>, // Optional naming to avoid HashMap overhead
}

impl From<Tensor> for Xs {
    fn from(tensor: Tensor) -> Self {
        Self {
            tensors: vec![tensor],
            names: None,
        }
    }
}

impl From<Vec<Tensor>> for Xs {
    fn from(tensors: Vec<Tensor>) -> Self {
        Self {
            tensors,
            names: None,
        }
    }
}

impl Xs {
    /// Create a new empty Xs container
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new Xs container with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tensors: Vec::with_capacity(capacity),
            names: None,
        }
    }

    /// Create Xs from a vector of tensors
    pub fn from_tensors(tensors: Vec<Tensor>) -> Self {
        Self {
            tensors,
            names: None,
        }
    }

    /// Create Xs with named tensors
    pub fn from_named_tensors(tensors: Vec<Tensor>, names: Vec<String>) -> Result<Self> {
        if tensors.len() != names.len() {
            anyhow::bail!("Tensors and names must have the same length");
        }
        Ok(Self {
            tensors,
            names: Some(names),
        })
    }

    /// Add a tensor to the container
    pub fn push(&mut self, tensor: Tensor) {
        self.tensors.push(tensor);
    }

    /// Add a named tensor to the container
    pub fn push_named(&mut self, tensor: Tensor, name: String) -> Result<()> {
        if let Some(ref mut names) = self.names {
            if names.contains(&name) {
                anyhow::bail!("Name '{}' already exists", name);
            }
            names.push(name);
        } else {
            // Initialize names vector if not present
            let mut names: Vec<String> = (0..self.tensors.len())
                .map(|i| format!("tensor_{}", i))
                .collect();
            names.push(name);
            self.names = Some(names);
        }
        self.tensors.push(tensor);
        Ok(())
    }

    /// Add a named tensor (backward compatibility)
    pub fn push_kv(&mut self, name: &str, tensor: Tensor) -> Result<()> {
        self.push_named(tensor, name.to_string())
    }

    /// Get the number of tensors
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if the container is empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get tensor by name (if names are available)
    pub fn get_by_name(&self, name: &str) -> Option<&Tensor> {
        if let Some(ref names) = self.names {
            names
                .iter()
                .position(|n| n == name)
                .and_then(|idx| self.tensors.get(idx))
        } else {
            None
        }
    }

    /// Get mutable tensor by name (if names are available)
    pub fn get_mut_by_name(&mut self, name: &str) -> Option<&mut Tensor> {
        if let Some(ref names) = self.names {
            let idx = names.iter().position(|n| n == name)?;
            self.tensors.get_mut(idx)
        } else {
            None
        }
    }

    /// Get all tensor names (if available)
    pub fn names(&self) -> Option<&[String]> {
        self.names.as_deref()
    }

    /// Parallel iterator over tensors
    pub fn par_iter(&self) -> impl ParallelIterator<Item = &Tensor> {
        self.tensors.par_iter()
    }

    /// Parallel enumerated iterator
    pub fn par_enumerate(&self) -> impl ParallelIterator<Item = (usize, &Tensor)> {
        self.tensors.par_iter().enumerate()
    }

    /// Apply a function to all tensors in parallel
    pub fn map<F>(&self, f: F) -> Result<Xs>
    where
        F: Fn(&Tensor) -> Result<Tensor> + Sync + Send,
    {
        let results: Result<Vec<_>> = self.tensors.par_iter().map(f).collect();
        Ok(Self::from_tensors(results?))
    }

    /// Filter and map tensors in parallel
    pub fn filter_map<F>(&self, f: F) -> Result<Xs>
    where
        F: Fn(&Tensor) -> Option<Result<Tensor>> + Sync + Send,
    {
        let results: Result<Vec<_>> = self.tensors.par_iter().filter_map(f).collect();
        Ok(Self::from_tensors(results?))
    }

    /// Apply ReLU to all tensors
    pub fn relu(&self) -> Result<Xs> {
        self.map(|t| t.relu())
    }

    /// Apply softmax to all tensors along specified dimension
    pub fn softmax(&self, dim: usize) -> Result<Xs> {
        self.map(|t| t.softmax(dim))
    }

    /// Stack all tensors along a new dimension (TODO: implement in Tensor)
    pub fn stack(&self, dim: usize) -> Result<crate::Tensor> {
        if self.tensors.is_empty() {
            anyhow::bail!("Cannot stack empty tensor collection");
        }
        crate::Tensor::stack(&self.tensors, dim)
    }

    /// Concatenate all tensors along an existing dimension
    pub fn concat(&self, dim: usize) -> Result<crate::Tensor> {
        if self.tensors.is_empty() {
            anyhow::bail!("Cannot concatenate empty tensor collection");
        }
        crate::Tensor::concat(&self.tensors, dim)
    }
}

impl Deref for Xs {
    type Target = Vec<Tensor>;

    fn deref(&self) -> &Self::Target {
        &self.tensors
    }
}

impl Index<&str> for Xs {
    type Output = Tensor;

    fn index(&self, name: &str) -> &Self::Output {
        self.get_by_name(name).unwrap_or_else(|| {
            let available_names = if let Some(ref names) = self.names {
                names.iter().map(|s| s.as_str()).collect::<Vec<_>>()
            } else {
                vec!["<no names available>"]
            };
            panic!(
                "Name '{}' was not found in Xs. Available names: {:?}",
                name, available_names
            )
        })
    }
}

impl Index<usize> for Xs {
    type Output = Tensor;

    fn index(&self, index: usize) -> &Self::Output {
        self.tensors.get(index).unwrap_or_else(|| {
            panic!(
                "Index {} was not found in Xs. Available indices: 0..{}",
                index,
                self.tensors.len()
            )
        })
    }
}

// Iterator implementations for Xs
impl<'a> IntoIterator for &'a Xs {
    type Item = &'a Tensor;
    type IntoIter = std::slice::Iter<'a, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

impl IntoIterator for Xs {
    type Item = Tensor;
    type IntoIter = std::vec::IntoIter<Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        self.tensors.into_iter()
    }
}

// // Additional iterator methods
// impl Xs {
//     /// Get an iterator over tensors
//     pub fn iter(&self) -> std::slice::Iter<Tensor> {
//         self.tensors.iter()
//     }

//     /// Get a mutable iterator over tensors
//     pub fn iter_mut(&mut self) -> std::slice::IterMut<Tensor> {
//         self.tensors.iter_mut()
//     }

//     /// Get an iterator over (name, tensor) pairs if names are available
//     pub fn named_iter(&self) -> Option<impl Iterator<Item = (&str, &Tensor)>> {
//         self.names.as_ref().map(|names| {
//             names
//                 .iter()
//                 .zip(self.tensors.iter())
//                 .map(|(name, tensor)| (name.as_str(), tensor))
//         })
//     }
// }
