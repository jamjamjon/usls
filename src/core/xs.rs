use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use std::collections::HashMap;
use std::ops::{Deref, Index};

use crate::{generate_random_string, X};

/// Collection of named tensors with associated images and texts.
#[derive(Builder, Debug, Default, Clone)]
pub struct Xs {
    map: HashMap<String, X>,
    names: Vec<String>,
    // TODO: move to Processor
    pub images: Vec<Vec<DynamicImage>>,
    pub texts: Vec<Vec<DynamicImage>>,
}

impl From<X> for Xs {
    fn from(x: X) -> Self {
        let mut xs = Self::default();
        xs.push(x);
        xs
    }
}

impl From<Vec<X>> for Xs {
    fn from(xs: Vec<X>) -> Self {
        let mut ys = Self::default();
        for x in xs {
            ys.push(x);
        }
        ys
    }
}

impl Xs {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn derive(&self) -> Self {
        Self {
            map: Default::default(),
            names: Default::default(),
            ..self.clone()
        }
    }

    pub fn push(&mut self, value: X) {
        loop {
            let key = generate_random_string(5);
            if !self.map.contains_key(&key) {
                self.names.push(key.to_string());
                self.map.insert(key.to_string(), value);
                break;
            }
        }
    }

    pub fn push_kv(&mut self, key: &str, value: X) -> Result<()> {
        if !self.map.contains_key(key) {
            self.names.push(key.to_string());
            self.map.insert(key.to_string(), value);
            Ok(())
        } else {
            anyhow::bail!("Xs already contains key: {:?}", key)
        }
    }
}

impl Deref for Xs {
    type Target = HashMap<String, X>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl Index<&str> for Xs {
    type Output = X;

    fn index(&self, index: &str) -> &Self::Output {
        self.map.get(index).unwrap_or_else(|| {
            let available_keys: Vec<&str> = self.map.keys().map(|s| s.as_str()).collect();
            panic!(
                "Key '{}' was not found in Xs. Available keys: {:?}",
                index, available_keys
            )
        })
    }
}

impl Index<usize> for Xs {
    type Output = X;

    fn index(&self, index: usize) -> &Self::Output {
        self.names
            .get(index)
            .and_then(|key| self.map.get(key))
            .unwrap_or_else(|| {
                panic!(
                    "Index {} was not found in Xs. Available indices: 0..{}",
                    index,
                    self.names.len()
                )
            })
    }
}

pub struct XsIter<'a> {
    inner: std::vec::IntoIter<&'a X>,
}

impl<'a> Iterator for XsIter<'a> {
    type Item = &'a X;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a> IntoIterator for &'a Xs {
    type Item = &'a X;
    type IntoIter = XsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let values: Vec<&X> = self.names.iter().map(|x| &self.map[x]).collect();
        XsIter {
            inner: values.into_iter(),
        }
    }
}
