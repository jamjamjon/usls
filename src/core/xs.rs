use anyhow::Result;
use std::collections::HashMap;
use std::ops::{Deref, Index};

use crate::X;

#[derive(Debug, Default, Clone)]
pub struct Xs {
    names: Vec<String>,
    map: HashMap<String, X>,
}

impl Xs {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn add(&mut self, key: &str, value: X) -> Result<()> {
        if !self.map.contains_key(key) {
            self.names.push(key.to_string());
        }
        self.map.insert(key.to_string(), value);
        Ok(())
    }

    pub fn names(&self) -> &Vec<String> {
        &self.names
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
        self.map.get(index).expect("Index was not found in `Xs`")
    }
}

impl Index<usize> for Xs {
    type Output = X;

    fn index(&self, index: usize) -> &Self::Output {
        self.names
            .get(index)
            .and_then(|key| self.map.get(key))
            .expect("Index was not found in `Xs`")
    }
}

impl<'a> IntoIterator for &'a Xs {
    type Item = &'a X;
    type IntoIter = std::collections::hash_map::Values<'a, String, X>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.values()
    }
}
