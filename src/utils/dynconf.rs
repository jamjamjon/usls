use std::ops::Index;

/// Dynamic Confidences
#[derive(Clone, PartialEq, PartialOrd)]
pub struct DynConf(Vec<f32>);

impl Default for DynConf {
    fn default() -> Self {
        Self(vec![0.3f32])
    }
}

impl From<f32> for DynConf {
    fn from(conf: f32) -> Self {
        Self(vec![conf])
    }
}

impl std::fmt::Debug for DynConf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("").field("DynConf", &self.0).finish()
    }
}

impl std::fmt::Display for DynConf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl Index<usize> for DynConf {
    type Output = f32;

    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl DynConf {
    /// Create a new DynConf with proper error handling
    pub fn new(confs: &[f32], n: usize) -> anyhow::Result<Self> {
        if confs.is_empty() && n != 0 {
            return Err(anyhow::anyhow!(
                "Cannot create DynConf: no values provided but {} elements required",
                n
            ));
        }

        if n == 0 {
            return Ok(Self(Vec::new()));
        }

        let confs = if confs.len() >= n {
            confs[..n].to_vec()
        } else {
            let val = confs.last().ok_or_else(|| {
                anyhow::anyhow!(
                    "Cannot create DynConf: empty confs slice but {} elements required",
                    n
                )
            })?;
            let mut result = Vec::with_capacity(n);
            result.extend_from_slice(confs);
            result.resize(n, *val);
            result
        };

        Ok(Self(confs))
    }

    /// Create a new DynConf with default fallback (for backward compatibility)
    /// This method will use default confidence of 0.3 if creation fails
    pub fn new_or_default(confs: &[f32], n: usize) -> Self {
        Self::new(confs, n).unwrap_or_else(|_| Self::from_default(n))
    }

    /// Create a new DynConf with fallback to specified value if creation fails
    pub fn new_or(confs: &[f32], n: usize, x: f32) -> Self {
        Self::new(confs, n).unwrap_or_else(|_| Self::from_value(x, n))
    }

    /// Get the length of the confidence vector
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the confidence vector is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the underlying confidence values
    pub fn values(&self) -> &[f32] {
        &self.0
    }

    /// Create DynConf with default confidence values (0.3)
    fn from_default(n: usize) -> Self {
        Self(vec![0.3; n])
    }

    /// Create DynConf with specified confidence value
    fn from_value(value: f32, n: usize) -> Self {
        Self(vec![value; n])
    }
}
