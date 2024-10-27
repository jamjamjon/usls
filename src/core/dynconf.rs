use std::ops::Index;

/// Dynamic Confidences
#[derive(Clone, PartialEq, PartialOrd)]
pub struct DynConf(Vec<f32>);

impl Default for DynConf {
    fn default() -> Self {
        Self(vec![0.4f32])
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
    pub fn new(confs: &[f32], n: usize) -> Self {
        if confs.is_empty() && n != 0 {
            panic!("Error: No value found in confs")
        }
        let confs = if confs.len() >= n {
            confs[..n].to_vec()
        } else {
            let val = confs.last().unwrap();
            let mut confs = confs.to_vec();
            for _ in 0..(n - confs.len()) {
                confs.push(*val);
            }
            confs
        };

        Self(confs)
    }
}
