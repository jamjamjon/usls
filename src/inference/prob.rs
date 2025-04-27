use aksr::Builder;

use crate::{InstanceMeta, Style};

#[derive(Builder, Clone, PartialEq, Default, Debug)]
pub struct Prob {
    meta: InstanceMeta,
    style: Option<Style>,
}

// #[derive(Builder, Clone, PartialEq, Default, Debug)]
// pub struct Probs(#[args(aka = "probs")] Vec<Prob>);

impl Prob {
    pub fn new_probs(probs: &[f32], names: Option<&[&str]>, k: usize) -> Vec<Self> {
        let mut pairs: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(id, &prob)| (id, prob))
            .collect();

        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs.truncate(k);

        pairs
            .into_iter()
            .map(|(id, confidence)| {
                let mut meta = InstanceMeta::default()
                    .with_id(id)
                    .with_confidence(confidence);

                if let Some(names) = names {
                    if id < names.len() {
                        meta = meta.with_name(names[id]);
                    }
                }

                Prob::default().with_meta(meta)
            })
            .collect()
    }

    pub fn with_uid(mut self, uid: usize) -> Self {
        self.meta = self.meta.with_uid(uid);
        self
    }

    pub fn with_id(mut self, id: usize) -> Self {
        self.meta = self.meta.with_id(id);
        self
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.meta = self.meta.with_name(name);
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.meta = self.meta.with_confidence(confidence);
        self
    }

    pub fn uid(&self) -> usize {
        self.meta.uid()
    }

    pub fn name(&self) -> Option<&str> {
        self.meta.name()
    }

    pub fn confidence(&self) -> Option<f32> {
        self.meta.confidence()
    }

    pub fn id(&self) -> Option<usize> {
        self.meta.id()
    }
}
