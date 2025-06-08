use aksr::Builder;

use crate::{InstanceMeta, Style};

/// Probability result with classification metadata.
#[derive(Builder, Clone, PartialEq, Default, Debug)]
pub struct Prob {
    meta: InstanceMeta,
    style: Option<Style>,
}

// #[derive(Builder, Clone, PartialEq, Default, Debug)]
// pub struct Probs(#[args(aka = "probs")] Vec<Prob>);

impl Prob {
    impl_meta_methods!();
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
                    if !names.is_empty() {
                        meta = meta.with_name(names[id]);
                    }
                }

                Prob::default().with_meta(meta)
            })
            .collect()
    }
}
