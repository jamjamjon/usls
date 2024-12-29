use aksr::Builder;

/// Probabilities for classification.
#[derive(Builder, Clone, PartialEq, Default)]
pub struct Prob {
    probs: Vec<f32>,
    names: Option<Vec<String>>,
}

impl std::fmt::Debug for Prob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("").field("Top5", &self.topk(5)).finish()
    }
}

impl Prob {
    pub fn topk(&self, k: usize) -> Vec<(usize, f32, Option<String>)> {
        let mut probs = self
            .probs
            .iter()
            .enumerate()
            .map(|(a, b)| (a, *b))
            .collect::<Vec<_>>();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut topk = Vec::new();
        for &(id, confidence) in probs.iter().take(k) {
            topk.push((
                id,
                confidence,
                self.names.as_ref().map(|names| names[id].to_owned()),
            ));
        }
        topk
    }

    pub fn top1(&self) -> (usize, f32, Option<String>) {
        self.topk(1)[0].to_owned()
    }
}
