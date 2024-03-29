use ndarray::{Array, Axis, IxDyn};

#[derive(Clone, PartialEq, Default)]
pub struct Embedding {
    data: Array<f32, IxDyn>,
    names: Option<Vec<String>>,
}

impl std::fmt::Debug for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("").field("Top5", &self.topk(5)).finish()
    }
}

impl Embedding {
    pub fn new(data: Array<f32, IxDyn>, names: Option<Vec<String>>) -> Self {
        Self { data, names }
    }

    pub fn data(&self) -> &Array<f32, IxDyn> {
        &self.data
    }

    pub fn topk(&self, k: usize) -> Vec<(usize, f32, Option<String>)> {
        let mut probs = self
            .data
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

    pub fn norm(&self) -> Array<f32, IxDyn> {
        let std_ = self.data.mapv(|x| x * x).sum_axis(Axis(0)).mapv(f32::sqrt);
        self.data.clone() / std_
    }

    pub fn top1(&self) -> (usize, f32, Option<String>) {
        self.topk(1)[0].to_owned()
    }
}
