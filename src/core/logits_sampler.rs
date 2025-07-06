use anyhow::Result;
use rand::distr::{weighted::WeightedIndex, Distribution};

/// Logits sampler for text generation with temperature and nucleus sampling.
#[derive(Debug, Clone)]
pub struct LogitsSampler {
    /// Temperature parameter for controlling randomness in sampling.
    temperature: f32,
    /// Top-p parameter for nucleus sampling.
    p: f32,
}

impl Default for LogitsSampler {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            p: 0.0,
        }
    }
}

impl LogitsSampler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_topp(mut self, p: f32) -> Self {
        self.p = p.clamp(0.0, 1.0);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(1e-7);
        self
    }

    pub fn decode(&self, logits: &[f32]) -> Result<u32> {
        if self.p == 0.0 {
            self.search_by_argmax(logits)
        } else {
            self.sample_by_topp(logits)
        }
    }

    fn search_by_argmax(&self, logits: &[f32]) -> Result<u32> {
        // no need to do softmax
        let (token_id, _) = logits
            .iter()
            .enumerate()
            .reduce(|max, x| if x.1 > max.1 { x } else { max })
            .ok_or_else(|| anyhow::anyhow!("Empty logits array provided to argmax search"))?;
        Ok(token_id as u32)
    }

    fn sample_by_topp(&self, logits: &[f32]) -> Result<u32> {
        let logits = self.softmax(logits);
        let mut logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();
        logits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or_else(|| {
                log::warn!("NaN or invalid value encountered in logits sorting");
                std::cmp::Ordering::Equal
            })
        });

        // candidates
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        let mut acc_prob: f32 = 0.0;
        for (idx, prob) in logits.iter() {
            acc_prob += prob;
            if acc_prob >= self.p {
                if candidates.is_empty() {
                    candidates.push((*idx, acc_prob));
                }
                break;
            }
            candidates.push((*idx, acc_prob));
        }

        // sample
        let choices: Vec<usize> = candidates.iter().map(|&(idx, _)| idx).collect();
        let probs: Vec<f32> = candidates.iter().map(|&(_, prob)| prob).collect();
        let dist = WeightedIndex::new(probs)?;
        let mut rng = rand::rng();
        let token_id = choices[dist.sample(&mut rng)];
        Ok(token_id as u32)
    }

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let logits_t = logits
            .iter()
            .map(|&x| x / self.temperature)
            .collect::<Vec<f32>>();
        let max_logit = logits_t.iter().fold(f32::MIN, |a, &b| a.max(b));
        let exps: Vec<f32> = logits_t.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();
        exps.iter().map(|&exp| exp / sum_exps).collect()
    }
}
