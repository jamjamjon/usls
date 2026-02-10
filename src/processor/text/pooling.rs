use ndarray::{s, Array2, Axis, Ix3};

use crate::tensor::{XView, X};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Cls,
    First,
    Last,
    Mean,
    Max,
}

impl Pooling {
    /// Apply pooling on hidden states.
    ///
    /// - last_hidden_states: [B, T, D]
    /// - mask (optional): [B, T], true = valid token
    ///
    /// Returns:
    /// - pooled embedding: [B, D]
    pub fn apply(&self, last_hidden_states: &XView<f32>, mask: Option<&XView<bool>>) -> X<f32> {
        let states = last_hidden_states
            .0
            .view()
            .into_dimensionality::<Ix3>()
            .expect("pooling expects a 3D tensor");

        let mask_view = mask.map(|m| {
            m.0.view()
                .into_dimensionality::<ndarray::Dim<[usize; 2]>>()
                .expect("mask must be 2D")
        });

        let pooled = match self {
            Pooling::Cls | Pooling::First => states.index_axis(Axis(1), 0).to_owned(),
            Pooling::Last => {
                let t = states.len_of(Axis(1));
                states.index_axis(Axis(1), t - 1).to_owned()
            }
            Pooling::Mean => {
                let (b, _, d) = states.dim();
                let mut out = Array2::zeros((b, d));
                for i in 0..b {
                    let mut count = 0.0;
                    for j in 0..states.len_of(Axis(1)) {
                        let allow = mask_view.as_ref().map(|m| m[[i, j]]).unwrap_or(true);
                        if allow {
                            out.row_mut(i).scaled_add(1.0, &states.slice(s![i, j, ..]));
                            count += 1.0;
                        }
                    }
                    if count > 0.0 {
                        *out.row_mut(i) /= count;
                    }
                }
                out
            }
            Pooling::Max => {
                let (b, _, d) = states.dim();
                let mut out = Array2::from_elem((b, d), f32::NEG_INFINITY);
                for i in 0..b {
                    for j in 0..states.len_of(Axis(1)) {
                        let allow = mask_view.as_ref().map(|m| m[[i, j]]).unwrap_or(true);
                        if allow {
                            let row = states.slice(s![i, j, ..]);
                            for k in 0..d {
                                out[[i, k]] = out[[i, k]].max(row[k]);
                            }
                        }
                    }
                }
                out
            }
        };

        X::from(pooled.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_states() -> X<f32> {
        let data =
            ndarray::Array3::from_shape_vec((1, 3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        X::from(data)
    }

    fn build_mask() -> X<bool> {
        let mask_arr = ndarray::Array2::from_shape_vec((1, 3), vec![true, false, true]).unwrap();
        X::from(mask_arr.map(|v| *v as u8 != 0))
    }

    #[test]
    fn pooling_cls_first_use_first_token() {
        let states = build_states();
        let view = states.view();
        let pooled = Pooling::Cls.apply(&view, None);
        let out = pooled
            .0
            .into_dimensionality::<ndarray::Dim<[usize; 2]>>()
            .unwrap();
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 2.0);
    }

    #[test]
    fn pooling_last() {
        let states = build_states();
        let view = states.view();
        let pooled = Pooling::Last.apply(&view, None);
        let out = pooled
            .0
            .into_dimensionality::<ndarray::Dim<[usize; 2]>>()
            .unwrap();
        assert_eq!(out[[0, 0]], 5.0);
        assert_eq!(out[[0, 1]], 6.0);
    }

    #[test]
    fn pooling_mean_with_mask() {
        let states = build_states();
        let view = states.view();
        let mask = build_mask();
        let mask_view = mask.view();
        let pooled = Pooling::Mean.apply(&view, Some(&mask_view));
        let out = pooled
            .0
            .into_dimensionality::<ndarray::Dim<[usize; 2]>>()
            .unwrap();
        // mask keeps token 0 and 2 (values 1,2 and 5,6)
        assert_eq!(out[[0, 0]], (1.0 + 5.0) / 2.0);
        assert_eq!(out[[0, 1]], (2.0 + 6.0) / 2.0);
    }

    #[test]
    fn pooling_max_ignores_masked_positions() {
        let states = build_states();
        let view = states.view();
        let mask = build_mask();
        let mask_view = mask.view();
        let pooled = Pooling::Max.apply(&view, Some(&mask_view));
        let out = pooled
            .0
            .into_dimensionality::<ndarray::Dim<[usize; 2]>>()
            .unwrap();
        // only allowed rows are 0 and 2, max per column is max(1,5)=5 and max(2,6)=6
        assert_eq!(out[[0, 0]], 5.0);
        assert_eq!(out[[0, 1]], 6.0);
    }
}
