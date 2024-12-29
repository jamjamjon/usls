use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Axis};

use crate::{elapsed, models::BaseModelVisual, Keypoint, Options, Text, Ts, Xs, Ys, Y};

#[derive(Builder, Debug)]
pub struct SLANet {
    base: BaseModelVisual,
    td_tokens: Vec<&'static str>,
    eos: usize,
    sos: usize,
    ts: Ts,
    spec: String,
}

impl SLANet {
    pub fn summary(&mut self) {
        self.ts.summary();
    }

    pub fn new(options: Options) -> Result<Self> {
        let base = BaseModelVisual::new(options)?;
        let spec = base.engine().spec().to_owned();
        let sos = 0;
        let eos = base.processor().vocab().len() - 1;
        let td_tokens = vec!["<td>", "<td", "<td></td>"];
        let ts = base.ts().clone();

        Ok(Self {
            base,
            td_tokens,
            eos,
            sos,
            ts,
            spec,
        })
    }

    pub fn forward(&mut self, xs: &[DynamicImage]) -> Result<Ys> {
        let ys = elapsed!("preprocess", self.ts, { self.base.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.base.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    fn postprocess(&self, xs: Xs) -> Result<Ys> {
        let mut ys: Vec<Y> = Vec::new();
        for (bid, (bboxes, structures)) in xs[0]
            .axis_iter(Axis(0))
            .zip(xs[1].axis_iter(Axis(0)))
            .enumerate()
        {
            let mut y_texts: Vec<Text> = vec!["<html>".into(), "<body>".into(), "<table>".into()];
            let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
            let (image_height, image_width) = self.base.processor().image0s_size[bid];
            for (i, structure) in structures.axis_iter(Axis(0)).enumerate() {
                let (token_id, &_confidence) = match structure
                    .into_iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                {
                    None => continue,
                    Some((id, conf)) => (id, conf),
                };
                if token_id == self.eos {
                    break;
                }
                if token_id == self.sos {
                    continue;
                }

                // token
                let token = self.base.processor().vocab()[token_id].as_str();

                // keypoint
                if self.td_tokens.contains(&token) {
                    let slice_bboxes = bboxes.slice(s![i, ..]);
                    let x14 = slice_bboxes
                        .slice(s![0..;2])
                        .mapv(|x| x * image_width as f32);
                    let y14 = slice_bboxes
                        .slice(s![1..;2])
                        .mapv(|x| x * image_height as f32);
                    y_kpts.push(
                        (0..=3)
                            .map(|i| (x14[i], y14[i], i as isize).into())
                            .collect(),
                    );
                }

                y_texts.push(token.into());
            }

            // clean up text
            if y_texts.len() == 3 {
                y_texts.clear();
            } else {
                y_texts.extend_from_slice(&["</table>".into(), "</body>".into(), "</html>".into()]);
            }

            ys.push(Y::default().with_keypoints(&y_kpts).with_texts(&y_texts));
        }

        Ok(ys.into())
    }
}
