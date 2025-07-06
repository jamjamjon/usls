use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};

use crate::{elapsed_module, models::BaseModelVisual, Config, Image, Keypoint, Text, Xs, Y};

#[derive(Builder, Debug)]
pub struct SLANet {
    base: BaseModelVisual,
    td_tokens: Vec<&'static str>,
    eos: usize,
    sos: usize,
    spec: String,
}

impl SLANet {
    pub fn new(config: Config) -> Result<Self> {
        let base = BaseModelVisual::new(config)?;
        let spec = base.engine().spec().to_owned();
        let sos = 0;
        let eos = base.processor().vocab().len() - 1;
        let td_tokens = vec!["<td>", "<td", "<td></td>"];

        Ok(Self {
            base,
            td_tokens,
            eos,
            sos,
            spec,
        })
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("SLANet", "preprocess", self.base.preprocess(xs)?);
        let ys = elapsed_module!("SLANet", "inference", self.base.inference(ys)?);
        let ys = elapsed_module!("SLANet", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (bid, (bboxes, structures)) in xs[0]
            .axis_iter(Axis(0))
            .zip(xs[1].axis_iter(Axis(0)))
            .enumerate()
        {
            let mut y_texts: Vec<&str> = vec!["<html>", "<body>", "<table>"];
            let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
            let (image_height, image_width) = (
                self.base.processor().images_transform_info[bid].height_src,
                self.base.processor().images_transform_info[bid].width_src,
            );
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
                            .map(|i| {
                                Keypoint::from((x14[i], y14[i]))
                                    .with_id(i)
                                    .with_confidence(1.)
                            })
                            .collect(),
                    );
                }

                y_texts.push(token);
            }

            // clean up text
            if y_texts.len() == 3 {
                y_texts.clear();
            } else {
                y_texts.extend_from_slice(&["</table>", "</body>", "</html>"]);
            }

            ys.push(
                Y::default()
                    .with_keypointss(&y_kpts)
                    .with_texts(&y_texts.into_iter().map(Text::from).collect::<Vec<_>>()),
            );
        }

        Ok(ys)
    }
}
