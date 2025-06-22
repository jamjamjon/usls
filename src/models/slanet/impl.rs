use aksr::Builder;
use anyhow::Result;

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
        let ys = elapsed_module!("slanet", "preprocess", self.base.preprocess(xs)?);
        let ys = elapsed_module!("slanet", "inference", self.base.inference(ys)?);
        let ys = elapsed_module!("slanet", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (bid, (bboxes, structures)) in xs[0].iter_dim(0).zip(xs[1].iter_dim(0)).enumerate() {
            let mut y_texts: Vec<&str> = vec!["<html>", "<body>", "<table>"];
            let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
            let (image_height, image_width) = (
                self.base.processor().images_transform_info[bid].height_src,
                self.base.processor().images_transform_info[bid].width_src,
            );
            for (i, structure) in structures.iter_dim(0).enumerate() {
                let structure_vec = structure.to_vec::<f32>()?;
                let (token_id, _confidence) = match structure_vec
                    .into_iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
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
                    let slice_bboxes = bboxes.slice(&[i..i + 1, 0..bboxes.shape()[1]])?;

                    // Extract x and y coordinates manually (every 2nd element starting from 0 and 1)
                    let mut x_coords = Vec::new();
                    let mut y_coords = Vec::new();

                    let slice_vec = slice_bboxes.to_vec::<f32>()?;
                    let bbox_width = slice_bboxes.shape()[1];
                    for coord_idx in (0..bbox_width).step_by(2) {
                        if coord_idx < bbox_width {
                            let x_val = slice_vec.get(coord_idx).copied().unwrap_or(0.0);
                            x_coords.push(x_val * image_width as f32);
                        }
                        if coord_idx + 1 < bbox_width {
                            let y_val = slice_vec.get(coord_idx + 1).copied().unwrap_or(0.0);
                            y_coords.push(y_val * image_height as f32);
                        }
                    }
                    y_kpts.push(
                        (0..=3)
                            .map(|idx| {
                                let x_coord = x_coords.get(idx).copied().unwrap_or(0.0);
                                let y_coord = y_coords.get(idx).copied().unwrap_or(0.0);
                                Keypoint::from((x_coord, y_coord))
                                    .with_id(idx)
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
