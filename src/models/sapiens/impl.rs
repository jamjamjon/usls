use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array2, Axis};

use crate::{elapsed_module, Config, Engine, Image, Mask, Ops, Polygon, Processor, Task, Xs, Y};

#[derive(Builder, Debug)]
pub struct Sapiens {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    task: Task,
    names_body: Vec<String>,
    processor: Processor,
    spec: String,
}

impl Sapiens {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&1024.into()).opt(),
            engine.try_width().unwrap_or(&768.into()).opt(),
        );
        let task = config.task.expect("No sapiens task specified.");
        let names_body = config.class_names;
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            task,
            names_body,
            processor,
            spec,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("Sapiens", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("Sapiens", "inference", self.inference(ys)?);
        let ys = elapsed_module!("Sapiens", "postprocess", {
            if let Task::InstanceSegmentation = self.task {
                self.postprocess_seg(ys)?
            } else {
                unimplemented!()
            }
        });

        Ok(ys)
    }

    pub fn postprocess_seg(&self, xs: Xs) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (idx, b) in xs[0].axis_iter(Axis(0)).enumerate() {
            // rescale
            let (h1, w1) = (
                self.processor.images_transform_info[idx].height_src,
                self.processor.images_transform_info[idx].width_src,
            );
            let masks = Ops::interpolate_3d(b.to_owned(), w1 as _, h1 as _, "Bilinear")?;

            // generate mask
            let mut mask = Array2::<usize>::zeros((h1 as _, w1 as _));
            let mut ids = Vec::new();
            for hh in 0..h1 {
                for ww in 0..w1 {
                    let pt_slice = masks.slice(s![.., hh as usize, ww as usize]);
                    let (i, c) = match pt_slice
                        .into_iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                    {
                        Some((i, c)) => (i, c),
                        None => continue,
                    };

                    if *c <= 0. || i == 0 {
                        continue;
                    }
                    mask[[hh as _, ww as _]] = i;

                    if !ids.contains(&i) {
                        ids.push(i);
                    }
                }
            }

            // generate masks and polygons
            let mut y_masks: Vec<Mask> = Vec::new();
            let mut y_polygons: Vec<Polygon> = Vec::new();
            for i in ids.iter() {
                let luma = mask
                    .mapv(|x| if x == *i { 255 } else { 0 })
                    .into_raw_vec_and_offset()
                    .0;
                let mut mask = match Mask::new(&luma, w1, h1) {
                    Ok(luma) => luma.with_id(*i),
                    Err(_) => continue,
                };
                if let Some(polygon) = mask.polygon() {
                    y_polygons.push(polygon);
                }
                if !self.names_body.is_empty() {
                    mask = mask.with_name(&self.names_body[*i]);
                }
                y_masks.push(mask);
            }
            ys.push(Y::default().with_masks(&y_masks).with_polygons(&y_polygons));
        }

        Ok(ys)
    }
}
