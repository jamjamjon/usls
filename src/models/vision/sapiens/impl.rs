use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array2, Axis};

use crate::{
    elapsed_module, inputs, Config, Engine, Engines, FromConfig, Image, ImageProcessor, Mask,
    Model, Module, Ops, Polygon, Task, Xs, X, Y,
};

/// Sapiens: Foundation for Human Vision Models
#[derive(Builder, Debug)]
pub struct Sapiens {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub task: Task,
    pub names_body: Vec<String>,
    pub processor: ImageProcessor,
    pub spec: String,
}

impl Model for Sapiens {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&1024.into()).opt(),
            engine.try_width().unwrap_or(&768.into()).opt(),
        );
        let task = config.task.expect("No sapiens task specified.");
        let names_body = config.inference.class_names;
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            task,
            names_body,
            processor,
            spec,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x = elapsed_module!("Sapiens", "preprocess", self.processor.process(images)?);
        let ys = elapsed_module!(
            "Sapiens",
            "inference",
            engines.run(&Module::Model, inputs![&x]?)?
        );
        elapsed_module!("Sapiens", "postprocess", self.postprocess(&ys))
    }
}

impl Sapiens {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let xs = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?;
        let xs = X::from(xs);

        if let Task::InstanceSegmentation = self.task {
            self.postprocess_seg(&xs)
        } else {
            unimplemented!()
        }
    }

    fn postprocess_seg(&self, xs: &X) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (idx, b) in xs.axis_iter(Axis(0)).enumerate() {
            // rescale
            let info = &self.processor.images_transform_info[idx];
            let (h1, w1) = (info.height_src, info.width_src);
            // Interpolate all channels at once using the unified interpolate function
            let (n, h, w) = (b.shape()[0], b.shape()[1], b.shape()[2]);
            let masks_flat = Ops::interpolate_nd(
                &b.to_owned().into_raw_vec_and_offset().0,
                n,
                w as _,
                h as _,
                w1 as _,
                h1 as _,
                false,
            )?;
            let masks = ndarray::Array::from_shape_vec((n, h1 as usize, w1 as usize), masks_flat)?
                .into_dyn();

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
