use aksr::Builder;
use anyhow::{Context, Result};
use image::GenericImageView;
use ndarray::{s, Array, Array2, Array3, Axis, IxDyn};
use ndarray_npy::ReadNpyExt;

use crate::{
    elapsed_module, ort_inputs, Config, DType, Engine, FromConfig, Hbb, Hub, Image, ImageProcessor,
    Keypoint, LogitsSampler, Scale, Task, TextProcessor, X, Y,
};

#[derive(Builder, Debug)]
pub struct Moondream2 {
    vision_encoder: Engine,
    vision_projection: Engine,
    text_decoder: Engine,
    text_encoder: Engine,
    coord_decoder: Option<Engine>,
    coord_encoder: Option<Engine>,
    size_decoder: Option<Engine>,
    size_encoder: Option<Engine>,
    initial_kv_cache: X, // TODO: use f16
    scale: Scale,
    dtype: DType,
    max_length: usize,
    eos_token_id: u32,
    max_objects: usize,
    num_patch: usize,
    patch_size: usize,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
    seq_len: usize,
}

impl Moondream2 {
    pub fn new(mut config: Config) -> Result<Self> {
        let max_length = 2048;
        let max_objects = 50;
        let eos_token_id = 50256;
        let dtype = config
            .get_module(&crate::Module::VisualEncoder)
            .map(|e| e.dtype)
            .unwrap_or(crate::DType::Fp32);
        let scale = config.scale.take().unwrap_or(Scale::Billion(0.5));
        let initial_kv_cache: X = KVCache::new(&scale, &dtype)?.0.into();
        let vision_encoder =
            Engine::from_config(config.take_module(&crate::Module::VisualEncoder)?)?;
        let vision_projection =
            Engine::from_config(config.take_module(&crate::Module::VisualProjection)?)?;
        let text_decoder =
            Engine::from_config(config.take_module(&crate::Module::TextualDecoder)?)?;
        let text_encoder =
            Engine::from_config(config.take_module(&crate::Module::TextualEncoder)?)?;
        let coord_decoder = config
            .take_module(&crate::Module::CoordDecoder)
            .ok()
            .map(Engine::from_config)
            .transpose()?;
        let coord_encoder = config
            .take_module(&crate::Module::CoordEncoder)
            .ok()
            .map(Engine::from_config)
            .transpose()?;
        let size_decoder = config
            .take_module(&crate::Module::SizeDecoder)
            .ok()
            .map(Engine::from_config)
            .transpose()?;
        let size_encoder = config
            .take_module(&crate::Module::SizeEncoder)
            .ok()
            .map(Engine::from_config)
            .transpose()?;
        let (num_patch, patch_size) = (
            vision_encoder.batch().opt(),
            vision_encoder.try_height().unwrap_or(&378.into()).opt(),
        );
        let seq_len = vision_projection.inputs_minoptmax[0][1].opt();
        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(patch_size as _)
            .with_image_height(patch_size as _);
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();

        Ok(Self {
            vision_encoder,
            vision_projection,
            text_decoder,
            text_encoder,
            coord_decoder,
            coord_encoder,
            size_encoder,
            size_decoder,
            initial_kv_cache,
            max_length,
            max_objects,
            eos_token_id,
            scale,
            dtype,
            num_patch,
            patch_size,
            image_processor,
            text_processor,
            seq_len,
        })
    }

    pub fn encode_image(&mut self, x: &Image) -> Result<X> {
        let patches_emb = elapsed_module!(
            "Moondream2",
            "preprocess_image",
            self.encode(x)?.clone().insert_axis(0)?
        );
        let image_embedding = elapsed_module!("Moondream2", "encode_image", {
            let ys = self.vision_projection.run(ort_inputs![patches_emb]?)?;
            X::from(ys.get::<f32>(0)?)
        });

        Ok(image_embedding)
    }

    pub fn forward(&mut self, xs: &[Image], task: &Task) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for x in xs.iter() {
            let y = self.forward_once(x, task)?;
            ys.push(y);
        }

        Ok(ys)
    }

    pub fn forward_once(&mut self, images: &Image, task: &Task) -> Result<Y> {
        let image_embedding = self.encode_image(images)?;
        let kv_cache = self.prepare_kv_cache(&image_embedding)?;

        match task {
            Task::Caption(n) => {
                let input_ids = match n {
                    0 => vec![198., 198., 16438., 8305., 25.],
                    _ => vec![198., 198., 24334., 1159., 25.],
                };
                let text = self.generate_text(&input_ids, kv_cache)?;
                let y = Y::default().with_texts(&[text.into()]);

                Ok(y)
            }
            Task::Vqa(query) => {
                let input_ids: Vec<_> = [198., 198., 24361., 25.]
                    .iter()
                    .chain(&self.text_processor.encode_text_ids(query, false)?)
                    .chain(&[198., 198., 33706., 25.])
                    .cloned()
                    .collect();

                let text = self.generate_text(&input_ids, kv_cache)?;
                let y = Y::default().with_texts(&[text.into()]);

                Ok(y)
            }
            Task::OpenSetDetection(object) => {
                let input_ids: Vec<_> = [198., 198., 47504., 25.]
                    .iter()
                    .chain(
                        &self
                            .text_processor
                            .encode_text_ids(&format!(" {}", object), false)?,
                    )
                    .chain(&[628.])
                    .cloned()
                    .collect();
                let (_, y_bboxes) =
                    self.generate_points_boxes(&input_ids, kv_cache, object, true)?;

                Ok(Y::default().with_hbbs(&y_bboxes))
            }
            Task::OpenSetKeypointsDetection(object) => {
                let input_ids: Vec<_> = [198., 198., 12727., 25.]
                    .iter()
                    .chain(
                        &self
                            .text_processor
                            .encode_text_ids(&format!(" {}", object), false)?,
                    )
                    .chain(&[628.])
                    .cloned()
                    .collect();
                let (y_kpts, _) =
                    self.generate_points_boxes(&input_ids, kv_cache, object, false)?;

                Ok(Y::default().with_keypointss(&y_kpts))
            }
            x => anyhow::bail!("Unsupported Moondream2 task: {}", x),
        }
    }

    fn generate_text(&mut self, input_ids: &[f32], kv_cache: Array<f32, IxDyn>) -> Result<String> {
        let input_ids = X::from(input_ids.to_vec()).insert_axis(0)?;
        let mut input_embeds = {
            let ys = self.text_encoder.run(ort_inputs![input_ids]?)?;
            X::from(ys.get::<f32>(0)?)
        };
        let logits_sampler = LogitsSampler::new();
        let mut token_ids: Vec<u32> = Vec::new();
        let mut pos = self.seq_len + self.initial_kv_cache.shape()[4];
        let mut inc = input_embeds.shape()[1];
        let mut kv_cache = kv_cache.clone();

        // generate
        for _ in 0..self.max_length {
            let kv_cache_slice: X = kv_cache
                .slice(s![.., .., .., .., ..pos, ..])
                .into_owned()
                .into_dyn()
                .into();
            let (logits, new_kv_cache) = {
                let decoder_outputs = self
                    .text_decoder
                    .run(ort_inputs![input_embeds.clone(), kv_cache_slice]?)?;
                let logits = X::from(decoder_outputs.get_by_name::<f32>("logits")?);
                let new_kv_cache = X::from(decoder_outputs.get_by_name::<f32>("new_kv_cache")?);
                (logits, new_kv_cache)
            };

            // update
            kv_cache
                .slice_mut(s![.., .., .., .., pos..pos + inc, ..])
                .assign(&new_kv_cache.0);
            pos += inc;

            // decode
            let token_id = logits_sampler.decode(
                logits
                    .slice(s![-1, ..])
                    .as_slice()
                    .context("Failed to get slice when decode `logits`")?,
            )?;

            // break
            if token_id == self.eos_token_id {
                break;
            }

            // update
            token_ids.push(token_id);
            inc = 1;

            // encode
            let next_tokens = X::from(vec![token_id as f32]).insert_axis(1)?;
            input_embeds = {
                let ys = self.text_encoder.run(ort_inputs![next_tokens]?)?;
                X::from(ys.get::<f32>(0)?)
            };
        }

        let text = self.text_processor.decode_tokens(&token_ids, true)?;

        Ok(text)
    }

    fn generate_points_boxes(
        &mut self,
        input_ids: &[f32],
        kv_cache: Array<f32, IxDyn>,
        object: &str,
        generate_boxes: bool,
    ) -> Result<(Vec<Vec<Keypoint>>, Vec<Hbb>)> {
        let mut y_bboxes: Vec<Hbb> = Vec::new();
        let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
        let (image_height, image_width) = (
            self.image_processor.images_transform_info()[0].height_src,
            self.image_processor.images_transform_info()[0].width_src,
        );

        let mut pos = self.seq_len + self.initial_kv_cache.shape()[4];
        let logits_sampler = LogitsSampler::new();

        // initial input_embeds
        let input_ids = X::from(input_ids.to_vec()).insert_axis(0)?;
        let mut hidden = {
            let ys = self.text_encoder.run(ort_inputs![input_ids]?)?;
            X::from(ys.get::<f32>(0)?)
        };
        let mut kv_cache = kv_cache;

        // generate
        loop {
            let logits = self.run_decoder(&mut hidden, &mut kv_cache, &mut pos)?;

            // decode
            let token_id = logits_sampler.decode(
                logits
                    .slice(s![-1, ..])
                    .as_slice()
                    .context("Failed to get slice for `logits`")?,
            )?;

            // break
            if token_id == self.eos_token_id {
                break;
            }

            // cx
            let input: X = hidden.slice(s![0, -1, ..]).into_owned().into_dyn().into();
            let cx = {
                let ys = self
                    .coord_decoder
                    .as_mut()
                    .unwrap()
                    .run(ort_inputs![input]?)?;
                X::from(ys.get::<f32>(0)?)
            }; // [1024]
            let ratio = cx.shape()[0] as f32;
            let cx = logits_sampler
                .decode(cx.as_slice().context("Failed to get slice for `cx`")?)?
                as f32
                / ratio;
            hidden = {
                let cx_input = X::from(vec![cx]);
                let ys = self
                    .coord_encoder
                    .as_mut()
                    .unwrap()
                    .run(ort_inputs![cx_input]?)?;
                X::from(ys.get::<f32>(0)?).insert_axis(0)?.insert_axis(0)?
            };

            // cy
            let _logits = self.run_decoder(&mut hidden, &mut kv_cache, &mut pos)?;
            let input: X = hidden.slice(s![0, -1, ..]).into_owned().into_dyn().into();
            let cy = {
                let ys = self
                    .coord_decoder
                    .as_mut()
                    .unwrap()
                    .run(ort_inputs![input]?)?;
                X::from(ys.get::<f32>(0)?)
            };
            let ratio = cy.shape()[0] as f32;

            let cy = logits_sampler
                .decode(cy.as_slice().context("Failed to get slice for `cy`")?)?
                as f32
                / ratio;

            hidden = {
                let cy_input = X::from(vec![cy]);
                let ys = self
                    .coord_encoder
                    .as_mut()
                    .unwrap()
                    .run(ort_inputs![cy_input]?)?;
                X::from(ys.get::<f32>(0)?).insert_axis(0)?.insert_axis(0)?
            };

            if !generate_boxes {
                y_kpts.push(vec![Keypoint::from((
                    cx * image_width as f32,
                    cy * image_height as f32,
                ))
                .with_id(0)
                .with_confidence(1.)
                .with_name(object)]);

                // keep?
                if y_kpts.len() > self.max_objects {
                    break;
                }
            } else {
                // wh
                let _logits = self.run_decoder(&mut hidden, &mut kv_cache, &mut pos)?;
                let input: X = hidden.slice(s![0, -1, ..]).into_owned().into_dyn().into();
                let size = {
                    let ys = self
                        .size_decoder
                        .as_mut()
                        .unwrap()
                        .run(ort_inputs![input]?)?;
                    X::from(ys.get::<f32>(0)?)
                }; // [2, 1024]

                let ratio = size.shape()[1] as f32;
                let w = logits_sampler.decode(
                    size.slice(s![0, ..])
                        .as_slice()
                        .context("Failed to get slice when decode `w`")?,
                )? as f32
                    / ratio;

                // h
                let h = logits_sampler.decode(
                    size.slice(s![1, ..])
                        .as_slice()
                        .context("Failed to get slice when decode `h`")?,
                )? as f32
                    / ratio;

                hidden = {
                    let wh_input = X::from(vec![w, h]);
                    let ys = self
                        .size_encoder
                        .as_mut()
                        .unwrap()
                        .run(ort_inputs![wh_input]?)?;
                    X::from(ys.get::<f32>(0)?).insert_axis(0)?.insert_axis(0)?
                }; // [1024]

                let xmin = cx - w / 2.;
                let ymin = cy - h / 2.;

                y_bboxes.push(
                    Hbb::from((
                        xmin * image_width as f32,
                        ymin * image_height as f32,
                        w * image_width as f32,
                        h * image_height as f32,
                    ))
                    .with_name(object)
                    .with_id(0)
                    .with_confidence(1.),
                );

                // Keep?
                if y_bboxes.len() > self.max_objects {
                    break;
                }
            }
        }

        Ok((y_kpts, y_bboxes))
    }

    fn prepare_kv_cache(&mut self, image_embedding: &X) -> Result<Array<f32, IxDyn>> {
        let kv_cache_new = {
            let ys = self.text_decoder.run(ort_inputs![
                image_embedding.clone(),
                self.initial_kv_cache.clone()
            ]?)?;
            X::from(ys.get_by_name::<f32>("new_kv_cache")?)
        };

        let kv_cache_new = ndarray::concatenate(
            Axis(4),
            &[
                kv_cache_new.0.view().into_dyn(),
                self.initial_kv_cache.0.view().into_dyn(),
            ],
        )?;

        // fill with max sequence length
        let mut shapes = self.initial_kv_cache.shape().to_vec();
        shapes[4] = self.max_length;
        let mut kv_cache = Array::zeros(shapes);
        kv_cache
            .slice_mut(s![.., .., .., .., ..kv_cache_new.dim()[4], ..])
            .assign(&kv_cache_new);

        Ok(kv_cache.into_dyn())
    }

    fn run_decoder(
        &mut self,
        input_embeds: &mut X,
        kv_cache: &mut Array<f32, IxDyn>,
        pos: &mut usize,
    ) -> Result<X> {
        let kv_cache_slice: X = kv_cache
            .slice(s![.., .., .., .., ..*pos, ..])
            .into_owned()
            .into_dyn()
            .into();
        let (hidden, new_kv_cache, logits) = {
            let decoder_outputs = self
                .text_decoder
                .run(ort_inputs![input_embeds.clone(), kv_cache_slice]?)?;
            let hidden = X::from(decoder_outputs.get_by_name::<f32>("hidden")?);
            let new_kv_cache = X::from(decoder_outputs.get_by_name::<f32>("new_kv_cache")?);
            let logits = X::from(decoder_outputs.get_by_name::<f32>("logits")?);
            (hidden, new_kv_cache, logits)
        };

        // update
        let inc = hidden.shape()[1]; // -2
        kv_cache
            .slice_mut(s![.., .., .., .., *pos..*pos + inc, ..])
            .assign(&new_kv_cache.0);
        *pos += inc;
        *input_embeds = hidden;

        Ok(logits)
    }

    fn create_patches(image: &Image, image_patch_size: usize) -> (Vec<Image>, (u32, u32)) {
        let mut patches = vec![image.clone()];
        let image = image.to_rgb8();

        let res_templates = vec![(1, 2), (2, 1), (2, 2)];
        let (im_width, im_height) = image.dimensions();
        let max_dim = im_width.max(im_height);
        let selected_template = if max_dim < (image_patch_size as f32 * 1.4) as u32 {
            (1, 1)
        } else {
            let aspect_ratio = im_width as f32 / im_height as f32;
            res_templates
                .into_iter()
                .min_by(|a, b| {
                    let diff_a = ((a.1 as f32 / a.0 as f32) - aspect_ratio).abs();
                    let diff_b = ((b.1 as f32 / b.0 as f32) - aspect_ratio).abs();
                    diff_a.partial_cmp(&diff_b).unwrap()
                })
                .unwrap()
        };
        let patch_width = im_width / selected_template.1;
        let patch_height = im_height / selected_template.0;

        for row in 0..selected_template.0 {
            for col in 0..selected_template.1 {
                let x_min = col * patch_width;
                let y_min = row * patch_height;
                let _x_max = x_min + patch_width;
                let _y_max = y_min + patch_height;
                let cropped = image
                    .view(x_min, y_min, patch_width, patch_height)
                    .to_image();

                patches.push(Image::from(cropped));
            }
        }

        (patches, selected_template)
    }

    pub fn encode(&mut self, x: &Image) -> Result<X> {
        let (patches, selected_template) = Self::create_patches(x, self.patch_size);
        let patches = self.image_processor.process(&patches)?;
        let template = (
            (selected_template.0 as usize),
            (selected_template.1 as usize),
        );
        let patch_emb = {
            let ys = self.vision_encoder.run(ort_inputs![patches]?)?;
            X::from(ys.get::<f32>(0)?)
        };
        let patch_emb = patch_emb.clone().0.into_dimensionality::<ndarray::Ix3>()?;
        let patch_emb = Self::process_patch_emb(patch_emb, template)?;
        let patch_emb = X::from(patch_emb.into_dyn()); // TODO .insert_axis(x),

        Ok(patch_emb)
    }

    fn process_patch_emb(patch_emb: Array3<f32>, template: (usize, usize)) -> Result<Array2<f32>> {
        let (_, seq_len, enc_dim) = patch_emb.dim(); // (N, 729, 720)
        let global_patch = patch_emb.slice(s![0, .., ..]).into_owned();
        if template == (1, 1) {
            Ok(ndarray::concatenate(
                Axis(1),
                &[global_patch.view(), global_patch.view()],
            )?)
        } else {
            let w = (seq_len as f32).sqrt() as usize;
            let mut rows = Vec::new();
            for r in 0..template.0 {
                let mut row = Vec::new();
                for c in 0..template.1 {
                    let idx = r * template.1 + c;
                    let patch = patch_emb.slice(s![idx, .., ..]).into_owned();
                    let patch = patch.into_shape_with_order((w, w, enc_dim))?;
                    row.push(patch);
                }
                let row_concat = ndarray::concatenate(
                    Axis(1),
                    &row.iter().map(|x| x.view()).collect::<Vec<_>>(),
                )?;
                rows.push(row_concat);
            }

            let patch_emb =
                ndarray::concatenate(Axis(0), &rows.iter().map(|x| x.view()).collect::<Vec<_>>())?;
            let patch_emb = Self::adaptive_avg_pool2d(patch_emb, (w, w))
                .into_shape_with_order((w * w, enc_dim))?;

            Ok(ndarray::concatenate(
                Axis(1),
                &[global_patch.view(), patch_emb.view()],
            )?)
        }
    }

    fn adaptive_avg_pool2d(x: Array3<f32>, output_size: (usize, usize)) -> Array3<f32> {
        let (height, width, channels) = x.dim();
        let (out_height, out_width) = output_size;
        let stride_h = height / out_height;
        let stride_w = width / out_width;
        let kernel_h = height - (out_height - 1) * stride_h;
        let kernel_w = width - (out_width - 1) * stride_w;
        let mut output = Array3::zeros((out_height, out_width, channels));
        for i in 0..out_height {
            for j in 0..out_width {
                let h_start = i * stride_h;
                let h_end = h_start + kernel_h;
                let w_start = j * stride_w;
                let w_end = w_start + kernel_w;

                for c in 0..channels {
                    let mut sum = 0.0;
                    let mut count = 0;

                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            if h < height && w < width {
                                sum += x[(h, w, c)];
                                count += 1;
                            }
                        }
                    }
                    output[(i, j, c)] = sum / count as f32;
                }
            }
        }

        output
    }
}

#[derive(Builder, Debug)]
struct KVCache(pub Array<f32, IxDyn>);

impl KVCache {
    pub fn new(scale: &Scale, dtype: &DType) -> Result<Self> {
        let f = format!("moondream2/{}-initial-kv-cache-{}.npy", scale, dtype);
        let f = Hub::default().try_fetch(&f)?;
        let file = std::fs::File::open(f)?;
        let x = Array::<f32, IxDyn>::read_npy(file)?.into_dyn();

        Ok(Self(x))
    }
}
