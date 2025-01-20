use aksr::Builder;
use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, Array2, Array3, Axis, IxDyn};
use ndarray_npy::ReadNpyExt;

use crate::{
    BaseModelTextual, Bbox, DType, Engine, Hub, Keypoint, LogitsSampler, Options, Processor, Scale,
    Task, Ts, Xs, Ys, X, Y,
};

#[derive(Builder, Debug)]
pub struct Moondream2 {
    vision_encoder: VisionEncoder,
    vision_projection: VisionProjection,
    pub text_decoder: BaseModelTextual,
    text_encoder: BaseModelTextual,
    coord_decoder: Option<BaseModelTextual>,
    coord_encoder: Option<BaseModelTextual>,
    size_decoder: Option<BaseModelTextual>,
    size_encoder: Option<BaseModelTextual>,
    initial_kv_cache: X, // TODO: use f16
    scale: Scale,
    dtype: DType,
    max_length: usize,
    eos_token_id: u32,
    max_objects: usize,
}

impl Moondream2 {
    // TODO
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        options_vision_encoder: Options,
        options_vision_projection: Options,
        options_text_encoder: Options,
        options_text_decoder: Options,
        options_coord_encoder: Option<Options>,
        options_coord_decoder: Option<Options>,
        options_size_encoder: Option<Options>,
        options_size_decoder: Option<Options>,
    ) -> Result<Self> {
        let max_length = 2048;
        let max_objects = 50;
        let eos_token_id = 50256;
        let dtype = options_vision_encoder.model_dtype;
        let scale = options_vision_encoder
            .model_scale
            .unwrap_or(Scale::Billion(0.5));
        let initial_kv_cache: X = KVCache::new(&scale, &dtype)?.0.into();
        let vision_encoder = VisionEncoder::new(options_vision_encoder)?;
        let vision_projection = VisionProjection::new(options_vision_projection)?;
        let text_decoder = BaseModelTextual::new(options_text_decoder)?;
        let text_encoder = BaseModelTextual::new(options_text_encoder)?;
        let coord_decoder = options_coord_decoder
            .map(BaseModelTextual::new)
            .transpose()?;
        let coord_encoder = options_coord_encoder
            .map(BaseModelTextual::new)
            .transpose()?;
        let size_decoder = options_size_decoder
            .map(BaseModelTextual::new)
            .transpose()?;
        let size_encoder = options_size_encoder
            .map(BaseModelTextual::new)
            .transpose()?;

        Ok(Self {
            vision_encoder,
            vision_projection,
            text_decoder,
            initial_kv_cache,
            max_length,
            max_objects,
            text_encoder,
            coord_decoder,
            coord_encoder,
            size_encoder,
            size_decoder,
            eos_token_id,
            scale,
            dtype,
        })
    }

    pub fn encode_image(&mut self, x: &DynamicImage) -> Result<X> {
        let patches_emb = self.vision_encoder.encode(x)?.clone().insert_axis(0)?;
        let image_embedding = self.vision_projection.inference(patches_emb.into())?[0].to_owned();

        Ok(image_embedding)
    }

    pub fn forward(&mut self, xs: &[DynamicImage], task: &Task) -> Result<Ys> {
        let mut ys: Vec<Y> = Vec::new();
        for x in xs.iter() {
            let y = self.forward_once(x, task)?;
            ys.push(y);
        }

        Ok(ys.into())
    }

    pub fn forward_once(&mut self, images: &DynamicImage, task: &Task) -> Result<Y> {
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
                    .chain(
                        &self
                            .text_encoder
                            .processor()
                            .encode_text_ids(query, false)?,
                    )
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
                            .text_encoder
                            .processor()
                            .encode_text_ids(&format!(" {}", object), false)?,
                    )
                    .chain(&[628.])
                    .cloned()
                    .collect();
                let (_, y_bboxes) =
                    self.generate_points_boxes(&input_ids, kv_cache, object, true)?;

                Ok(Y::default().with_bboxes(&y_bboxes))
            }
            Task::OpenSetKeypointsDetection(object) => {
                let input_ids: Vec<_> = [198., 198., 12727., 25.]
                    .iter()
                    .chain(
                        &self
                            .text_encoder
                            .processor()
                            .encode_text_ids(&format!(" {}", object), false)?,
                    )
                    .chain(&[628.])
                    .cloned()
                    .collect();
                let (y_kpts, _) =
                    self.generate_points_boxes(&input_ids, kv_cache, object, false)?;

                Ok(Y::default().with_keypoints(&y_kpts))
            }
            x => anyhow::bail!("Unsupported Moondream2 task: {}", x),
        }
    }

    fn generate_text(&mut self, input_ids: &[f32], kv_cache: Array<f32, IxDyn>) -> Result<String> {
        let input_ids = X::from(input_ids.to_vec()).insert_axis(0)?;
        let mut input_embeds = self.text_encoder.inference(Xs::from(input_ids))?[0].to_owned();
        let logits_sampler = LogitsSampler::new();
        let mut token_ids: Vec<u32> = Vec::new();
        let mut pos = self.vision_projection.seq_len() + self.initial_kv_cache.shape()[4];
        let mut inc = input_embeds.shape()[1];
        let mut kv_cache = kv_cache.clone();

        // generate
        for _ in 0..self.max_length {
            // TODO
            let input = Xs::from(vec![
                input_embeds.clone(),
                kv_cache
                    .slice(s![.., .., .., .., ..pos, ..])
                    .into_owned()
                    .into_dyn()
                    .into(),
            ]);
            let decoder_outputs = self.text_decoder.inference(input)?;

            // update
            let logits = &decoder_outputs["logits"];
            let new_kv_cache = &decoder_outputs["new_kv_cache"];
            kv_cache
                .slice_mut(s![.., .., .., .., pos..pos + inc, ..])
                .assign(new_kv_cache);
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
            input_embeds = self.text_encoder.inference(Xs::from(next_tokens))?[0].to_owned();
        }

        let text = self
            .text_encoder
            .processor()
            .decode_tokens(&token_ids, true)?;

        Ok(text)
    }

    fn generate_points_boxes(
        &mut self,
        input_ids: &[f32],
        kv_cache: Array<f32, IxDyn>,
        object: &str,
        generate_boxes: bool,
    ) -> Result<(Vec<Vec<Keypoint>>, Vec<Bbox>)> {
        let mut y_bboxes: Vec<Bbox> = Vec::new();
        let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
        let (image_height, image_width) = self.vision_encoder.processor.image0s_size[0];
        let mut pos = self.vision_projection.seq_len() + self.initial_kv_cache.shape()[4];
        let logits_sampler = LogitsSampler::new();

        // initial input_embeds
        let input_ids = X::from(input_ids.to_vec()).insert_axis(0)?;
        let mut hidden = self.text_encoder.inference(Xs::from(input_ids))?[0].to_owned();
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
            let cx = self
                .coord_decoder
                .as_mut()
                .unwrap()
                .inference(Xs::from(input))?[0]
                .clone(); // [1024]
            let ratio = cx.shape()[0] as f32;
            let cx = logits_sampler
                .decode(cx.as_slice().context("Failed to get slice for `cx`")?)?
                as f32
                / ratio;
            hidden = self
                .coord_encoder
                .as_mut()
                .unwrap()
                .inference(Xs::from(X::from(vec![cx])))?[0]
                .clone()
                .insert_axis(0)?
                .insert_axis(0)?;

            // cy
            let _logits = self.run_decoder(&mut hidden, &mut kv_cache, &mut pos)?;
            let input: X = hidden.slice(s![0, -1, ..]).into_owned().into_dyn().into();
            let cy = self
                .coord_decoder
                .as_mut()
                .unwrap()
                .inference(Xs::from(input))?[0]
                .clone();
            let ratio = cy.shape()[0] as f32;

            let cy = logits_sampler
                .decode(cy.as_slice().context("Failed to get slice for `cy`")?)?
                as f32
                / ratio;

            hidden = self
                .coord_encoder
                .as_mut()
                .unwrap()
                .inference(Xs::from(X::from(vec![cy])))?[0]
                .clone()
                .insert_axis(0)?
                .insert_axis(0)?;

            if !generate_boxes {
                y_kpts.push(vec![Keypoint::from((
                    cx * image_width as f32,
                    cy * image_height as f32,
                    0,
                ))
                .with_name(object)]);

                // keep?
                if y_kpts.len() > self.max_objects {
                    break;
                }
            } else {
                // wh
                let _logits = self.run_decoder(&mut hidden, &mut kv_cache, &mut pos)?;
                let input: X = hidden.slice(s![0, -1, ..]).into_owned().into_dyn().into();
                let size = self
                    .size_decoder
                    .as_mut()
                    .unwrap()
                    .inference(Xs::from(input))?[0]
                    .clone(); // [2, 1024]

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

                hidden = self
                    .size_encoder
                    .as_mut()
                    .unwrap()
                    .inference(Xs::from(X::from(vec![w, h])))?[0]
                    .clone()
                    .insert_axis(0)?
                    .insert_axis(0)?; // [1024]

                let xmin = cx - w / 2.;
                let ymin = cy - h / 2.;

                y_bboxes.push(
                    Bbox::from((
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
        let kv_cache_new = self.text_decoder.inference(Xs::from(vec![
            image_embedding.clone(),
            self.initial_kv_cache.clone(),
        ]))?["new_kv_cache"]
            .to_owned();

        // TODO
        let kv_cache_new = ndarray::concatenate(
            Axis(4),
            &[kv_cache_new.view(), self.initial_kv_cache.view()],
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
        let decoder_outputs = self.text_decoder.inference(Xs::from(vec![
            input_embeds.clone(),
            kv_cache
                .slice(s![.., .., .., .., ..*pos, ..])
                .into_owned()
                .into_dyn()
                .into(),
        ]))?;
        let hidden = &decoder_outputs["hidden"];
        let new_kv_cache = &decoder_outputs["new_kv_cache"];

        // update
        let inc = hidden.shape()[1]; // -2
        kv_cache
            .slice_mut(s![.., .., .., .., *pos..*pos + inc, ..])
            .assign(new_kv_cache);
        *pos += inc;
        *input_embeds = hidden.to_owned();

        Ok(decoder_outputs["logits"].to_owned())
    }
}

#[derive(Debug, Builder)]
pub struct VisionEncoder {
    engine: Engine,
    num_patch: usize,
    patch_size: usize,
    processor: Processor,
    ts: Ts,
}

impl VisionEncoder {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (num_patch, patch_size, ts) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&378.into()).opt(),
            engine.ts.clone(),
        );
        let processor = options
            .to_processor()?
            .with_image_width(patch_size as _)
            .with_image_height(patch_size as _);

        Ok(Self {
            engine,
            patch_size,
            num_patch,
            processor,
            ts,
        })
    }

    fn create_patches(
        image: &DynamicImage,
        image_patch_size: usize,
    ) -> (Vec<DynamicImage>, (u32, u32)) {
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

                patches.push(DynamicImage::from(cropped));
            }
        }

        (patches, selected_template)
    }

    pub fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn encode(&mut self, x: &DynamicImage) -> Result<X> {
        let (patches, selected_template) = Self::create_patches(x, self.patch_size);
        let patches = self.processor.process_images(&patches)?;
        let template = (
            (selected_template.0 as usize),
            (selected_template.1 as usize),
        );
        let patch_emb = self.inference(patches.clone().into())?[0].clone();
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

#[derive(Debug, Builder)]
pub struct VisionProjection {
    engine: Engine,
    seq_len: usize,
    ts: Ts,
}

impl VisionProjection {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (seq_len, ts) = (engine.inputs_minoptmax[0][1].opt(), engine.ts.clone());

        Ok(Self {
            engine,
            seq_len,
            ts,
        })
    }

    pub fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
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
