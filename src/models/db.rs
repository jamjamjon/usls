use crate::{
    ops, Annotator, Bbox, DynConf, MinOptMax, Options, OrtEngine, Point, Polygon, Results,
};
use anyhow::Result;
use image::{DynamicImage, ImageBuffer};
use ndarray::{Array, Axis, IxDyn};

#[derive(Debug)]
pub struct DB {
    engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    annotator: Annotator,
    confs: DynConf,
    saveout: Option<String>,
    names: Option<Vec<String>>,
}

impl DB {
    pub fn new(options: &Options) -> Result<Self> {
        let engine = OrtEngine::new(options)?;
        let (batch, height, width) = (
            engine.inputs_minoptmax()[0][0].to_owned(),
            engine.inputs_minoptmax()[0][2].to_owned(),
            engine.inputs_minoptmax()[0][3].to_owned(),
        );
        let annotator = Annotator::default();
        let names = Some(vec!["Text".to_string()]);
        let confs = DynConf::new(&options.confs, 1);
        engine.dry_run()?;

        Ok(Self {
            engine,
            names,
            confs,
            height,
            width,
            batch,
            saveout: options.saveout.to_owned(),
            annotator,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Results>> {
        let xs_ = ops::letterbox(xs, self.height.opt as u32, self.width.opt as u32)?;
        let ys = self.engine.run(&[xs_])?;
        let ys = self.postprocess(ys, xs)?;
        match &self.saveout {
            None => {}
            Some(saveout) => {
                for (img0, y) in xs.iter().zip(ys.iter()) {
                    let mut img = img0.to_rgb8();
                    self.annotator.plot(&mut img, y);
                    self.annotator.save(&img, saveout);
                }
            }
        }
        Ok(ys)
    }

    pub fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<Results>> {
        let mut ys = Vec::new();
        for (idx, mask) in xs[0].axis_iter(Axis(0)).enumerate() {
            let mut ys_bbox = Vec::new();
            // input image
            let image_width = xs0[idx].width() as f32;
            let image_height = xs0[idx].height() as f32;

            // h,w,1
            let h = mask.dim()[1];
            let w = mask.dim()[2];
            let mask = mask.into_shape((h, w, 1))?.into_owned();

            // build image from ndarray
            let mask_im: ImageBuffer<image::Luma<_>, Vec<f32>> =
                ImageBuffer::from_raw(w as u32, h as u32, mask.into_raw_vec())
                    .expect("Faild to create image from ndarray");
            let mut mask_im = image::DynamicImage::from(mask_im);

            // rescale
            let (_, w_mask, h_mask) = ops::scale_wh(image_width, image_height, w as f32, h as f32);
            let mask_original = mask_im.crop(0, 0, w_mask as u32, h_mask as u32);
            let mask_original = mask_original.resize_exact(
                image_width as u32,
                image_height as u32,
                image::imageops::FilterType::Triangle,
            );

            // contours
            let contours: Vec<imageproc::contours::Contour<i32>> =
                imageproc::contours::find_contours(&mask_original.into_luma8());

            for contour in contours.iter() {
                // polygon
                let points: Vec<Point> = contour
                    .points
                    .iter()
                    .map(|p| Point::new(p.x as f32, p.y as f32))
                    .collect();
                let polygon = Polygon::new(&points);
                let mut rect = polygon.find_min_rect();

                // min size filter
                if rect.height() < 3.0 || rect.width() < 3.0 {
                    continue;
                }

                // confs filter
                let confidence = polygon.area() / rect.area();
                if confidence < self.confs[0] {
                    continue;
                }

                // TODO: expand polygon
                let unclip_ratio = 1.5;
                let delta = rect.area() * unclip_ratio / rect.perimeter();

                // save
                let y_bbox = Bbox::new(
                    rect.expand(delta, delta, image_width, image_height),
                    0,
                    confidence,
                    self.names.as_ref().map(|names| names[0].clone()),
                );
                ys_bbox.push(y_bbox);
            }
            let y = Results {
                probs: None,
                bboxes: Some(ys_bbox),
                keypoints: None,
                masks: None,
            };
            ys.push(y);
        }

        Ok(ys)
    }

    pub fn batch(&self) -> isize {
        self.batch.opt
    }

    pub fn width(&self) -> isize {
        self.width.opt
    }

    pub fn height(&self) -> isize {
        self.height.opt
    }
}
