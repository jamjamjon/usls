use crate::{ops, Bbox, DynConf, MinOptMax, Options, OrtEngine, Polygon, Ys};
use anyhow::Result;
use image::{DynamicImage, ImageBuffer};
use ndarray::{Array, Axis, IxDyn};

#[derive(Debug)]
pub struct DB {
    engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    confs: DynConf,
    unclip_ratio: f32,
    binary_thresh: f32,
    min_width: f32,
    min_height: f32,
}

impl DB {
    pub fn new(options: &Options) -> Result<Self> {
        let engine = OrtEngine::new(options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );
        let confs = DynConf::new(&options.confs, 1);
        let unclip_ratio = options.unclip_ratio;
        let binary_thresh = 0.2;
        let min_width = options.min_width.unwrap_or(0.0);
        let min_height = options.min_height.unwrap_or(0.0);
        engine.dry_run()?;

        Ok(Self {
            engine,
            confs,
            height,
            width,
            batch,
            min_width,
            min_height,
            unclip_ratio,
            binary_thresh,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Ys>> {
        let xs_ = ops::letterbox(xs, self.height.opt as u32, self.width.opt as u32, 144.0)?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let xs_ = ops::standardize(xs_, &[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225]);
        let ys = self.engine.run(&[xs_])?;
        let ys = self.postprocess(ys, xs)?;
        Ok(ys)
    }

    pub fn postprocess(&self, xs: Vec<Array<f32, IxDyn>>, xs0: &[DynamicImage]) -> Result<Vec<Ys>> {
        let mut ys = Vec::new();
        for (idx, luma) in xs[0].axis_iter(Axis(0)).enumerate() {
            let mut y_bbox = Vec::new();

            // reshape
            let h = luma.dim()[1];
            let w = luma.dim()[2];
            let luma = luma.into_shape((h, w, 1))?.into_owned();

            // build image from ndarray
            let raw_vec = luma
                .into_raw_vec()
                .iter()
                .map(|x| if x <= &self.binary_thresh { 0.0 } else { *x })
                .collect::<Vec<_>>();
            let mask_im: ImageBuffer<image::Luma<_>, Vec<f32>> =
                ImageBuffer::from_raw(w as u32, h as u32, raw_vec)
                    .expect("Faild to create image from ndarray");
            let mut mask_im = image::DynamicImage::from(mask_im);

            // input image
            let image_width = xs0[idx].width() as f32;
            let image_height = xs0[idx].height() as f32;

            // rescale mask image
            let (ratio, w_mask, h_mask) =
                ops::scale_wh(image_width, image_height, w as f32, h as f32);
            let mask_im = mask_im.crop(0, 0, w_mask as u32, h_mask as u32);
            let mask_im = mask_im.resize_exact(
                image_width as u32,
                image_height as u32,
                image::imageops::FilterType::Triangle,
            );
            let mask_im = mask_im.into_luma8();

            // contours
            let contours: Vec<imageproc::contours::Contour<i32>> =
                imageproc::contours::find_contours_with_threshold(&mask_im, 1);

            // loop
            let mut y_polygons: Vec<Polygon> = Vec::new();
            for contour in contours.iter() {
                if contour.points.len() <= 1 {
                    continue;
                }
                let polygon = Polygon::from_imageproc_points(&contour.points);
                let perimeter = polygon.perimeter();
                let delta = polygon.area() * ratio.round() * self.unclip_ratio / perimeter;
                let polygon = polygon
                    // .simplify(6e-4 * perimeter)
                    .offset(delta, image_width, image_height)
                    .resample(50)
                    .convex_hull();
                let rect = polygon.find_min_rect();
                if rect.height() < self.min_height || rect.width() < self.min_width {
                    continue;
                }
                let confidence = polygon.area() / rect.area();
                if confidence < self.confs[0] {
                    continue;
                }
                let bbox = Bbox::new(rect, 0, confidence, None);
                y_bbox.push(bbox);
                y_polygons.push(polygon);
            }
            ys.push(
                Ys::default()
                    .with_bboxes(&y_bbox)
                    .with_polygons(&y_polygons),
            );
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
