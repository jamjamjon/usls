use crate::{ops, DynConf, Mbr, MinOptMax, Options, OrtEngine, Polygon, Y};
use anyhow::Result;
use image::DynamicImage;
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
        let mut engine = OrtEngine::new(options)?;
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

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Y>> {
        let xs_ = ops::letterbox(xs, self.height.opt as u32, self.width.opt as u32, 144.0)?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let xs_ = ops::standardize(xs_, &[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225]);
        let ys = self.engine.run(&[xs_])?;
        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Vec<Array<f32, IxDyn>>, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        let mut ys = Vec::new();
        for (idx, luma) in xs[0].axis_iter(Axis(0)).enumerate() {
            let mut y_bbox = Vec::new();
            let mut y_polygons: Vec<Polygon> = Vec::new();
            let mut y_mbrs: Vec<Mbr> = Vec::new();

            // reshape
            let h = luma.dim()[1];
            let w = luma.dim()[2];
            let luma = luma.into_shape((h, w, 1))?.into_owned();

            // build image from ndarray
            let v = luma
                .into_raw_vec()
                .iter()
                .map(|x| if x <= &self.binary_thresh { 0.0 } else { *x })
                .collect::<Vec<_>>();
            let mut mask_im =
                ops::build_dyn_image_from_raw(v, self.height() as u32, self.width() as u32);

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
            for contour in contours.iter() {
                if contour.border_type == imageproc::contours::BorderType::Hole
                    && contour.points.len() <= 2
                {
                    continue;
                }
                let mask = Polygon::default().with_points_imageproc(&contour.points);
                let delta = mask.area() * ratio.round() as f64 * self.unclip_ratio as f64
                    / mask.perimeter();
                let mask = mask
                    .unclip(delta, image_width as f64, image_height as f64)
                    .resample(50)
                    // .simplify(6e-4)
                    .convex_hull();
                if let Some(bbox) = mask.bbox() {
                    if bbox.height() < self.min_height || bbox.width() < self.min_width {
                        continue;
                    }
                    let confidence = mask.area() as f32 / bbox.area();
                    if confidence < self.confs[0] {
                        continue;
                    }
                    y_bbox.push(bbox.with_confidence(confidence).with_id(0));

                    if let Some(mbr) = mask.mbr() {
                        y_mbrs.push(mbr.with_confidence(confidence).with_id(0));
                    }
                    y_polygons.push(mask.with_id(0));
                } else {
                    continue;
                }
            }
            ys.push(
                Y::default()
                    .with_bboxes(&y_bbox)
                    .with_polygons(&y_polygons)
                    .with_mbrs(&y_mbrs),
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
