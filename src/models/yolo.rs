use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis};
use rayon::prelude::*;
use regex::Regex;

use crate::{
    Bbox, BoxType, DynConf, Keypoint, Mask, Mbr, MinOptMax, Ops, Options, OrtEngine, Polygon, Prob,
    Vision, YOLOPreds, YOLOTask, YOLOVersion, X, Y,
};

#[derive(Debug)]
pub struct YOLO {
    engine: OrtEngine,
    nc: usize,
    nk: usize, // TODO
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    confs: DynConf,
    kconfs: DynConf,
    iou: f32,
    names: Option<Vec<String>>,
    names_kpt: Option<Vec<String>>,
    apply_nms: bool,
    apply_probs_softmax: bool,
    task: YOLOTask,
    yolo_preds: YOLOPreds,
    version: Option<YOLOVersion>,
}

impl Vision for YOLO {
    type Input = DynamicImage;

    fn new(options: Options) -> Result<Self> {
        let mut engine = OrtEngine::new(&options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );

        // YOLO Task
        let task = options
            .yolo_task
            .or(engine.try_fetch("task").and_then(|x| match x.as_str() {
                "classify" => Some(YOLOTask::Classify),
                "detect" => Some(YOLOTask::Detect),
                "pose" => Some(YOLOTask::Pose),
                "segment" => Some(YOLOTask::Segment),
                "obb" => Some(YOLOTask::Obb),
                s => {
                    println!("YOLO Task: {s:?} is unsupported");
                    None
                }
            }));

        // YOLO Outputs Format
        let (version, yolo_preds) = match options.yolo_version {
            Some(ver) => match &task {
                None => anyhow::bail!("No clear YOLO Task specified for Version: {ver:?}."),
                Some(task) => match task {
                    YOLOTask::Classify => match ver {
                        YOLOVersion::V5 => (Some(ver), YOLOPreds::n_clss().apply_softmax(true)),
                        YOLOVersion::V8 => (Some(ver), YOLOPreds::n_clss()),
                        x => anyhow::bail!("YOLOTask::Classify is unsupported for {x:?}. Try using `.with_yolo_preds()` for customization.")
                    }
                    YOLOTask::Detect => match ver {
                        YOLOVersion::V5 | YOLOVersion::V6 | YOLOVersion::V7 => (Some(ver),YOLOPreds::n_a_cxcywh_confclss()),
                        YOLOVersion::V8 => (Some(ver),YOLOPreds::n_cxcywh_clss_a()),
                        YOLOVersion::V9 => (Some(ver),YOLOPreds::n_cxcywh_clss_a()),
                        YOLOVersion::V10 => (Some(ver),YOLOPreds::n_a_xyxy_confcls().apply_nms(false)),
                    }
                    YOLOTask::Pose => match ver {
                        YOLOVersion::V8 => (Some(ver),YOLOPreds::n_cxcywh_clss_xycs_a()),
                        x => anyhow::bail!("YOLOTask::Pose is unsupported for {x:?}. Try using `.with_yolo_preds()` for customization.")
                    }
                    YOLOTask::Segment => match ver {
                        YOLOVersion::V5 => (Some(ver), YOLOPreds::n_a_cxcywh_confclss_coefs()),
                        YOLOVersion::V8 => (Some(ver), YOLOPreds::n_cxcywh_clss_coefs_a()),
                        x => anyhow::bail!("YOLOTask::Segment is unsupported for {x:?}. Try using `.with_yolo_preds()` for customization.")
                    }
                    YOLOTask::Obb => match ver {
                        YOLOVersion::V8 => (Some(ver), YOLOPreds::n_cxcywh_clss_r_a()),
                        x => anyhow::bail!("YOLOTask::Segment is unsupported for {x:?}. Try using `.with_yolo_preds()` for customization.")
                    }
                }
            }
            None => match options.yolo_preds {
                None => anyhow::bail!("No clear YOLO version or YOLO Format specified."),
                Some(fmt) => (None, fmt)
            }
        };

        let task = task.unwrap_or(yolo_preds.task());
        let (apply_nms, apply_probs_softmax) = (yolo_preds.apply_nms, yolo_preds.apply_softmax);

        // Class names
        let mut names = options.names.or(Self::fetch_names(&engine));
        let nc = match options.nc {
            Some(nc) => {
                match &names {
                    None => names = Some((0..nc).map(|x| x.to_string()).collect::<Vec<String>>()),
                    Some(names) => {
                        assert_eq!(
                            nc,
                            names.len(),
                            "The length of `nc` and `class names` is not equal."
                        );
                    }
                }
                nc
            }
            None => match &names {
                Some(names) => names.len(),
                None => panic!(
                    "Can not parse model without `nc` and `class names`. Try to make it explicit with `options.with_nc(80)`"
                ),
            },
        };

        // Keypoints names
        let names_kpt = options.names2;

        // The number of keypoints
        let nk = engine
            .try_fetch("kpt_shape")
            .map(|kpt_string| {
                let re = Regex::new(r"([0-9]+), ([0-9]+)").unwrap();
                let caps = re.captures(&kpt_string).unwrap();
                caps.get(1).unwrap().as_str().parse::<usize>().unwrap()
            })
            .unwrap_or(0_usize);
        let confs = DynConf::new(&options.confs, nc);
        let kconfs = DynConf::new(&options.kconfs, nk);
        let iou = options.iou.unwrap_or(0.45);

        // Summary
        println!("YOLO Task: {:?}, Version: {:?}", task, version);

        engine.dry_run()?;

        Ok(Self {
            engine,
            confs,
            kconfs,
            iou,
            nc,
            nk,
            height,
            width,
            batch,
            task,
            names,
            names_kpt,
            apply_nms,
            apply_probs_softmax,
            yolo_preds,
            version,
        })
    }

    fn preprocess(&self, xs: &[Self::Input]) -> Result<Vec<X>> {
        let xs_ = match self.task {
            YOLOTask::Classify => {
                X::resize(xs, self.height() as u32, self.width() as u32, "Bilinear")?
                    .normalize(0., 255.)?
                    .nhwc2nchw()?
            }
            _ => X::apply(&[
                Ops::Letterbox(
                    xs,
                    self.height() as u32,
                    self.width() as u32,
                    "CatmullRom",
                    114,
                    "auto",
                    false,
                ),
                Ops::Normalize(0., 255.),
                Ops::Nhwc2nchw,
            ])?,
        };
        Ok(vec![xs_])
    }

    fn inference(&mut self, xs: Vec<X>) -> Result<Vec<X>> {
        self.engine.run(xs)
    }

    fn postprocess(&self, xs: Vec<X>, xs0: &[Self::Input]) -> Result<Vec<Y>> {
        let protos = if xs.len() == 2 { Some(&xs[1]) } else { None };
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, preds)| {
                let image_width = xs0[idx].width() as f32;
                let image_height = xs0[idx].height() as f32;
                let ratio =
                    (self.width() as f32 / image_width).min(self.height() as f32 / image_height);

                match self.task {
                    // Classifcation
                    YOLOTask::Classify => {
                        let y = if self.apply_probs_softmax {
                            let exps = preds.mapv(|x| x.exp());
                            let stds = exps.sum_axis(Axis(0));
                            exps / stds
                        } else {
                            preds.into_owned()
                        };
                        Some(
                            Y::default().with_probs(
                                &Prob::default()
                                    .with_probs(&y.into_raw_vec())
                                    .with_names(self.names.to_owned()),
                            ),
                        )
                    }
                    _ => {
                        // parse preds to get slices for each task
                        let (
                            slice_bboxes,
                            slice_id,
                            slice_clss,
                            slice_kpts,
                            slice_coefs,
                            slice_radians,
                        ) = self.yolo_preds.parse_preds(preds, self.nc);

                        let mut y = Y::default();
                        let (y_bboxes, y_mbrs) =
                            slice_bboxes
                                .axis_iter(Axis(0))
                                .into_par_iter()
                                .enumerate()
                                .filter_map(|(i, bbox)| {
                                    // confidence & class_id
                                    let (class_id, confidence) = match &slice_id {
                                        Some(ids) => (ids[[i, 0]] as _, slice_clss[[i, 0]] as _),
                                        None => {
                                            let (class_id, &confidence) = slice_clss
                                                .slice(s![i, ..])
                                                .into_iter()
                                                .enumerate()
                                                .max_by(|a, b| a.1.total_cmp(b.1))?;
                                            (class_id, confidence)
                                        }
                                    };

                                    // confidence filtering
                                    if confidence < self.confs[class_id] {
                                        return None;
                                    }

                                    // Bboxes
                                    let (cx, cy, x, y, w, h) =
                                        match self.yolo_preds.bbox.as_ref()? {
                                            BoxType::Cxcywh => {
                                                let cx = bbox[0] / ratio;
                                                let cy = bbox[1] / ratio;
                                                let w = bbox[2] / ratio;
                                                let h = bbox[3] / ratio;
                                                let x = (cx - w / 2.).clamp(0.0, image_width);
                                                let y = (cy - h / 2.).clamp(0.0, image_height);
                                                (cx, cy, x, y, w, h)
                                            }
                                            BoxType::Xyxy => {
                                                let x = bbox[0] / ratio;
                                                let y = bbox[1] / ratio;
                                                let x2 = bbox[2] / ratio;
                                                let y2 = bbox[3] / ratio;
                                                let (w, h) = (x2 - x, y2 - y);
                                                let cx = x + w / 2.;
                                                let cy = y + h / 2.;
                                                (cx, cy, x, y, w, h)
                                            }
                                            _ => todo!(),
                                        };

                                    let (y_bbox, y_mbr) =
                                        match &slice_radians {
                                            Some(slice_radians) => {
                                                let radians = slice_radians[[i, 0]];
                                                let (w, h, radians) = if w > h {
                                                    (w, h, radians)
                                                } else {
                                                    (h, w, radians + std::f32::consts::PI / 2.)
                                                };
                                                let radians = radians % std::f32::consts::PI;
                                                (
                                                    None,
                                                    Some(
                                                        Mbr::from_cxcywhr(
                                                            cx as f64,
                                                            cy as f64,
                                                            w as f64,
                                                            h as f64,
                                                            radians as f64,
                                                        )
                                                        .with_confidence(confidence)
                                                        .with_id(class_id as isize)
                                                        .with_name(self.names.as_ref().map(
                                                            |names| names[class_id].to_owned(),
                                                        )),
                                                    ),
                                                )
                                            }
                                            None => (
                                                Some(
                                                    Bbox::default()
                                                        .with_xywh(x, y, w, h)
                                                        .with_confidence(confidence)
                                                        .with_id(class_id as isize)
                                                        .with_id_born(i as isize)
                                                        .with_name(self.names.as_ref().map(
                                                            |names| names[class_id].to_owned(),
                                                        )),
                                                ),
                                                None,
                                            ),
                                        };

                                    Some((y_bbox, y_mbr))
                                })
                                .collect::<(Vec<_>, Vec<_>)>();

                        let y_bboxes: Vec<Bbox> = y_bboxes.into_iter().flatten().collect();
                        let y_mbrs: Vec<Mbr> = y_mbrs.into_iter().flatten().collect();

                        // Mbrs
                        if !y_mbrs.is_empty() {
                            y = y.with_mbrs(&y_mbrs);
                            if self.apply_nms {
                                y = y.apply_nms(self.iou);
                            }
                            return Some(y);
                        }

                        // Bboxes
                        if !y_bboxes.is_empty() {
                            y = y.with_bboxes(&y_bboxes);
                            if self.apply_nms {
                                y = y.apply_nms(self.iou);
                            }
                        }

                        // Pose
                        if let Some(pred_kpts) = slice_kpts {
                            let kpt_step = self.yolo_preds.kpt_step().unwrap_or(3);
                            if let Some(bboxes) = y.bboxes() {
                                let y_kpts = bboxes
                                    .into_par_iter()
                                    .filter_map(|bbox| {
                                        let pred = pred_kpts.slice(s![bbox.id_born(), ..]);
                                        let kpts = (0..self.nk)
                                            .into_par_iter()
                                            .map(|i| {
                                                let kx = pred[kpt_step * i] / ratio;
                                                let ky = pred[kpt_step * i + 1] / ratio;
                                                let kconf = pred[kpt_step * i + 2];
                                                if kconf < self.kconfs[i] {
                                                    Keypoint::default()
                                                } else {
                                                    Keypoint::default()
                                                        .with_id(i as isize)
                                                        .with_confidence(kconf)
                                                        .with_name(
                                                            self.names_kpt
                                                                .as_ref()
                                                                .map(|names| names[i].to_owned()),
                                                        )
                                                        .with_xy(
                                                            kx.max(0.0f32).min(image_width),
                                                            ky.max(0.0f32).min(image_height),
                                                        )
                                                }
                                            })
                                            .collect::<Vec<_>>();
                                        Some(kpts)
                                    })
                                    .collect::<Vec<_>>();
                                y = y.with_keypoints(&y_kpts);
                            }
                        }

                        // Segment
                        if let Some(coefs) = slice_coefs {
                            if let Some(bboxes) = y.bboxes() {
                                let (y_polygons, y_masks) = bboxes
                                    .into_par_iter()
                                    .filter_map(|bbox| {
                                        let coefs = coefs.slice(s![bbox.id_born(), ..]).to_vec();

                                        let proto = protos.as_ref()?.slice(s![idx, .., .., ..]);
                                        let (nm, mh, mw) = proto.dim();

                                        // coefs * proto => mask
                                        let coefs = Array::from_shape_vec((1, nm), coefs).ok()?; // (n, nm)
                                        let proto = proto.into_shape((nm, mh * mw)).ok()?; // (nm, mh * mw)
                                        let mask = coefs.dot(&proto); // (mh, mw, n)

                                        // Mask rescale
                                        let mask = Ops::resize_lumaf32_vec(
                                            &mask.into_raw_vec(),
                                            mw as _,
                                            mh as _,
                                            image_width as _,
                                            image_height as _,
                                            true,
                                            "Bilinear",
                                        )
                                        .ok()?;

                                        let mut mask: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                                            image::ImageBuffer::from_raw(
                                                image_width as _,
                                                image_height as _,
                                                mask,
                                            )?;
                                        let (xmin, ymin, xmax, ymax) =
                                            (bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax());

                                        // Using bbox to crop the mask
                                        for (y, row) in mask.enumerate_rows_mut() {
                                            for (x, _, pixel) in row {
                                                if x < xmin as _
                                                    || x > xmax as _
                                                    || y < ymin as _
                                                    || y > ymax as _
                                                {
                                                    *pixel = image::Luma([0u8]);
                                                }
                                            }
                                        }

                                        // Find contours
                                        let contours: Vec<imageproc::contours::Contour<i32>> =
                                            imageproc::contours::find_contours_with_threshold(
                                                &mask, 0,
                                            );

                                        Some((
                                            contours
                                                .into_par_iter()
                                                .map(|x| {
                                                    Polygon::default()
                                                        .with_id(bbox.id())
                                                        .with_points_imageproc(&x.points)
                                                        .with_name(bbox.name().cloned())
                                                })
                                                .max_by(|x, y| x.area().total_cmp(&y.area()))?,
                                            Mask::default()
                                                .with_mask(mask)
                                                .with_id(bbox.id())
                                                .with_name(bbox.name().cloned()),
                                        ))
                                    })
                                    .collect::<(Vec<_>, Vec<_>)>();

                                y = y.with_polygons(&y_polygons).with_masks(&y_masks);
                            }
                        }

                        Some(y)
                    }
                }
            })
            .collect();

        Ok(ys)
    }
}

impl YOLO {
    pub fn batch(&self) -> isize {
        self.batch.opt
    }

    pub fn width(&self) -> isize {
        self.width.opt
    }

    pub fn height(&self) -> isize {
        self.height.opt
    }

    pub fn version(&self) -> Option<&YOLOVersion> {
        self.version.as_ref()
    }

    pub fn task(&self) -> &YOLOTask {
        &self.task
    }

    // pub fn yolo_preds(&self) -> &YOLOPreds {
    //     &self.yolo_preds
    // }

    fn fetch_names(engine: &OrtEngine) -> Option<Vec<String>> {
        // fetch class names from onnx metadata
        // String format: `{0: 'person', 1: 'bicycle', 2: 'sports ball', ..., 27: "yellow_lady's_slipper"}`
        engine.try_fetch("names").map(|names| {
            let re = Regex::new(r#"(['"])([-()\w '"]+)(['"])"#).unwrap();
            let mut names_ = vec![];
            for (_, [_, name, _]) in re.captures_iter(&names).map(|x| x.extract()) {
                names_.push(name.to_string());
            }
            names_
        })
    }
}
