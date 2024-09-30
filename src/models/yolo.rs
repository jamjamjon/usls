use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis};
use rayon::prelude::*;
use regex::Regex;

use crate::{
    Bbox, BoxType, DynConf, Keypoint, Mask, Mbr, MinOptMax, Ops, Options, OrtEngine, Polygon, Prob,
    Vision, Xs, YOLOPreds, YOLOTask, YOLOVersion, X, Y,
};

#[derive(Debug)]
pub struct YOLO {
    engine: OrtEngine,
    nc: usize,
    nk: usize,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    confs: DynConf,
    kconfs: DynConf,
    iou: f32,
    names: Vec<String>,
    names_kpt: Vec<String>,
    task: YOLOTask,
    layout: YOLOPreds,
    find_contours: bool,
    version: Option<YOLOVersion>,
}

impl Vision for YOLO {
    type Input = DynamicImage;

    fn new(options: Options) -> Result<Self> {
        let span = tracing::span!(tracing::Level::INFO, "YOLO-new");
        let _guard = span.enter();

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
                    tracing::error!("YOLO Task: {s:?} is unsupported");
                    None
                }
            }));

        // YOLO Outputs Format
        let (version, layout) = match options.yolo_version {
            Some(ver) => match &task {
                None => anyhow::bail!("No clear YOLO Task specified for Version: {ver:?}."),
                Some(task) => match task {
                    YOLOTask::Classify => match ver {
                        YOLOVersion::V5 => (Some(ver), YOLOPreds::n_clss().apply_softmax(true)),
                        YOLOVersion::V8 | YOLOVersion::V11 => (Some(ver), YOLOPreds::n_clss()),
                        x => anyhow::bail!("YOLOTask::Classify is unsupported for {x:?}. Try using `.with_yolo_preds()` for customization.")
                    }
                    YOLOTask::Detect => match ver {
                        YOLOVersion::V5 | YOLOVersion::V6 | YOLOVersion::V7 => (Some(ver), YOLOPreds::n_a_cxcywh_confclss()),
                        YOLOVersion::V8 | YOLOVersion::V9 | YOLOVersion::V11 => (Some(ver), YOLOPreds::n_cxcywh_clss_a()),
                        YOLOVersion::V10 => (Some(ver), YOLOPreds::n_a_xyxy_confcls().apply_nms(false)),
                        YOLOVersion::RTDETR => (Some(ver), YOLOPreds::n_a_cxcywh_clss_n().apply_nms(false)),
                    }
                    YOLOTask::Pose => match ver {
                        YOLOVersion::V8 | YOLOVersion::V11 => (Some(ver), YOLOPreds::n_cxcywh_clss_xycs_a()),
                        x => anyhow::bail!("YOLOTask::Pose is unsupported for {x:?}. Try using `.with_yolo_preds()` for customization.")
                    }
                    YOLOTask::Segment => match ver {
                        YOLOVersion::V5 => (Some(ver), YOLOPreds::n_a_cxcywh_confclss_coefs()),
                        YOLOVersion::V8 | YOLOVersion::V11 => (Some(ver), YOLOPreds::n_cxcywh_clss_coefs_a()),
                        x => anyhow::bail!("YOLOTask::Segment is unsupported for {x:?}. Try using `.with_yolo_preds()` for customization.")
                    }
                    YOLOTask::Obb => match ver {
                        YOLOVersion::V8 | YOLOVersion::V11 => (Some(ver), YOLOPreds::n_cxcywh_clss_r_a()),
                        x => anyhow::bail!("YOLOTask::Segment is unsupported for {x:?}. Try using `.with_yolo_preds()` for customization.")
                    }
                }
            }
            None => match options.yolo_preds {
                None => anyhow::bail!("No clear YOLO version or YOLO Format specified."),
                Some(fmt) => (None, fmt)
            }
        };

        let task = task.unwrap_or(layout.task());

        // Class names: user-defined.or(parsed)
        let names_parsed = Self::fetch_names(&engine);
        let names = match names_parsed {
            Some(names_parsed) => match options.names {
                Some(names) => {
                    if names.len() == names_parsed.len() {
                        Some(names)
                    } else {
                        anyhow::bail!(
                            "The lengths of parsed class names: {} and user-defined class names: {} do not match.",
                            names_parsed.len(),
                            names.len(),
                        );
                    }
                }
                None => Some(names_parsed),
            },
            None => options.names,
        };

        // nc: names.len().or(options.nc)
        let nc = match &names {
            Some(names) => names.len(),
            None => match options.nc {
                Some(nc) => nc,
                None => anyhow::bail!(
                    "Unable to obtain the number of classes. Please specify them explicitly using `options.with_nc(usize)` or `options.with_names(&[&str])`."
                ),
            }
        };

        // Class names
        let names = match names {
            None => Self::n2s(nc),
            Some(names) => names,
        };

        // Keypoint names & nk
        let (nk, names_kpt) = match Self::fetch_kpts(&engine) {
            None => (0, vec![]),
            Some(nk) => match options.names2 {
                Some(names) => {
                    if names.len() == nk {
                        (nk, names)
                    } else {
                        anyhow::bail!(
                            "The lengths of user-defined keypoint names: {} and nk: {} do not match.",
                            names.len(),
                            nk,
                        );
                    }
                }
                None => (nk, Self::n2s(nk)),
            },
        };

        // Confs & Iou
        let confs = DynConf::new(&options.confs, nc);
        let kconfs = DynConf::new(&options.kconfs, nk);
        let iou = options.iou.unwrap_or(0.45);

        // Summary
        tracing::info!("YOLO Task: {:?}, Version: {:?}", task, version);

        // dry run
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
            layout,
            version,
            find_contours: options.find_contours,
        })
    }

    fn preprocess(&self, xs: &[Self::Input]) -> Result<Xs> {
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
        Ok(Xs::from(xs_))
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    fn postprocess(&self, xs: Xs, xs0: &[Self::Input]) -> Result<Vec<Y>> {
        let protos = if xs.len() == 2 { Some(&xs[1]) } else { None };
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, preds)| {
                let mut y = Y::default();

                // parse preditions
                let (
                    slice_bboxes,
                    slice_id,
                    slice_clss,
                    slice_confs,
                    slice_kpts,
                    slice_coefs,
                    slice_radians,
                ) = self.layout.parse_preds(preds, self.nc);

                // Classifcation
                if let YOLOTask::Classify = self.task {
                    let x = if self.layout.apply_softmax {
                        let exps = slice_clss.mapv(|x| x.exp());
                        let stds = exps.sum_axis(Axis(0));
                        exps / stds
                    } else {
                        slice_clss.into_owned()
                    };
                    let mut probs = Prob::default().with_probs(&x.into_raw_vec_and_offset().0);
                    probs = probs
                        .with_names(&self.names.iter().map(|x| x.as_str()).collect::<Vec<_>>());

                    return Some(y.with_probs(&probs));
                }

                let image_width = xs0[idx].width() as f32;
                let image_height = xs0[idx].height() as f32;
                let ratio =
                    (self.width() as f32 / image_width).min(self.height() as f32 / image_height);

                // Other tasks
                let (y_bboxes, y_mbrs) = slice_bboxes?
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

                                match &slice_confs {
                                    None => (class_id, confidence),
                                    Some(slice_confs) => {
                                        (class_id, confidence * slice_confs[[i, 0]])
                                    }
                                }
                            }
                        };

                        // filtering
                        if confidence < self.confs[class_id] {
                            return None;
                        }

                        // Bboxes
                        let bbox = bbox.mapv(|x| x / ratio);
                        let bbox = if self.layout.is_bbox_normalized {
                            (
                                bbox[0] * self.width() as f32,
                                bbox[1] * self.height() as f32,
                                bbox[2] * self.width() as f32,
                                bbox[3] * self.height() as f32,
                            )
                        } else {
                            (bbox[0], bbox[1], bbox[2], bbox[3])
                        };
                        let (cx, cy, x, y, w, h) = match self.layout.box_type()? {
                            BoxType::Cxcywh => {
                                let (cx, cy, w, h) = bbox;
                                let x = (cx - w / 2.).max(0.);
                                let y = (cy - h / 2.).max(0.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Xyxy => {
                                let (x, y, x2, y2) = bbox;
                                let (w, h) = (x2 - x, y2 - y);
                                let (cx, cy) = ((x + x2) / 2., (y + y2) / 2.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Xywh => {
                                let (x, y, w, h) = bbox;
                                let (cx, cy) = (x + w / 2., y + h / 2.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Cxcyxy => {
                                let (cx, cy, x2, y2) = bbox;
                                let (w, h) = ((x2 - cx) * 2., (y2 - cy) * 2.);
                                let x = (x2 - w).max(0.);
                                let y = (y2 - h).max(0.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::XyCxcy => {
                                let (x, y, cx, cy) = bbox;
                                let (w, h) = ((cx - x) * 2., (cy - y) * 2.);
                                (cx, cy, x, y, w, h)
                            }
                        };

                        let (y_bbox, y_mbr) = match &slice_radians {
                            Some(slice_radians) => {
                                let radians = slice_radians[[i, 0]];
                                let (w, h, radians) = if w > h {
                                    (w, h, radians)
                                } else {
                                    (h, w, radians + std::f32::consts::PI / 2.)
                                };
                                let radians = radians % std::f32::consts::PI;

                                let mut mbr = Mbr::from_cxcywhr(
                                    cx as f64,
                                    cy as f64,
                                    w as f64,
                                    h as f64,
                                    radians as f64,
                                )
                                .with_confidence(confidence)
                                .with_id(class_id as isize);
                                mbr = mbr.with_name(&self.names[class_id]);

                                (None, Some(mbr))
                            }
                            None => {
                                let mut bbox = Bbox::default()
                                    .with_xywh(x, y, w, h)
                                    .with_confidence(confidence)
                                    .with_id(class_id as isize)
                                    .with_id_born(i as isize);
                                bbox = bbox.with_name(&self.names[class_id]);

                                (Some(bbox), None)
                            }
                        };

                        Some((y_bbox, y_mbr))
                    })
                    .collect::<(Vec<_>, Vec<_>)>();

                let y_bboxes: Vec<Bbox> = y_bboxes.into_iter().flatten().collect();
                let y_mbrs: Vec<Mbr> = y_mbrs.into_iter().flatten().collect();

                // Mbrs
                if !y_mbrs.is_empty() {
                    y = y.with_mbrs(&y_mbrs);
                    if self.layout.apply_nms {
                        y = y.apply_nms(self.iou);
                    }
                    return Some(y);
                }

                // Bboxes
                if !y_bboxes.is_empty() {
                    y = y.with_bboxes(&y_bboxes);
                    if self.layout.apply_nms {
                        y = y.apply_nms(self.iou);
                    }
                }

                // Pose
                if let Some(pred_kpts) = slice_kpts {
                    let kpt_step = self.layout.kpt_step().unwrap_or(3);
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
                                            let mut kpt = Keypoint::default()
                                                .with_id(i as isize)
                                                .with_confidence(kconf)
                                                .with_xy(
                                                    kx.max(0.0f32).min(image_width),
                                                    ky.max(0.0f32).min(image_height),
                                                );

                                            kpt = kpt.with_name(&self.names_kpt[i]);
                                            kpt
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
                                let proto = proto.to_shape((nm, mh * mw)).ok()?; // (nm, mh * mw)
                                let mask = coefs.dot(&proto); // (mh, mw, n)

                                // Mask rescale
                                let mask = Ops::resize_lumaf32_u8(
                                    &mask.into_raw_vec_and_offset().0,
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
                                let polygons = if self.find_contours {
                                    let contours: Vec<imageproc::contours::Contour<i32>> =
                                        imageproc::contours::find_contours_with_threshold(&mask, 0);
                                    contours
                                        .into_par_iter()
                                        .map(|x| {
                                            let mut polygon = Polygon::default()
                                                .with_id(bbox.id())
                                                .with_points_imageproc(&x.points)
                                                .verify();
                                            if let Some(name) = bbox.name() {
                                                polygon = polygon.with_name(name);
                                            }
                                            polygon
                                        })
                                        .max_by(|x, y| x.area().total_cmp(&y.area()))?
                                } else {
                                    Polygon::default()
                                };

                                let mut mask = Mask::default().with_mask(mask).with_id(bbox.id());
                                if let Some(name) = bbox.name() {
                                    mask = mask.with_name(name);
                                }

                                Some((polygons, mask))
                            })
                            .collect::<(Vec<_>, Vec<_>)>();

                        if !y_polygons.is_empty() {
                            y = y.with_polygons(&y_polygons);
                        }
                        if !y_masks.is_empty() {
                            y = y.with_masks(&y_masks);
                        }
                    }
                }

                Some(y)
            })
            .collect();

        Ok(ys)
    }
}

impl YOLO {
    pub fn batch(&self) -> usize {
        self.batch.opt()
    }

    pub fn width(&self) -> usize {
        self.width.opt()
    }

    pub fn height(&self) -> usize {
        self.height.opt()
    }

    pub fn version(&self) -> Option<&YOLOVersion> {
        self.version.as_ref()
    }

    pub fn task(&self) -> &YOLOTask {
        &self.task
    }

    pub fn layout(&self) -> &YOLOPreds {
        &self.layout
    }

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

    fn fetch_kpts(engine: &OrtEngine) -> Option<usize> {
        engine.try_fetch("kpt_shape").map(|s| {
            let re = Regex::new(r"([0-9]+), ([0-9]+)").unwrap();
            let caps = re.captures(&s).unwrap();
            caps.get(1).unwrap().as_str().parse::<usize>().unwrap()
        })
    }

    fn n2s(n: usize) -> Vec<String> {
        (0..n).map(|x| format!("# {}", x)).collect::<Vec<String>>()
    }
}
