use anyhow::Result;
use clap::ValueEnum;
use image::DynamicImage;
use ndarray::{s, Array, Axis};
use rayon::prelude::*;
use regex::Regex;

use crate::{
    Bbox, DynConf, Keypoint, Mbr, MinOptMax, Ops, Options, OrtEngine, Polygon, Prob, Vision, X, Y,
};

const CXYWH_OFFSET: usize = 4;
const KPT_STEP: usize = 3;

#[derive(Debug, Clone, ValueEnum)]
pub enum YOLOTask {
    Classify,
    Detect,
    Pose,
    Segment,
    Obb,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum YOLOVersion {
    V5,
    V8,
    V9,
    V10,
    Customized,
}

#[derive(Debug)]
pub struct YOLO {
    engine: OrtEngine,
    nc: usize,
    nk: usize,
    nm: usize,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    task: YOLOTask,
    version: YOLOVersion,
    confs: DynConf,
    kconfs: DynConf,
    iou: f32,
    names: Option<Vec<String>>,
    names_kpt: Option<Vec<String>>,
    apply_nms: bool,
    anchors_first: bool,
    conf_independent: bool,
    apply_probs_softmax: bool,
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
        let task = match options.yolo_task {
            Some(task) => task,
            None => match engine.try_fetch("task") {
                None => {
                    println!("No clear YOLO task specified, using default: Detect");
                    YOLOTask::Detect
                }
                Some(x) => match x.as_str() {
                    "classify" => YOLOTask::Classify,
                    "detect" => YOLOTask::Detect,
                    "pose" => YOLOTask::Pose,
                    "segment" => YOLOTask::Segment,
                    "obb" => YOLOTask::Obb,
                    x => todo!("YOLO Task: {x:?} is not supported"),
                },
            },
        };
        let version = match options.yolo_version {
            None => {
                println!("No clear YOLO version specified, using default: YOLOv8");
                YOLOVersion::V8
            }
            Some(x) => x,
        };

        // output format
        let (anchors_first, conf_independent, apply_nms, apply_probs_softmax) = match version {
            YOLOVersion::V5 => (true, true, true, true),
            YOLOVersion::V8 | YOLOVersion::V9 => (false, false, true, false),
            YOLOVersion::V10 => (true, false, false, false),
            YOLOVersion::Customized => (
                options.anchors_first,
                options.conf_independent,
                options.apply_nms,
                options.apply_probs_softmax,
            ),
        };

        // try from custom class names, and then model metadata
        let mut names = options.names.or(Self::fetch_names(&engine));
        let nc = match options.nc {
            Some(nc) => {
                match &names {
                    None => names = Some((0..nc).map(|x| x.to_string()).collect::<Vec<String>>()),
                    Some(names) => {
                        assert_eq!(
                            nc,
                            names.len(),
                            "the length of `nc` and `class names` is not equal."
                        );
                    }
                }
                nc
            }
            None => match &names {
                Some(names) => names.len(),
                None => panic!(
                    "Can not parse model without `nc` and `class names`. Try to make it explicit."
                ),
            },
        };

        let names_kpt = options.names2.or(None);

        // try from model metadata
        let nk = engine
            .try_fetch("kpt_shape")
            .map(|kpt_string| {
                let re = Regex::new(r"([0-9]+), ([0-9]+)").unwrap();
                let caps = re.captures(&kpt_string).unwrap();
                caps.get(1).unwrap().as_str().parse::<usize>().unwrap()
            })
            .unwrap_or(0_usize);
        let nm = if let YOLOTask::Segment = task {
            engine.oshapes()[1][1] as usize
        } else {
            0_usize
        };
        let confs = DynConf::new(&options.confs, nc);
        let kconfs = DynConf::new(&options.kconfs, nk);
        engine.dry_run()?;

        Ok(Self {
            engine,
            confs,
            kconfs,
            iou: options.iou,
            nc,
            nk,
            nm,
            height,
            width,
            batch,
            task,
            version,
            names,
            names_kpt,
            anchors_first,
            conf_independent,
            apply_nms,
            apply_probs_softmax,
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

                match self.task {
                    // Detection
                    YOLOTask::Detect | YOLOTask::Segment | YOLOTask::Pose => {
                        let ratio = (self.width() as f32 / image_width)
                            .min(self.height() as f32 / image_height);
                        let y_bboxes = preds
                            .axis_iter(if self.anchors_first { Axis(0) } else { Axis(1) })
                            .into_par_iter()
                            .enumerate()
                            .filter_map(|(i, pred)| match self.version {
                                YOLOVersion::V10 => {
                                    let class_id = pred[CXYWH_OFFSET + 1] as usize;
                                    let confidence = pred[CXYWH_OFFSET];
                                    if confidence < self.confs[class_id] {
                                        None
                                    } else {
                                        let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                                        let x = bbox[0] / ratio;
                                        let y = bbox[1] / ratio;
                                        let x2 = bbox[2] / ratio;
                                        let y2 = bbox[3] / ratio;
                                        let w = x2 - x;
                                        let h = y2 - y;
                                        let y_bbox = Bbox::default()
                                            .with_xywh(x, y, w, h)
                                            .with_confidence(confidence)
                                            .with_id(class_id as _)
                                            .with_id_born(i as _)
                                            .with_name(
                                                self.names
                                                    .as_ref()
                                                    .map(|names| names[class_id].to_owned()),
                                            );
                                        Some(y_bbox)
                                    }
                                }
                                _ => {
                                    let (conf_, clss) = if self.conf_independent {
                                        (
                                            pred[CXYWH_OFFSET],
                                            pred.slice(s![
                                                CXYWH_OFFSET + 1..CXYWH_OFFSET + self.nc + 1
                                            ]),
                                        )
                                    } else {
                                        (1.0, pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc]))
                                    };
                                    let (id, &confidence) = clss
                                        .into_iter()
                                        .enumerate()
                                        .max_by(|a, b| a.1.total_cmp(b.1))?;
                                    let confidence = confidence * conf_;
                                    if confidence < self.confs[id] {
                                        None
                                    } else {
                                        let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                                        let cx = bbox[0] / ratio;
                                        let cy = bbox[1] / ratio;
                                        let w = bbox[2] / ratio;
                                        let h = bbox[3] / ratio;
                                        let x = cx - w / 2.;
                                        let y = cy - h / 2.;
                                        let x = x.max(0.0).min(image_width);
                                        let y = y.max(0.0).min(image_height);
                                        let y_bbox = Bbox::default()
                                            .with_xywh(x, y, w, h)
                                            .with_confidence(confidence)
                                            .with_id(id as isize)
                                            .with_id_born(i as isize)
                                            .with_name(
                                                self.names
                                                    .as_ref()
                                                    .map(|names| names[id].to_owned()),
                                            );
                                        Some(y_bbox)
                                    }
                                }
                            })
                            .collect::<Vec<_>>();

                        // NMS
                        let mut y = Y::default().with_bboxes(&y_bboxes);
                        if self.apply_nms {
                            y = y.apply_bboxes_nms(self.iou);
                        }

                        // Pose
                        if let YOLOTask::Pose = self.task {
                            if let Some(bboxes) = y.bboxes() {
                                let y_kpts = bboxes
                                    .into_par_iter()
                                    .filter_map(|bbox| {
                                        let pred = if self.anchors_first {
                                            preds.slice(s![
                                                bbox.id_born(),
                                                preds.shape()[1] - KPT_STEP * self.nk..,
                                            ])
                                        } else {
                                            preds.slice(s![
                                                preds.shape()[0] - KPT_STEP * self.nk..,
                                                bbox.id_born(),
                                            ])
                                        };

                                        let kpts = (0..self.nk)
                                            .into_par_iter()
                                            .map(|i| {
                                                let kx = pred[KPT_STEP * i] / ratio;
                                                let ky = pred[KPT_STEP * i + 1] / ratio;
                                                let kconf = pred[KPT_STEP * i + 2];
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
                        if let YOLOTask::Segment = self.task {
                            if let Some(bboxes) = y.bboxes() {
                                let y_polygons = bboxes
                                    .into_par_iter()
                                    .filter_map(|bbox| {
                                        let coefs = if self.anchors_first {
                                            preds
                                                .slice(s![
                                                    bbox.id_born(),
                                                    preds.shape()[1] - self.nm..
                                                ])
                                                .to_vec()
                                        } else {
                                            preds
                                                .slice(s![
                                                    preds.shape()[0] - self.nm..,
                                                    bbox.id_born()
                                                ])
                                                .to_vec()
                                        };
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

                                        contours
                                            .into_par_iter()
                                            .map(|x| {
                                                Polygon::default()
                                                    .with_id(bbox.id())
                                                    .with_points_imageproc(&x.points)
                                                    .with_name(bbox.name().cloned())
                                            })
                                            .max_by(|x, y| x.area().total_cmp(&y.area()))
                                    })
                                    .collect::<Vec<_>>();
                                y = y.with_polygons(&y_polygons);
                            }
                        }
                        Some(y)
                    }
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
                                Prob::default()
                                    .with_probs(&y.into_raw_vec())
                                    .with_names(self.names.to_owned()),
                            ),
                        )
                    }
                    YOLOTask::Obb => {
                        let ratio = (self.width() as f32 / image_width)
                            .min(self.height() as f32 / image_height);

                        let y_mbrs = preds
                            .axis_iter(if self.anchors_first { Axis(0) } else { Axis(1) })
                            .into_par_iter()
                            .filter_map(|pred| {
                                {
                                    // xywhclsr
                                    let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc]);
                                    let radians = pred[pred.len() - 1];
                                    let (id, &confidence) = clss
                                        .into_iter()
                                        .enumerate()
                                        .max_by(|a, b| a.1.total_cmp(b.1))?;
                                    if confidence < self.confs[id] {
                                        None
                                    } else {
                                        let xywh = pred.slice(s![0..CXYWH_OFFSET]);
                                        let cx = xywh[0] / ratio;
                                        let cy = xywh[1] / ratio;
                                        let w = xywh[2] / ratio;
                                        let h = xywh[3] / ratio;
                                        let (w, h, radians) = if w > h {
                                            (w, h, radians)
                                        } else {
                                            (h, w, radians + std::f32::consts::PI / 2.)
                                        };
                                        let radians = radians % std::f32::consts::PI;
                                        Some(
                                            Mbr::from_cxcywhr(
                                                cx as f64,
                                                cy as f64,
                                                w as f64,
                                                h as f64,
                                                radians as f64,
                                            )
                                            .with_confidence(confidence)
                                            .with_id(id as isize)
                                            .with_name(
                                                self.names
                                                    .as_ref()
                                                    .map(|names| names[id].to_owned()),
                                            ),
                                        )
                                    }
                                }
                            })
                            .collect::<Vec<_>>();

                        Some(Y::default().with_mbrs(&y_mbrs).apply_mbrs_nms(self.iou))
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
