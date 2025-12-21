use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array, Axis};
use rayon::prelude::*;
use regex::Regex;
use tracing::{error, info};

use crate::{
    elapsed_module, ort_inputs, Config, DynConf, Engine, FromConfig, Hbb, Image, ImageProcessor,
    Keypoint, Mask, Module, NmsOps, Obb, Ops, Prob, Task, Version, X, Y,
};

use super::{BoxType, YOLOPredsFormat};

/// YOLO (You Only Look Once) object detection model.
///
/// A versatile deep learning model that can perform multiple computer vision tasks including:
/// - Object Detection
/// - Instance Segmentation
/// - Keypoint Detection
/// - Image Classification
/// - Oriented Object Detection
#[derive(Debug, Builder)]
pub struct YOLO {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,

    task: Task,
    version: Option<Version>,
    spec: String,
    layout: YOLOPredsFormat,
    names: Vec<String>,
    names_kpt: Vec<String>,
    nc: usize,
    nk: usize,
    confs: DynConf,
    kconfs: DynConf,
    iou: f32,
    topk: usize,
    processor: ImageProcessor,
    classes_excluded: Vec<usize>,
    classes_retained: Vec<usize>,
    pub(crate) embedding: Option<X>,
}

impl TryFrom<Config> for YOLO {
    type Error = anyhow::Error;

    fn try_from(config: Config) -> Result<Self, Self::Error> {
        Self::new(config)
    }
}

impl YOLO {
    /// Creates a new YOLO model instance from the provided configuration.
    pub fn new(mut config: Config) -> Result<Self> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width, spec) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
            engine.spec().to_owned(),
        );

        let task: Option<Task> = match config.task() {
            Some(task) => Some(task.clone()),
            None => match engine.try_fetch("task") {
                Some(x) => match x.as_str() {
                    "classify" => Some(Task::ImageClassification),
                    "detect" => Some(Task::ObjectDetection),
                    "pose" => Some(Task::KeypointsDetection),
                    "segment" => Some(Task::InstanceSegmentation),
                    "obb" => Some(Task::OrientedObjectDetection),
                    x => {
                        error!("Unsupported YOLO Task: {}", x);
                        None
                    }
                },
                None => None,
            },
        };

        // Task & layout
        let version = config.version.as_ref();
        let (layout, task) = match &config.inference.yolo_preds_format {
            // customized
            Some(layout) => {
                // check task
                let task_parsed = layout.task();
                let task = match task {
                    Some(task) => {
                        if task_parsed != task {
                            anyhow::bail!(
                                "Task specified: {:?} is inconsistent with parsed from yolo_preds_format: {:?}",
                                task,
                                task_parsed
                            );
                        }
                        task_parsed
                    }
                    None => task_parsed,
                };

                (layout.clone(), task)
            }

            // version + task
            None => match (task, version) {
                (Some(task), Some(version)) => {
                    let layout = match (task.clone(), version) {
                        (Task::ImageClassification, Version(5, 0, _)) => {
                            YOLOPredsFormat::n_clss().apply_softmax(true)
                        }
                        (
                            Task::ImageClassification,
                            Version(8, 0, _) | Version(11, 0, _) | Version(12, 0, _),
                        ) => YOLOPredsFormat::n_clss(),
                        (
                            Task::ObjectDetection,
                            Version(5, 0, _) | Version(6, 0, _) | Version(7, 0, _),
                        ) => YOLOPredsFormat::n_a_cxcywh_confclss(),
                        (
                            Task::ObjectDetection,
                            Version(8, 0, _)
                            | Version(9, 0, _)
                            | Version(11, 0, _)
                            | Version(12, 0, _)
                            | Version(13, 0, _),
                        ) => YOLOPredsFormat::n_cxcywh_clss_a(),
                        (Task::ObjectDetection, Version(10, 0, _)) => {
                            YOLOPredsFormat::n_a_xyxy_confcls().apply_nms(false)
                        }
                        (Task::KeypointsDetection, Version(8, 0, _) | Version(11, 0, _)) => {
                            YOLOPredsFormat::n_cxcywh_clss_xycs_a()
                        }
                        (Task::InstanceSegmentation, Version(5, 0, _)) => {
                            YOLOPredsFormat::n_a_cxcywh_confclss_coefs()
                        }
                        (
                            Task::InstanceSegmentation,
                            Version(8, 0, _)
                            | Version(9, 0, _)
                            | Version(11, 0, _)
                            | Version(12, 0, _),
                        ) => YOLOPredsFormat::n_cxcywh_clss_coefs_a(),
                        (Task::OrientedObjectDetection, Version(8, 0, _) | Version(11, 0, _)) => {
                            YOLOPredsFormat::n_cxcywh_clss_r_a()
                        }
                        (task, version) => {
                            anyhow::bail!("Task: {:?} is unsupported for Version: {:?}. Try using `.with_yolo_preds()` for customization.", task, version)
                        }
                    };

                    (layout, task)
                }
                (None, Some(version)) => {
                    let layout = match version {
                        // single task, no need to specified task
                        Version(6, 0, _) | Version(7, 0, _) => {
                            YOLOPredsFormat::n_a_cxcywh_confclss()
                        }
                        Version(10, 0, _) => YOLOPredsFormat::n_a_xyxy_confcls().apply_nms(false),
                        _ => {
                            anyhow::bail!(
                                "No clear YOLO Task specified for Version: {:?}.",
                                version
                            )
                        }
                    };

                    (layout, Task::ObjectDetection)
                }
                (Some(task), None) => {
                    anyhow::bail!("No clear YOLO Version specified for Task: {:?}.", task)
                }
                (None, None) => {
                    anyhow::bail!("No clear YOLO Task and Version specified.")
                }
            },
        };

        // Class names
        let names_parsed = Self::fetch_names_from_onnx(&engine);
        let names_customized = config.inference.class_names.clone();
        let names: Vec<_> = match (names_parsed, names_customized.is_empty()) {
            (None, true) => vec![],
            (None, false) => names_customized,
            (Some(names_parsed), true) => names_parsed,
            (Some(names_parsed), false) => {
                if names_parsed.len() == names_customized.len() {
                    names_customized // prioritize user-defined
                } else {
                    anyhow::bail!(
                        "The lengths of parsed class names: {} and user-defined class names: {} do not match.",
                        names_parsed.len(),
                        names_customized.len(),
                    );
                }
            }
        };

        // Class names & Number of class
        let nc = match config.inference.num_classes {
            None => names.len(),
            Some(n) => {
                if names.is_empty() {
                    n
                } else if names.len() != n {
                    anyhow::bail!(
                        "The lengths of class names: {} and user-defined num_classes: {} do not match.",
                        names.len(),
                        n,
                    )
                } else {
                    n
                }
            }
        };
        if nc == 0 && names.is_empty() {
            anyhow::bail!(
                    "Neither class names nor the number of classes were specified. \
                    \nConsider specify them with `Config::default().with_nc()` or `Config::default().with_class_names()`"
                );
        }

        // Keypoint names & Number of keypoints
        let names_kpt = config.inference.keypoint_names.clone();
        let nk = if let Task::KeypointsDetection = task {
            match (names_kpt.is_empty(),  Self::fetch_nk_from_onnx(&engine).or(config.inference.num_keypoints)) {
                (false, Some(nk)) => {
                    if names_kpt.len() != nk {
                        anyhow::bail!(
                            "The lengths of user-defined keypoint class names: {} and num_keypoints: {} do not match.",
                            names_kpt.len(),
                            nk,
                        );
                    }
                  nk
                },
                (false, None) => names_kpt.len(),
                (true, Some(nk)) => nk,
                (true, None) => anyhow::bail!(
                    "Neither keypoint names nor the number of keypoints were specified when doing `KeypointsDetection` task. \
                    \nConsider specify them with `Config::default().with_nk()` or `Config::default().with_keypoint_names()`"
                ),
            }
        } else {
            0
        };

        // Attributes - clone inference data before moving image_processor
        let topk = config.inference.topk.unwrap_or(5);
        let class_confs = config.inference.class_confs.clone();
        let keypoint_confs = config.inference.keypoint_confs.clone();
        let confs = DynConf::new_or_default(&class_confs, nc);
        let kconfs = DynConf::new_or_default(&keypoint_confs, nk);
        let iou = config.inference.iou.unwrap_or(0.45);
        let classes_excluded = config.inference.classes_excluded.clone();
        let classes_retained = config.inference.classes_retained.clone();
        let mut info = format!(
            "YOLO Version: {}, Task: {:?}, Category Count: {}, Keypoint Count: {}, TopK: {}",
            version.map_or("Unknown".into(), |x| x.to_string()),
            task,
            nc,
            nk,
            topk,
        );
        if !classes_excluded.is_empty() {
            info = format!("{}, classes_excluded: {:?}", info, classes_excluded);
        }
        if !classes_retained.is_empty() {
            info = format!("{}, classes_retained: {:?}", info, classes_retained);
        }

        // Move image_processor out before using it
        let processor = {
            let image_processor_config = config.image_processor;
            ImageProcessor::from_config(image_processor_config)?
                .with_image_width(width as _)
                .with_image_height(height as _)
        };

        info!("{}", info);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            task,
            version: version.copied(),
            spec,
            layout,
            names,
            names_kpt,
            nc,
            nk,
            confs,
            kconfs,
            iou,
            topk,
            processor,
            classes_excluded,
            classes_retained,
            embedding: None,
        })
    }

    /// Performs the complete inference pipeline on a batch of images.
    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let processed = elapsed_module!("YOLO", "preprocess", self.processor.process(xs)?);
        // Convert to X for .view() support (YOLO uses multi-input with embedding)
        let x = processed.as_host()?;
        let (preds, protos) = elapsed_module!("YOLO", "inference", {
            let ys = if let Some(ref embedding) = self.embedding {
                self.engine.run(ort_inputs![x.view(), embedding.view()]?)?
            } else {
                self.engine.run(ort_inputs![x.view()]?)?
            };
            let preds = ys.get::<f32>(0)?.to_owned();
            let protos = ys.try_get::<f32>(1).map(|p| p.to_owned());
            Ok::<_, anyhow::Error>((preds, protos))
        })?;
        elapsed_module!("YOLO", "postprocess", self.postprocess(preds, protos))
    }

    /// Post-processes model outputs to generate final predictions.
    pub(crate) fn postprocess(&self, preds: X, protos: Option<X>) -> Result<Vec<Y>> {
        let protos_ref = protos.as_ref();

        // postprocess
        let ys: Vec<Y> = preds
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, pred)| {
                let mut y = Y::default();

                // parse preds
                let (
                    slice_bboxes,
                    slice_id,
                    slice_clss,
                    slice_confs,
                    slice_kpts,
                    slice_coefs,
                    slice_radians,
                ) = self.layout.parse_preds(pred, self.nc);

                // ImageClassifcation
                if let Task::ImageClassification = self.task {
                    let x = if self.layout.apply_softmax {
                        let exps = slice_clss.mapv(|x| x.exp());
                        let stds = exps.sum_axis(Axis(0));
                        exps / stds
                    } else {
                        slice_clss.into_owned()
                    };
                    let probs = Prob::new_probs(
                        &x.into_raw_vec_and_offset().0,
                        Some(&self.names.iter().map(|x| x.as_str()).collect::<Vec<_>>()),
                        self.topk,
                    );
                    return Some(y.with_probs(&probs));
                }

                // Original image size
                let (image_height, image_width) = (
                    self.processor.images_transform_info()[idx].height_src,
                    self.processor.images_transform_info()[idx].width_src,
                );
                let ratio = self.processor.images_transform_info()[idx].height_scale;

                // ObjectDetection
                let (y_hbbs, y_obbs) = slice_bboxes?
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, bbox)| {
                        // confidence & class_id
                        let (class_id, confidence) = match &slice_id {
                            Some(ids) => (ids[[i, 0]] as usize, slice_clss[[i, 0]]),
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

                        // filter out class id
                        if !self.classes_excluded.is_empty()
                            && self.classes_excluded.contains(&class_id)
                        {
                            return None;
                        }

                        // filter by class id
                        if !self.classes_retained.is_empty()
                            && !self.classes_retained.contains(&class_id)
                        {
                            return None;
                        }

                        // filter by conf
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

                        let (y_hbb, y_obb) = match &slice_radians {
                            Some(slice_radians) => {
                                let radians = slice_radians[[i, 0]];
                                let (w, h, radians) = if w > h {
                                    (w, h, radians)
                                } else {
                                    (h, w, radians + std::f32::consts::PI / 2.)
                                };
                                let radians = radians % std::f32::consts::PI;
                                let mut obb = Obb::from_cxcywhr(cx, cy, w, h, radians)
                                    .with_confidence(confidence)
                                    .with_id(class_id);
                                if !self.names.is_empty() {
                                    obb = obb.with_name(&self.names[class_id]);
                                }

                                (None, Some(obb))
                            }
                            None => {
                                let mut hbb = Hbb::default()
                                    .with_xywh(x, y, w, h)
                                    .with_confidence(confidence)
                                    .with_id(class_id)
                                    .with_uid(i);
                                if !self.names.is_empty() {
                                    hbb = hbb.with_name(&self.names[class_id]);
                                }

                                (Some(hbb), None)
                            }
                        };

                        Some((y_hbb, y_obb))
                    })
                    .collect::<(Vec<_>, Vec<_>)>();

                let mut y_hbbs: Vec<Hbb> = y_hbbs.into_iter().flatten().collect();
                let mut y_obbs: Vec<Obb> = y_obbs.into_iter().flatten().collect();

                // Mbrs
                if !y_obbs.is_empty() {
                    if self.layout.apply_nms {
                        y_obbs.apply_nms_inplace(self.iou);
                    }
                    y = y.with_obbs(&y_obbs);
                    return Some(y);
                }

                // Bboxes
                if !y_hbbs.is_empty() {
                    if self.layout.apply_nms {
                        y_hbbs.apply_nms_inplace(self.iou);
                    }
                    y = y.with_hbbs(&y_hbbs);
                }

                // KeypointsDetection
                if let Some(pred_kpts) = slice_kpts {
                    let kpt_step = self.layout.kpt_step().unwrap_or(3);
                    if !y.hbbs().is_empty() {
                        let y_kpts = y
                            .hbbs()
                            .into_par_iter()
                            .filter_map(|hbb| {
                                let pred = pred_kpts.slice(s![hbb.uid(), ..]);
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
                                                .with_id(i)
                                                .with_confidence(kconf)
                                                .with_xy(
                                                    kx.max(0.0f32).min(image_width as f32),
                                                    ky.max(0.0f32).min(image_height as f32),
                                                );
                                            if !self.names_kpt.is_empty() {
                                                kpt = kpt.with_name(&self.names_kpt[i]);
                                            }
                                            kpt
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                Some(kpts)
                            })
                            .collect::<Vec<_>>();
                        y = y.with_keypointss(&y_kpts);
                    }
                }

                // InstanceSegmentation
                if let Some(coefs) = slice_coefs {
                    if !y.hbbs().is_empty() {
                        let y_masks = y
                            .hbbs()
                            .into_par_iter()
                            .filter_map(|hbb| {
                                let coefs = coefs.slice(s![hbb.uid(), ..]).to_vec();
                                let proto = protos_ref?.slice(s![idx, .., .., ..]);
                                let (nm, mh, mw) = proto.dim();

                                // coefs * proto => mask
                                let coefs = Array::from_shape_vec((1, nm), coefs).ok()?;
                                let proto = proto.to_shape((nm, mh * mw)).ok()?;
                                let mask = coefs.dot(&proto);

                                // Resize mask from (mh, mw) to (image_height, image_width)
                                let mask_resized = Ops::resize_lumaf32_u8(
                                    &mask.into_raw_vec_and_offset().0,
                                    mw as _,
                                    mh as _,
                                    image_width as _,
                                    image_height as _,
                                    true,
                                    "Bilinear",
                                )
                                .ok()?;

                                // Create image buffer from resized mask
                                let mut mask_image: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                                    image::ImageBuffer::from_raw(
                                        image_width as _,
                                        image_height as _,
                                        mask_resized,
                                    )?;

                                // Crop mask using bounding box coordinates
                                let (xmin, ymin, xmax, ymax) =
                                    (hbb.xmin(), hbb.ymin(), hbb.xmax(), hbb.ymax());

                                for (y, row) in mask_image.enumerate_rows_mut() {
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

                                // Create Mask object with metadata
                                let mut mask = Mask::default().with_mask(mask_image);
                                if let Some(id) = hbb.id() {
                                    mask = mask.with_id(id);
                                }
                                if let Some(name) = hbb.name() {
                                    mask = mask.with_name(name);
                                }
                                if let Some(confidence) = hbb.confidence() {
                                    mask = mask.with_confidence(confidence);
                                }

                                Some(mask)
                            })
                            .collect::<Vec<_>>();

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

    /// Extracts class names from the ONNX model metadata if available.
    fn fetch_names_from_onnx(engine: &Engine) -> Option<Vec<String>> {
        // fetch class names from onnx metadata
        // String format: `{0: 'person', 1: 'bicycle', 2: 'sports ball', ..., 27: "yellow_lady's_slipper"}`
        Regex::new(r#"(['"])([->()\w '"]+)(['"])"#)
            .ok()?
            .captures_iter(&engine.try_fetch("names")?)
            .filter_map(|caps| caps.get(2).map(|m| m.as_str().to_string()))
            .collect::<Vec<_>>()
            .into()
    }

    /// Extracts the number of keypoints from the ONNX model metadata if available.
    fn fetch_nk_from_onnx(engine: &Engine) -> Option<usize> {
        Regex::new(r"(\d+), \d+")
            .ok()?
            .captures(&engine.try_fetch("kpt_shape")?)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse::<usize>().ok())
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn processor_mut(&mut self) -> &mut ImageProcessor {
        &mut self.processor
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn dims(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn set_nc(&mut self, nc: usize) {
        self.nc = nc;
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn names_mut(&mut self) -> &mut Vec<String> {
        &mut self.names
    }
}
