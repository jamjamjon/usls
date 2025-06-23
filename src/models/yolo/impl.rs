use aksr::Builder;
use anyhow::Result;
use log::{error, info};
use rayon::prelude::*;
use regex::Regex;

use crate::{
    elapsed_module,
    models::{BoxType, YOLOPredsFormat},
    Config, DynConf, Engine, Hbb, Image, Keypoint, Mask, NmsOps, Obb, Ops, Prob, Processor, Task,
    Version, Xs, Y,
};

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
    layout: YOLOPredsFormat,
    task: Task,
    version: Option<Version>,
    names: Vec<String>,
    names_kpt: Vec<String>,
    nc: usize,
    nk: usize,
    confs: DynConf,
    kconfs: DynConf,
    iou: f32,
    topk: usize,
    processor: Processor,
    spec: String,
    classes_excluded: Vec<usize>,
    classes_retained: Vec<usize>,
}

impl TryFrom<Config> for YOLO {
    type Error = anyhow::Error;

    fn try_from(config: Config) -> Result<Self, Self::Error> {
        Self::new(config)
    }
}

impl YOLO {
    /// Creates a new YOLO model instance from the provided configuration.
    ///
    /// This function initializes the model by:
    /// - Loading the ONNX model file
    /// - Setting up input dimensions and processing parameters
    /// - Configuring task-specific settings and output formats
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width, spec) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
            engine.spec().to_owned(),
        );
        let task: Option<Task> = match &config.task {
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
        let version = config.version;
        let (layout, task) = match &config.yolo_preds_format {
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
                        (Task::ImageClassification, Version(8, 0, _) | Version(11, 0, _)) => {
                            YOLOPredsFormat::n_clss()
                        }
                        (
                            Task::ObjectDetection,
                            Version(5, 0, _) | Version(6, 0, _) | Version(7, 0, _),
                        ) => YOLOPredsFormat::n_a_cxcywh_confclss(),
                        (
                            Task::ObjectDetection,
                            Version(8, 0, _)
                            | Version(9, 0, _)
                            | Version(11, 0, _)
                            | Version(12, 0, _),
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
                            Version(8, 0, _) | Version(9, 0, _) | Version(11, 0, _),
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
                        Version(9, 0, _) | Version(12, 0, _) => YOLOPredsFormat::n_cxcywh_clss_a(),
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
        let names_customized = config.class_names.to_vec();
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
        let nc = match config.nc() {
            None => names.len(),
            Some(n) => {
                if names.len() != n {
                    anyhow::bail!(
                        "The lengths of class names: {} and user-defined num_classes: {} do not match.",
                        names.len(),
                        n,
                    )
                }
                n
            }
        };
        if nc == 0 && names.is_empty() {
            anyhow::bail!(
                    "Neither class names nor the number of classes were specified. \
                    \nConsider specify them with `Config::default().with_nc()` or `Config::default().with_class_names()`"
                );
        }

        // Keypoint names & Number of keypoints
        let names_kpt = config.keypoint_names.to_vec();
        let nk = if let Task::KeypointsDetection = task {
            match (names_kpt.is_empty(),  Self::fetch_nk_from_onnx(&engine).or(config.nk())) {
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

        // Attributes
        let topk = config.topk().unwrap_or(5);
        let confs = DynConf::new_or_default(config.class_confs(), nc);
        let kconfs = DynConf::new_or_default(config.keypoint_confs(), nk);
        let iou = config.iou().unwrap_or(0.45);
        let classes_excluded = config.classes_excluded().to_vec();
        let classes_retained = config.classes_retained().to_vec();
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
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        info!("{}", info);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            task,
            version,
            spec,
            layout,
            names,
            names_kpt,
            confs,
            kconfs,
            iou,
            nc,
            nk,
            classes_excluded,
            classes_retained,
            processor,
            topk,
        })
    }

    /// Pre-processes input images before model inference.
    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let x = self.processor.process_images(xs)?;

        Ok(x.into())
    }

    /// Performs model inference on the pre-processed input.
    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    /// Performs the complete inference pipeline on a batch of images.
    ///
    /// The pipeline consists of:
    /// 1. Pre-processing the input images
    /// 2. Running model inference
    /// 3. Post-processing the outputs to generate final predictions
    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        // Forward pass
        let ys = elapsed_module!("yolo", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("yolo", "inference", self.inference(ys)?);
        let ys = elapsed_module!("yolo", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    /// Post-processes model outputs to generate final predictions.
    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .iter_dim(0)
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, preds)| {
                let mut y = Y::default();

                let t = std::time::Instant::now();
                // Parse predictions
                let (
                    slice_bboxes,
                    slice_id,
                    slice_clss,
                    slice_confs,
                    slice_kpts,
                    slice_coefs,
                    slice_radians,
                ) = self.layout.parse_preds(preds.view(), self.nc).ok()?;
                let t = t.elapsed();
                println!("parse preds: {:?}", t);

                // ImageClassifcation
                if let Task::ImageClassification = self.task {
                    let x = if self.layout.apply_softmax {
                        let exps = slice_clss.exp().ok()?;
                        let stds = exps.view().sum(Some(0), false).ok()?;
                        (exps / &stds).ok()?
                    } else {
                        slice_clss.to_owned()
                    };
                    let data = x.to_vec().ok()?;
                    let probs = Prob::new_probs(
                        &data,
                        Some(&self.names.iter().map(|x| x.as_str()).collect::<Vec<_>>()),
                        self.topk,
                    );

                    return Some(y.with_probs(&probs));
                }

                // Original image size
                let (image_height, image_width) = (
                    self.processor.images_transform_info[idx].height_src,
                    self.processor.images_transform_info[idx].width_src,
                );
                let ratio = self.processor.images_transform_info[idx].height_scale;

                // Other tasks
                let (y_hbbs, y_obbs) = slice_bboxes?
                    .to_owned()
                    .iter_dim(0)
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, hbb)| {
                        // confidence & class_id
                        let (class_id, confidence) = match &slice_id {
                            Some(ids) => {
                                let id_val = ids
                                    .slice((i, 0))
                                    .ok()?
                                    .to_owned()
                                    .ok()?
                                    .get_element::<f32>(0)
                                    .ok()? as usize;
                                let cls_val = slice_clss
                                    .slice((i, 0))
                                    .ok()?
                                    .to_owned()
                                    .ok()?
                                    .get_element::<f32>(0)
                                    .ok()?;
                                (id_val, cls_val)
                            }
                            None => {
                                let slice_row = slice_clss.slice((i..i + 1, ..)).ok()?;
                                let data: Vec<f32> = slice_row.to_vec().ok()?;
                                let (class_id, &confidence) =
                                    data.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;

                                match &slice_confs {
                                    None => (class_id, confidence),
                                    Some(slice_confs) => {
                                        let conf_val = slice_confs
                                            .slice((i, 0))
                                            .ok()?
                                            .to_owned()
                                            .ok()?
                                            .get_element::<f32>(0)
                                            .ok()?;
                                        (class_id, confidence * conf_val)
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
                        let hbb_scaled = (hbb / ratio).ok()?;
                        let hbb_data = hbb_scaled.to_vec().ok()?;
                        let hbb = if self.layout.is_bbox_normalized {
                            (
                                hbb_data[0] * self.width() as f32,
                                hbb_data[1] * self.height() as f32,
                                hbb_data[2] * self.width() as f32,
                                hbb_data[3] * self.height() as f32,
                            )
                        } else {
                            (hbb_data[0], hbb_data[1], hbb_data[2], hbb_data[3])
                        };
                        let (cx, cy, x, y, w, h) = match self.layout.box_type()? {
                            BoxType::Cxcywh => {
                                let (cx, cy, w, h): (f32, f32, f32, f32) = hbb;
                                let x = (cx - w / 2.0_f32).max(0.0_f32);
                                let y = (cy - h / 2.0_f32).max(0.0_f32);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Xyxy => {
                                let (x, y, x2, y2) = hbb;
                                let (w, h) = (x2 - x, y2 - y);
                                let (cx, cy) = ((x + x2) / 2., (y + y2) / 2.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Xywh => {
                                let (x, y, w, h) = hbb;
                                let (cx, cy) = (x + w / 2., y + h / 2.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Cxcyxy => {
                                let (cx, cy, x2, y2) = hbb;
                                let (w, h) = ((x2 - cx) * 2., (y2 - cy) * 2.);
                                let x = (x2 - w).max(0.);
                                let y = (y2 - h).max(0.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::XyCxcy => {
                                let (x, y, cx, cy) = hbb;
                                let (w, h) = ((cx - x) * 2., (cy - y) * 2.);
                                (cx, cy, x, y, w, h)
                            }
                        };

                        let (y_hbb, y_obb) = match &slice_radians {
                            Some(slice_radians) => {
                                let radians = slice_radians
                                    .slice((i, 0))
                                    .ok()?
                                    .to_owned()
                                    .ok()?
                                    .get_element::<f32>(0)
                                    .ok()?;
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
                    if let Some(hbbs) = y.hbbs() {
                        let y_kpts = hbbs
                            .into_par_iter()
                            .filter_map(|hbb| {
                                let pred = pred_kpts
                                    .slice(&[hbb.uid()..hbb.uid() + 1, 0..pred_kpts.shape()[1]])
                                    .ok()?;
                                let kpts = (0..self.nk)
                                    .into_par_iter()
                                    .filter_map(|i| {
                                        let kx = pred
                                            .slice((0, kpt_step * i))
                                            .ok()?
                                            .to_owned()
                                            .ok()?
                                            .get_element::<f32>(0)
                                            .ok()?
                                            / ratio;
                                        let ky = pred
                                            .slice((0, kpt_step * i + 1))
                                            .ok()?
                                            .to_owned()
                                            .ok()?
                                            .get_element::<f32>(0)
                                            .ok()?
                                            / ratio;
                                        let kconf = pred
                                            .slice((0, kpt_step * i + 2))
                                            .ok()?
                                            .to_owned()
                                            .ok()?
                                            .get_element::<f32>(0)
                                            .ok()?;
                                        if kconf < self.kconfs[i] {
                                            Some(Keypoint::default())
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
                                            Some(kpt)
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
                    if let Some(hbbs) = y.hbbs() {
                        let protos = &xs[1];
                        let y_masks = hbbs
                            .into_par_iter()
                            .filter_map(|hbb| {
                                let coefs_slice =
                                    coefs.slice((hbb.uid()..hbb.uid() + 1, ..)).ok()?;
                                let coefs: Vec<f32> = coefs_slice.to_vec().ok()?;
                                let proto = protos.slice((idx..idx + 1, .., .., ..)).ok()?;
                                let shape = proto.shape();
                                let (nm, mh, mw) = (shape[1], shape[2], shape[3]);

                                // coefs * proto => mask
                                let coefs_tensor =
                                    crate::Tensor::from_vec(coefs).reshape(vec![1, nm]).ok()?;
                                let proto_reshaped = proto.reshape(vec![nm, mh * mw]).ok()?;
                                // Transpose proto_reshaped to make dimensions compatible: [nm, mh*mw] -> [mh*mw, nm]
                                let proto_transposed = proto_reshaped.transpose().ok()?;
                                let mask = coefs_tensor.dot(&proto_transposed).ok()?;

                                // Mask rescale
                                let mask = Ops::resize_lumaf32_u8(
                                    &mask.to_vec::<f32>().ok()?,
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
                                    (hbb.xmin(), hbb.ymin(), hbb.xmax(), hbb.ymax());

                                // Using hbb to crop the mask
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

                                let mut mask = Mask::default().with_mask(mask);
                                if let Some(id) = hbb.id() {
                                    mask = mask.with_id(id);
                                }
                                if let Some(name) = hbb.name() {
                                    mask = mask.with_name(name);
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
        // Ok(ys.into())
    }

    /// Extracts class names from the ONNX model metadata if available.
    fn fetch_names_from_onnx(engine: &Engine) -> Option<Vec<String>> {
        // fetch class names from onnx metadata
        // String format: `{0: 'person', 1: 'bicycle', 2: 'sports ball', ..., 27: "yellow_lady's_slipper"}`
        Regex::new(r#"(['"])([-()\w '"]+)(['"])"#)
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
}
