use anyhow::Result;
use clap::ValueEnum;
use image::{DynamicImage, ImageBuffer};
use ndarray::{s, Array, Axis};
use regex::Regex;

use crate::{
    Bbox, DynConf, Keypoint, Mbr, MinOptMax, Ops, Options, OrtEngine, Polygon, Prob, X, Y,
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

impl YOLO {
    pub fn new(options: Options) -> Result<Self> {
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

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Y>> {
        // ----
        // speed test
        for _ in 0..10 {
            // let t = std::time::Instant::now();
            // // way 1
            // let _ = Ops::apply(&[
            //     Ops::Letterbox(
            //         xs,
            //         self.height() as u32,
            //         self.width() as u32,
            //         "catmullRom",
            //         114,
            //     ),
            //     Ops::Normalize(0., 255.),
            //     Ops::Nhwc2nchw,
            // ])?;
            // println!("> {:?}", std::time::Instant::now() - t);

            // way 2
            let t = std::time::Instant::now();
            let _xs = X::apply(&[
                Ops::Letterbox(
                    xs,
                    self.height() as u32,
                    self.width() as u32,
                    "catmullRom",
                    114,
                ),
                Ops::Normalize(0., 255.),
                Ops::Nhwc2nchw,
                Ops::InsertAxis(5),
            ])?;
            println!("{:?}", _xs.shape());
            println!(">> {:?}", std::time::Instant::now() - t);

            //way 3
            let t = std::time::Instant::now();
            let _xs = X::letterbox(
                xs,
                self.height() as u32,
                self.width() as u32,
                "catmullRom",
                114,
            )?
            .normalize(0., 255.)?
            .nhwc2nchw()?;
            println!(">>> {:?}", std::time::Instant::now() - t);
        }
        // ----

        let xs_ = match self.task {
            YOLOTask::Classify => {
                X::resize(xs, self.height() as u32, self.width() as u32, "bilinear")?
                    .normalize(0., 255.)?
                    .nhwc2nchw()?
            }
            _ => X::apply(&[
                Ops::Letterbox(
                    xs,
                    self.height() as u32,
                    self.width() as u32,
                    "catmullRom",
                    114,
                ),
                Ops::Normalize(0., 255.),
                Ops::Nhwc2nchw,
            ])?,
        };
        let ys = self.engine.run(vec![xs_])?;
        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Vec<X>, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        let mut ys = Vec::new();
        let protos = if xs.len() == 2 { Some(&xs[1]) } else { None };
        for (idx, preds) in xs[0].axis_iter(Axis(0)).enumerate() {
            let image_width = xs0[idx].width() as f32;
            let image_height = xs0[idx].height() as f32;

            // decode
            match self.task {
                YOLOTask::Classify => {
                    let y = if self.apply_probs_softmax {
                        let exps = preds.mapv(|x| x.exp());
                        let stds = exps.sum_axis(Axis(0));
                        exps / stds
                    } else {
                        preds.into_owned()
                    };
                    ys.push(
                        Y::default().with_probs(
                            Prob::default()
                                .with_probs(&y.into_raw_vec())
                                .with_names(self.names.to_owned()),
                        ),
                    );
                }
                YOLOTask::Obb => {
                    let mut y_mbrs: Vec<Mbr> = Vec::new();
                    let ratio = (self.width() as f32 / image_width)
                        .min(self.height() as f32 / image_height);
                    for pred in preds.axis_iter(if self.anchors_first { Axis(0) } else { Axis(1) })
                    {
                        // xywhclsr
                        let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc]);
                        let radians = pred[pred.len() - 1];
                        let (id, &confidence) = clss
                            .into_iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap();
                        if confidence < self.confs[id] {
                            continue;
                        }

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
                        y_mbrs.push(
                            Mbr::from_cxcywhr(
                                cx as f64,
                                cy as f64,
                                w as f64,
                                h as f64,
                                radians as f64,
                            )
                            .with_confidence(confidence)
                            .with_id(id as isize)
                            .with_name(self.names.as_ref().map(|names| names[id].to_owned())),
                        );
                    }
                    ys.push(Y::default().with_mbrs(&y_mbrs).apply_mbrs_nms(self.iou));
                }
                _ => {
                    let mut y_bboxes: Vec<Bbox> = Vec::new();
                    let ratio = (self.width() as f32 / image_width)
                        .min(self.height() as f32 / image_height);

                    // bboxes
                    for (i, pred) in preds
                        .axis_iter(if self.anchors_first { Axis(0) } else { Axis(1) })
                        .enumerate()
                    {
                        match self.version {
                            YOLOVersion::V10 => {
                                let class_id = pred[CXYWH_OFFSET + 1] as usize;
                                let confidence = pred[CXYWH_OFFSET];
                                if confidence < self.confs[class_id] {
                                    continue;
                                }
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
                                    .with_id(class_id as isize)
                                    .with_id_born(i as isize)
                                    .with_name(
                                        self.names.as_ref().map(|names| names[class_id].to_owned()),
                                    );
                                y_bboxes.push(y_bbox);
                            }
                            _ => {
                                let (conf_, clss) = if self.conf_independent {
                                    (
                                        pred[CXYWH_OFFSET],
                                        pred.slice(
                                            s![CXYWH_OFFSET + 1..CXYWH_OFFSET + self.nc + 1],
                                        ),
                                    )
                                } else {
                                    (1.0, pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc]))
                                };
                                let (id, &confidence) = clss
                                    .into_iter()
                                    .enumerate()
                                    .max_by(|a, b| a.1.total_cmp(b.1))
                                    .unwrap();
                                let confidence = confidence * conf_;
                                if confidence < self.confs[id] {
                                    continue;
                                }
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
                                        self.names.as_ref().map(|names| names[id].to_owned()),
                                    );
                                y_bboxes.push(y_bbox);
                            }
                        }
                    }

                    // nms
                    let mut y = Y::default().with_bboxes(&y_bboxes);
                    if self.apply_nms {
                        y = y.apply_bboxes_nms(self.iou);
                    }

                    // keypoints
                    if let YOLOTask::Pose = self.task {
                        if let Some(bboxes) = y.bboxes() {
                            let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
                            for bbox in bboxes.iter() {
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

                                let mut kpts_: Vec<Keypoint> = Vec::new();
                                for i in 0..self.nk {
                                    let kx = pred[KPT_STEP * i] / ratio;
                                    let ky = pred[KPT_STEP * i + 1] / ratio;
                                    let kconf = pred[KPT_STEP * i + 2];
                                    if kconf < self.kconfs[i] {
                                        kpts_.push(Keypoint::default());
                                    } else {
                                        kpts_.push(
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
                                                ),
                                        );
                                    }
                                }
                                y_kpts.push(kpts_);
                            }
                            y = y.with_keypoints(&y_kpts);
                        }
                    }

                    // masks
                    if let YOLOTask::Segment = self.task {
                        if let Some(bboxes) = y.bboxes() {
                            let mut y_polygons: Vec<Polygon> = Vec::new();
                            for bbox in bboxes.iter() {
                                let coefs = if self.anchors_first {
                                    preds
                                        .slice(s![bbox.id_born(), preds.shape()[1] - self.nm..])
                                        .to_vec()
                                } else {
                                    preds
                                        .slice(s![preds.shape()[0] - self.nm.., bbox.id_born()])
                                        .to_vec()
                                };
                                let proto = protos.unwrap().slice(s![idx, .., .., ..]);

                                // coefs * proto -> mask
                                let (nm, nh, nw) = proto.dim();
                                let coefs = Array::from_shape_vec((1, nm), coefs)?; // (n, nm)
                                let proto = proto.to_owned().into_shape((nm, nh * nw))?; // (nm, nh*nw)
                                let mask = coefs.dot(&proto).into_shape((nh, nw, 1))?; // (nh, nw, n)

                                // build image from ndarray
                                let mask: ImageBuffer<image::Luma<_>, Vec<f32>> =
                                    match ImageBuffer::from_raw(
                                        nw as u32,
                                        nh as u32,
                                        mask.clone().into_raw_vec(),
                                    ) {
                                        Some(buf) => buf,
                                        None => continue,
                                    };
                                let mask = image::DynamicImage::from(mask);

                                // rescale
                                let mask_original = Ops::descale_mask(
                                    mask,
                                    nw as f32,
                                    nh as f32,
                                    image_width,
                                    image_height,
                                );
                                let mut mask_original = mask_original.into_luma8();

                                // crop mask
                                for y in 0..image_height as usize {
                                    for x in 0..image_width as usize {
                                        if x < bbox.xmin() as usize
                                            || x > bbox.xmax() as usize
                                            || y < bbox.ymin() as usize
                                            || y > bbox.ymax() as usize
                                        // || mask_original.get_pixel(x as u32, y as u32).0 < [127]
                                        {
                                            mask_original.put_pixel(
                                                x as u32,
                                                y as u32,
                                                image::Luma([0u8]),
                                            );
                                        }
                                    }
                                }

                                // get masks from image
                                let contours: Vec<imageproc::contours::Contour<i32>> =
                                    imageproc::contours::find_contours_with_threshold(
                                        &mask_original,
                                        0,
                                    );
                                let polygon = match contours
                                    .iter()
                                    .map(|x| {
                                        Polygon::default()
                                            .with_id(bbox.id())
                                            .with_points_imageproc(&x.points)
                                            .with_name(bbox.name().cloned())
                                    })
                                    .max_by(|x, y| x.area().total_cmp(&y.area()))
                                {
                                    None => continue,
                                    Some(x) => x,
                                };
                                y_polygons.push(polygon);
                            }
                            y = y.with_polygons(&y_polygons);
                        }
                    }
                    ys.push(y);
                }
            }
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
