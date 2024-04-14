use anyhow::Result;
use clap::ValueEnum;
use image::DynamicImage;
use ndarray::{s, Array, Axis, IxDyn};
use regex::Regex;

use crate::{
    ops, Bbox, DynConf, Embedding, Keypoint, Mask, MinOptMax, Options, OrtEngine, Point, Rect, Ys,
};

const CXYWH_OFFSET: usize = 4;
const KPT_STEP: usize = 3;

#[derive(Debug, Clone, ValueEnum)]
enum YOLOTask {
    Classify,
    Detect,
    Pose,
    Segment,
    Obb, // TODO
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
    confs: DynConf,
    kconfs: DynConf,
    iou: f32,
    names: Option<Vec<String>>,
    names_kpt: Option<Vec<String>>,
    apply_nms: bool,
    anchors_first: bool,
}

impl YOLO {
    pub fn new(options: &Options) -> Result<Self> {
        let engine = OrtEngine::new(options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );
        let task = match engine
            .try_fetch("task")
            .unwrap_or("detect".to_string())
            .as_str()
        {
            "classify" => YOLOTask::Classify,
            "detect" => YOLOTask::Detect,
            "pose" => YOLOTask::Pose,
            "segment" => YOLOTask::Segment,
            x => todo!("{:?} is not supported for now!", x),
        };

        // try from custom class names, and then model metadata
        let mut names = options.names.to_owned().or(Self::fetch_names(&engine));
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

        let names_kpt = options.names2.to_owned().or(None);

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
            apply_nms: options.apply_nms,
            nc,
            nk,
            nm,
            height,
            width,
            batch,
            task,
            names,
            names_kpt,
            anchors_first: options.anchors_first,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Ys>> {
        let xs_ = ops::letterbox(xs, self.height() as u32, self.width() as u32, 144.0)?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let ys = self.engine.run(&[xs_])?;
        let ys = self.postprocess(ys, xs)?;
        Ok(ys)
    }

    pub fn postprocess(&self, xs: Vec<Array<f32, IxDyn>>, xs0: &[DynamicImage]) -> Result<Vec<Ys>> {
        if let YOLOTask::Classify = self.task {
            let mut ys = Vec::new();
            for batch in xs[0].axis_iter(Axis(0)) {
                ys.push(
                    Ys::default()
                        .with_probs(Embedding::new(batch.into_owned(), self.names.to_owned())),
                );
            }
            Ok(ys)
        } else {
            let (preds, protos) = if xs.len() == 2 {
                if xs[0].ndim() == 3 {
                    (&xs[0], Some(&xs[1]))
                } else {
                    (&xs[1], Some(&xs[0]))
                }
            } else {
                (&xs[0], None)
            };

            let mut ys = Vec::new();
            for (idx, anchor) in preds.axis_iter(Axis(0)).enumerate() {
                // [b, 4 + nc + nm, na]
                // input image
                let width_original = xs0[idx].width() as f32;
                let height_original = xs0[idx].height() as f32;
                let ratio = (self.width() as f32 / width_original)
                    .min(self.height() as f32 / height_original);

                #[allow(clippy::type_complexity)]
                let mut data: Vec<(Bbox, Option<Vec<Keypoint>>, Option<Vec<f32>>)> = Vec::new();
                for pred in anchor.axis_iter(if self.anchors_first { Axis(0) } else { Axis(1) }) {
                    // split preds for different tasks
                    let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                    let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc]);
                    let kpts = {
                        if let YOLOTask::Pose = self.task {
                            Some(pred.slice(s![pred.len() - KPT_STEP * self.nk..]))
                        } else {
                            None
                        }
                    };
                    let coefs = {
                        if let YOLOTask::Segment = self.task {
                            Some(pred.slice(s![pred.len() - self.nm..]).to_vec())
                        } else {
                            None
                        }
                    };

                    // confidence and index
                    let (id, &confidence) = clss
                        .into_iter()
                        .enumerate()
                        .reduce(|max, x| if x.1 > max.1 { x } else { max })
                        .unwrap();

                    // confidence filter
                    if confidence < self.confs[id] {
                        continue;
                    }

                    // bbox re-scale
                    let cx = bbox[0] / ratio;
                    let cy = bbox[1] / ratio;
                    let w = bbox[2] / ratio;
                    let h = bbox[3] / ratio;
                    let x = cx - w / 2.;
                    let y = cy - h / 2.;
                    let y_bbox = Bbox::new(
                        Rect::from_xywh(
                            x.max(0.0f32).min(width_original),
                            y.max(0.0f32).min(height_original),
                            w,
                            h,
                        ),
                        id,
                        confidence,
                        self.names.as_ref().map(|names| names[id].to_owned()),
                    );

                    // kpts
                    let y_kpts = {
                        if let Some(kpts) = kpts {
                            let mut kpts_ = Vec::new();
                            for i in 0..self.nk {
                                let kx = kpts[KPT_STEP * i] / ratio;
                                let ky = kpts[KPT_STEP * i + 1] / ratio;
                                let kconf = kpts[KPT_STEP * i + 2];
                                if kconf < self.kconfs[i] {
                                    kpts_.push(Keypoint::default());
                                } else {
                                    kpts_.push(Keypoint::new(
                                        Point::new(
                                            kx.max(0.0f32).min(width_original),
                                            ky.max(0.0f32).min(height_original),
                                        ),
                                        kconf,
                                        i as isize,
                                        self.names_kpt.as_ref().map(|names| names[i].to_owned()),
                                    ));
                                }
                            }
                            Some(kpts_)
                        } else {
                            None
                        }
                    };

                    // merged
                    data.push((y_bbox, y_kpts, coefs));
                }

                // nms
                if self.apply_nms {
                    Self::non_max_suppression(&mut data, self.iou);
                }

                // decode
                let mut y_bboxes: Vec<Bbox> = Vec::new();
                let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
                let mut y_masks: Vec<Mask> = Vec::new();
                for elem in data.into_iter() {
                    if let Some(kpts) = elem.1 {
                        y_kpts.push(kpts)
                    }

                    // decode masks
                    if let Some(coefs) = elem.2 {
                        let proto = protos.unwrap().slice(s![idx, .., .., ..]);
                        let (nm, nh, nw) = proto.dim();

                        // coefs * proto -> mask
                        let coefs = Array::from_shape_vec((1, nm), coefs)?; // (n, nm)
                        let proto = proto.to_owned().into_shape((nm, nh * nw))?; // (nm, nh*nw)
                        let mask = coefs.dot(&proto).into_shape((nh, nw, 1))?; // (nh, nw, n)

                        // build image from ndarray
                        let mask_im = ops::build_dyn_image_from_raw(
                            mask.into_raw_vec(),
                            nw as u32,
                            nh as u32,
                        );

                        // rescale masks
                        let mask_original = ops::descale_mask(
                            mask_im,
                            nw as f32,
                            nh as f32,
                            width_original,
                            height_original,
                        );

                        // crop mask with bbox
                        let mut mask_original = mask_original.into_luma8();
                        for y in 0..height_original as usize {
                            for x in 0..width_original as usize {
                                if x < elem.0.xmin() as usize
                                    || x > elem.0.xmax() as usize
                                    || y < elem.0.ymin() as usize
                                    || y > elem.0.ymax() as usize
                                {
                                    mask_original.put_pixel(x as u32, y as u32, image::Luma([0u8]));
                                }
                            }
                        }

                        // get masks from image
                        let masks = ops::get_masks_from_image(
                            mask_original,
                            1,
                            elem.0.id(),
                            elem.0.name().cloned(),
                        );
                        y_masks.extend(masks);
                    }
                    y_bboxes.push(elem.0);
                }

                // save result
                ys.push(
                    Ys::default()
                        .with_bboxes(&y_bboxes)
                        .with_keypoints(&y_kpts)
                        .with_masks(&y_masks),
                );
            }

            Ok(ys)
        }
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

    pub fn batch(&self) -> isize {
        self.batch.opt
    }

    pub fn width(&self) -> isize {
        self.width.opt
    }

    pub fn height(&self) -> isize {
        self.height.opt
    }

    #[allow(clippy::type_complexity)]
    fn non_max_suppression(
        xs: &mut Vec<(Bbox, Option<Vec<Keypoint>>, Option<Vec<f32>>)>,
        iou_threshold: f32,
    ) {
        xs.sort_by(|b1, b2| b2.0.confidence().partial_cmp(&b1.0.confidence()).unwrap());

        let mut current_index = 0;
        for index in 0..xs.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = xs[prev_index].0.iou(&xs[index].0);
                if iou > iou_threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                xs.swap(current_index, index);
                current_index += 1;
            }
        }
        xs.truncate(current_index);
    }
}
