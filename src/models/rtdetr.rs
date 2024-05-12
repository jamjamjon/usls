use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis, IxDyn};
use regex::Regex;

use crate::{ops, Bbox, DynConf, MinOptMax, Options, OrtEngine, Y};

#[derive(Debug)]
pub struct RTDETR {
    engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    confs: DynConf,
    nc: usize,
    names: Option<Vec<String>>,
}

impl RTDETR {
    pub fn new(options: Options) -> Result<Self> {
        let mut engine = OrtEngine::new(&options)?;
        let (batch, height, width) = (
            engine.inputs_minoptmax()[0][0].to_owned(),
            engine.inputs_minoptmax()[0][2].to_owned(),
            engine.inputs_minoptmax()[0][3].to_owned(),
        );
        let names: Option<_> = match options.names {
            None => engine.try_fetch("names").map(|names| {
                let re = Regex::new(r#"(['"])([-()\w '"]+)(['"])"#).unwrap();
                let mut names_ = vec![];
                for (_, [_, name, _]) in re.captures_iter(&names).map(|x| x.extract()) {
                    names_.push(name.to_string());
                }
                names_
            }),
            Some(names) => Some(names.to_owned()),
        };
        let nc = options.nc.unwrap_or(
            names
                .as_ref()
                .expect("Failed to get num_classes, make it explicit with `--nc`")
                .len(),
        );
        let confs = DynConf::new(&options.confs, nc);
        engine.dry_run()?;

        Ok(Self {
            engine,
            confs,
            nc,
            height,
            width,
            batch,
            names,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Y>> {
        let xs_ = ops::letterbox(
            xs,
            self.height() as u32,
            self.width() as u32,
            "catmullRom",
            Some(114),
        )?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let ys = self.engine.run(&[xs_])?;
        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Vec<Array<f32, IxDyn>>, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        const CXYWH_OFFSET: usize = 4; // cxcywh
        let preds = &xs[0];

        let mut ys = Vec::new();
        for (idx, anchor) in preds.axis_iter(Axis(0)).enumerate() {
            // [bs, num_query, 4 + nc]
            let width_original = xs0[idx].width() as f32;
            let height_original = xs0[idx].height() as f32;
            let ratio =
                (self.width() as f32 / width_original).min(self.height() as f32 / height_original);

            // save each result
            let mut y_bboxes = Vec::new();
            for pred in anchor.axis_iter(Axis(0)) {
                let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc]);

                // confidence & id
                let (id, &confidence) = clss
                    .into_iter()
                    .enumerate()
                    .reduce(|max, x| if x.1 > max.1 { x } else { max })
                    .unwrap();

                // confs filter
                if confidence < self.confs[id] {
                    continue;
                }

                // bbox -> input size scale -> rescale
                let x = (bbox[0] - bbox[2] / 2.) * self.width() as f32 / ratio;
                let y = (bbox[1] - bbox[3] / 2.) * self.height() as f32 / ratio;
                let w = bbox[2] * self.width() as f32 / ratio;
                let h = bbox[3] * self.height() as f32 / ratio;
                y_bboxes.push(
                    Bbox::default()
                        .with_xywh(
                            x.max(0.0f32).min(width_original),
                            y.max(0.0f32).min(height_original),
                            w,
                            h,
                        )
                        .with_confidence(confidence)
                        .with_id(id as isize)
                        .with_name(self.names.as_ref().map(|names| names[id].to_owned())),
                )
            }
            ys.push(Y::default().with_bboxes(&y_bboxes));
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
