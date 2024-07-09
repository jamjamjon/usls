use ndarray::{ArrayBase, ArrayView, Axis, Dim, IxDyn, IxDynImpl, ViewRepr};

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum YOLOTask {
    Classify,
    Detect,
    Pose,
    Segment,
    Obb,
}

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum YOLOVersion {
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
    // YOLOX,
    // YOLOv3,
    // YOLOv4,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoxType {
    /// 1
    Cxcywh,

    /// 2 Cxcybr
    Cxcyxy,

    /// 3 Tlbr
    Xyxy,

    /// 4  Tlwh
    Xywh,

    /// 5  Tlcxcy
    XyCxcy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClssType {
    Clss,
    ConfCls,
    ClsConf,
    ConfClss,
    ClssConf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KptsType {
    Xys,
    Xycs,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnchorsPosition {
    Before,
    After,
}

#[derive(Debug, Clone, PartialEq)]
pub struct YOLOPreds {
    pub clss: ClssType,
    pub bbox: Option<BoxType>,
    pub kpts: Option<KptsType>,
    pub coefs: Option<bool>,
    pub obb: Option<bool>,
    pub anchors: Option<AnchorsPosition>,
    pub is_bbox_normalized: bool,
    pub apply_nms: bool,
    pub apply_softmax: bool,
}

impl Default for YOLOPreds {
    fn default() -> Self {
        Self {
            clss: ClssType::Clss,
            bbox: None,
            kpts: None,
            coefs: None,
            obb: None,
            anchors: None,
            is_bbox_normalized: false,
            apply_nms: true,
            apply_softmax: false,
        }
    }
}

impl YOLOPreds {
    pub fn apply_nms(mut self, x: bool) -> Self {
        self.apply_nms = x;
        self
    }

    pub fn apply_softmax(mut self, x: bool) -> Self {
        self.apply_softmax = x;
        self
    }

    pub fn n_clss() -> Self {
        // Classification: NClss
        Self {
            clss: ClssType::Clss,
            ..Default::default()
        }
    }

    pub fn n_a_cxcywh_confclss() -> Self {
        // YOLOv5 | YOLOv6 | YOLOv7 | YOLOX : NACxcywhConfClss
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::ConfClss,
            anchors: Some(AnchorsPosition::Before),
            ..Default::default()
        }
    }

    pub fn n_a_cxcywh_confclss_coefs() -> Self {
        // YOLOv5 Segment : NACxcywhConfClssCoefs
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::ConfClss,
            coefs: Some(true),
            anchors: Some(AnchorsPosition::Before),
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_a() -> Self {
        // YOLOv8 | YOLOv9 : NCxcywhClssA
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    }

    pub fn n_a_xyxy_confcls() -> Self {
        // YOLOv10 : NAXyxyConfCls
        Self {
            bbox: Some(BoxType::Xyxy),
            clss: ClssType::ConfCls,
            anchors: Some(AnchorsPosition::Before),
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_a_n() -> Self {
        // RTDETR
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            anchors: Some(AnchorsPosition::After),
            is_bbox_normalized: true,
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_xycs_a() -> Self {
        // YOLOv8 Pose : NCxcywhClssXycsA
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            kpts: Some(KptsType::Xycs),
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_coefs_a() -> Self {
        // YOLOv8 Segment : NCxcywhClssCoefsA
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            coefs: Some(true),
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_r_a() -> Self {
        // YOLOv8 Obb : NCxcywhClssRA
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            obb: Some(true),
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    }

    pub fn task(&self) -> YOLOTask {
        match self.obb {
            Some(_) => YOLOTask::Obb,
            None => match self.coefs {
                Some(_) => YOLOTask::Segment,
                None => match self.kpts {
                    Some(_) => YOLOTask::Pose,
                    None => match self.bbox {
                        Some(_) => YOLOTask::Detect,
                        None => YOLOTask::Classify,
                    },
                },
            },
        }
    }

    pub fn is_anchors_first(&self) -> bool {
        matches!(self.anchors, Some(AnchorsPosition::Before))
    }

    pub fn is_cls_type(&self) -> bool {
        matches!(self.clss, ClssType::ClsConf | ClssType::ConfCls)
    }

    pub fn is_clss_type(&self) -> bool {
        matches!(
            self.clss,
            ClssType::ClssConf | ClssType::ConfClss | ClssType::Clss
        )
    }

    pub fn is_conf_at_end(&self) -> bool {
        matches!(self.clss, ClssType::ClssConf | ClssType::ClsConf)
    }

    pub fn is_conf_independent(&self) -> bool {
        !matches!(self.clss, ClssType::Clss)
    }

    pub fn kpt_step(&self) -> Option<usize> {
        match &self.kpts {
            Some(x) => match x {
                KptsType::Xycs => Some(3),
                KptsType::Xys => Some(2),
            },
            None => None,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn parse_preds<'a>(
        &'a self,
        preds: ArrayBase<ViewRepr<&'a f32>, Dim<IxDynImpl>>,
        nc: usize,
    ) -> (
        ArrayView<f32, IxDyn>,
        Option<ArrayView<f32, IxDyn>>,
        ArrayView<f32, IxDyn>,
        Option<ArrayView<f32, IxDyn>>,
        Option<ArrayView<f32, IxDyn>>,
        Option<ArrayView<f32, IxDyn>>,
        Option<ArrayView<f32, IxDyn>>,
    ) {
        let preds = if self.is_anchors_first() {
            preds
        } else {
            preds.reversed_axes()
        };

        // get each tasks slices
        let (slice_bboxes, xs) = preds.split_at(Axis(1), 4);
        let (slice_id, slice_clss, slice_confs, xs) = match self.clss {
            ClssType::ConfClss => {
                let slice_id = None;
                let (confs, xs) = xs.split_at(Axis(1), 1);
                let (clss, xs) = xs.split_at(Axis(1), nc);
                // let confs = confs.broadcast((confs.shape()[0], nc)).unwrap(); // 267ns
                // let clss = &confs * &clss;
                // let slice_clss = clss.to_owned();
                let slice_clss = clss;
                let slice_confs = Some(confs);
                (slice_id, slice_clss, slice_confs, xs)
            }
            ClssType::ClssConf => {
                let slice_id = None;
                let (clss, xs) = xs.split_at(Axis(1), nc);
                let (confs, xs) = xs.split_at(Axis(1), 1);
                // let confs = confs.broadcast((confs.shape()[0], nc)).unwrap();
                // TODO: par
                // let clss = &confs * &clss;
                // let slice_clss = clss;
                // let slice_clss = clss.to_owned();
                let slice_clss = clss;
                let slice_confs = Some(confs);
                // (slice_id, slice_clss, xs)
                (slice_id, slice_clss, slice_confs, xs)
            }
            ClssType::ConfCls => {
                let (clss, xs) = xs.split_at(Axis(1), 1);
                let (ids, xs) = xs.split_at(Axis(1), 1);
                let slice_id = Some(ids);
                // let slice_clss = clss.to_owned();
                let slice_clss = clss;
                let slice_confs = None;
                (slice_id, slice_clss, slice_confs, xs)
            }
            ClssType::ClsConf => {
                let (ids, xs) = xs.split_at(Axis(1), 1);
                let (clss, xs) = xs.split_at(Axis(1), 1);
                let slice_id = Some(ids);
                let slice_clss = clss;
                // let slice_clss = clss.to_owned();
                // (slice_id, slice_clss, xs)
                let slice_confs = None;
                (slice_id, slice_clss, slice_confs, xs)
            }
            ClssType::Clss => {
                let slice_id = None;
                let (clss, xs) = xs.split_at(Axis(1), nc);
                // let slice_clss = clss.to_owned();
                let slice_clss = clss;
                // (slice_id, slice_clss, xs)
                let slice_confs = None;
                (slice_id, slice_clss, slice_confs, xs)
            }
        };
        let (slice_kpts, slice_coefs, slice_radians) = match self.task() {
            YOLOTask::Pose => (Some(xs), None, None),
            YOLOTask::Segment => (None, Some(xs), None),
            YOLOTask::Obb => (None, None, Some(xs)),
            _ => (None, None, None),
        };

        (
            slice_bboxes,
            slice_id,
            slice_clss,
            slice_confs,
            slice_kpts,
            slice_coefs,
            slice_radians,
        )
    }
}
